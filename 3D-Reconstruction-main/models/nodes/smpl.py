from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import random
import logging

import torch
import numpy as np
from torch.nn import Parameter

from models.human_body import (
    phalp_colors,
    SMPLTemplate,
    get_on_mesh_init_geo_values,
    batch_rigid_transform,
    quaternion_to_matrix,
)
from models.gaussians.basics import *
from models.nodes.rigid import RigidNodes
from models.gaussians.vanilla import VanillaGaussians
from pytorch3d.ops import knn_points
from utils.geometry import uniform_sample_sphere, quat_to_rotmat, quaternion_multiply

from utils.geometry import uniform_sample_sphere, quat_to_rotmat, quaternion_multiply

RGB_tuples = torch.tensor(np.vstack([phalp_colors] * 10), dtype=torch.float32) / 255.0
logger = logging.getLogger()


class SMPLNodes(RigidNodes):
    def __init__(self, **kwargs):
        # self.smpl_points_num = 527240
        self.smpl_points_num = 6890
        super().__init__(**kwargs)

        self.use_voxel_deformer = self.ctrl_cfg.use_voxel_deformer
        # overide here, because we use only one dimension for scale
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)

    @property
    def num_instances(self):
        return self.instances_fv.shape[1]

    @property
    def num_frames(self):
        return self.instances_fv.shape[0]

    def create_from_pcd(self, instance_pts_dict: Dict[str, torch.Tensor]) -> None:
        """
        instance_pts_dict: {
            id in dataset: {
                "class_name": str,
                "pts": torch.Tensor, (N, 3)
                "colors": torch.Tensor, (N, 3)
                "poses": torch.Tensor, (num_frame, 4, 4)
                "size": torch.Tensor, (3, )
                "frame_info": torch.Tensor, (num_frame)
                "num_pts": int,
            },
        }
        """
        """
        For version 1:
            we simplified gaussian properties:
            - means: (Frame_N, N, 3), Not Optimized
            - scales: (N, 1), Optimized, we use gaussians with the same scale on xyz
            - quats: (N, 4), Not Optimized
            - features_dc: (N, 3), Optimized
            - features_rest: (N, num_sh_bases, 3), Optimized
            - opacities: (N, 1), Optimized
        """
        # collect all instances
        smpl_betas, smpl_qauts = [], []
        instances_quats, instances_trans, instances_size = [], [], []
        instances_fv, point_ids = [], []
        # instances_pts, instances_colors = [], []
        for id_in_model, (id_in_dataset, v) in enumerate(instance_pts_dict.items()):
            smpl_qauts.append(v["smpl_quats"][:, 1:, :].unsqueeze(1))
            instances_quats.append(v["smpl_quats"][:, 0, :].unsqueeze(1))
            instances_trans.append(v["smpl_trans"].unsqueeze(1))
            instances_fv.append(v["frame_info"].unsqueeze(1))
            smpl_betas.append(v["smpl_betas"].unsqueeze(0))
            instances_size.append(v["size"])
            # instances_pts.append(v["pts"])
            # instances_colors.append(v["colors"])
            point_ids.append(
                torch.full((self.smpl_points_num, 1), id_in_model, dtype=torch.long)
            )

        smpl_qauts = torch.cat(smpl_qauts, dim=1).to(
            self.device
        )  # (num_frame, num_instances, 23, 4)
        instances_quats = torch.cat(instances_quats, dim=1).to(
            self.device
        )  # (num_frame, num_instances, 4)
        instances_trans = torch.cat(instances_trans, dim=1).to(
            self.device
        )  # (num_frame, num_instances, 3)
        instances_fv = torch.cat(instances_fv, dim=1).to(
            self.device
        )  # (num_frame, num_instances)
        smpl_betas = torch.cat(smpl_betas, dim=0).to(self.device)  # (num_instances, 10)
        instances_size = torch.stack(instances_size).to(
            self.device
        )  # (num_instances, 3)
        point_ids = torch.cat(point_ids, dim=0).to(
            self.device
        )  # (self.smpl_points_num*num_instances, 1)
        self.instances_fv = instances_fv  # (num_frame, num_instances)

        self.template = SMPLTemplate(
            smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl",
            num_human=smpl_betas.shape[0],
            init_beta=smpl_betas,
            cano_pose_type="da_pose",
            use_voxel_deformer=self.use_voxel_deformer,
        )
        if self.use_voxel_deformer:
            self.template.voxel_deformer.enable_voxel_correction()

        opacity_init_value = torch.tensor(self.ctrl_cfg.opacity_init_value)
        x, q, s, o = get_on_mesh_init_geo_values(
            self.template,
            opacity_init_logit=torch.logit(opacity_init_value),
        )
        if self.ball_gaussians:
            s = s.mean(-1, keepdim=True)
        x = x.to(dtype=torch.float32, device=self.device)
        s = s.to(dtype=torch.float32, device=self.device)
        q = q.to(dtype=torch.float32, device=self.device)
        o = o.to(dtype=torch.float32, device=self.device)

        # knn init
        self.update_knn(x)

        if self.ctrl_cfg.constrain_xyz_offset:
            self.on_mesh_x = x.clone()

        # NOTE: In the future, we will also use colors of lidars to get the initialization of colors
        self.template = self.template.to(self.device)
        for fi in range(self.num_frames):
            instance_mask = instances_fv[fi]
            if instance_mask.sum() == 0:
                continue

            theta = torch.cat((instances_quats[fi].unsqueeze(1), smpl_qauts[fi]), dim=1)
            masked_theta = theta[instance_mask]
            masked_theta = masked_theta / masked_theta.norm(dim=-1, keepdim=True)
            W, A = self.template(
                masked_theta=masked_theta, instances_mask=instance_mask
            )
            T = torch.einsum("bnj, bjrc -> bnrc", W, A)
            R = T[:, :, :3, :3]  # [N, 3, 3]
            t = T[:, :, :3, 3]  # [N, 3]

            reshaped_means = x.reshape(self.num_instances, self.smpl_points_num, 3)
            deformed_means = (
                torch.einsum("bnij,bnj->bni", R, reshaped_means[instance_mask]) + t
            )  # [N, , 3]
            bbox_min = deformed_means.min(dim=1)[0]
            bbox_max = deformed_means.max(dim=1)[0]
            local_shift = (bbox_min + bbox_max) / 2
            instances_trans[fi, instance_mask] = (
                instances_trans[fi, instance_mask] - local_shift
            )

        self._means = Parameter(x, requires_grad=not self.ctrl_cfg.freeze_x)
        self._scales = Parameter(s, requires_grad=not self.ctrl_cfg.freeze_s)
        self._quats = Parameter(q, requires_grad=not self.ctrl_cfg.freeze_q)
        self._opacities = Parameter(o, requires_grad=not self.ctrl_cfg.freeze_o)

        self.instances_quats = Parameter(
            instances_quats.unsqueeze(2)
        )  # (num_frame, num_instances, 1, 4)
        self.instances_trans = Parameter(
            instances_trans
        )  # (num_frame, num_instances, 3)
        self.smpl_qauts = Parameter(smpl_qauts)  # (num_frame, num_instances, 23, 4)
        self.instances_size = instances_size  # (num_instances, 3)
        self.point_ids = point_ids  # (self.smpl_points_num*num_instances, 1)

        dim_sh = num_sh_bases(self.sh_degree)
        # NOTE: init_colors actually is for visualization, we use random color here
        # init_colors = RGB_tuples[self.point_ids.squeeze().cpu()].to(self.device)
        init_colors = torch.rand((self.num_points, 3), device=self.device)
        fused_color = RGB2SH(init_colors)  # float range [0, 1]
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(
            shs[:, 0, :], requires_grad=not self.ctrl_cfg.freeze_shs_dc
        )
        self._features_rest = Parameter(
            shs[:, 1:, :], requires_grad=not self.ctrl_cfg.freeze_shs_rest
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix + "ins_rotation"] = [self.instances_quats]
        param_groups[self.class_prefix + "ins_translation"] = [self.instances_trans]
        param_groups[self.class_prefix + "smpl_rotation"] = [self.smpl_qauts]
        if self.use_voxel_deformer:
            param_groups[self.class_prefix + "w_dc_vox"] = [
                self.template.voxel_deformer.voxel_w_correction
            ]
        #     param_groups[self.class_prefix+"w_rest_vox"] = [self.template.voxel_deformer.additional_correction]
        return param_groups

    @property
    def num_points(self):
        return self._means.shape[0]

    def update_knn(self, x: torch.Tensor) -> None:
        reshaped_x = x.reshape(self.num_instances, self.smpl_points_num, 3)
        _, nn_ind, _ = knn_points(
            reshaped_x, reshaped_x, K=self.ctrl_cfg.knn_neighbors, return_nn=False
        )
        self.nn_ind = nn_ind

    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        self.radii = radii
        self.xys_grad = xys_grad
        knn_update_interval = self.ctrl_cfg.get("knn_update_interval", 1000000)
        if self.step % knn_update_interval == 0:
            self.update_knn(self._means)

    def transform_means(self, means: torch.Tensor) -> torch.Tensor:
        """
        transform the means of instances to world space
        according to the pose at the current frame
        """
        assert (
            means.shape[0] == self.point_ids.shape[0]
        ), "its a bug here, we need to pass the mask for points_ids"
        instance_mask = self.instances_fv[self.cur_frame]
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_masked_theta = torch.cat(
                (
                    self.instances_quats[self.cur_frame - 1],
                    self.smpl_qauts[self.cur_frame - 1],
                ),
                dim=1,
            )[instance_mask]
            _next_masked_theta = torch.cat(
                (
                    self.instances_quats[self.cur_frame + 1],
                    self.smpl_qauts[self.cur_frame + 1],
                ),
                dim=1,
            )[instance_mask]
            _cur_masked_theta = torch.cat(
                (self.instances_quats[self.cur_frame], self.smpl_qauts[self.cur_frame]),
                dim=1,
            )[instance_mask]
            interpolated_theta = interpolate_quats(
                _prev_masked_theta, _next_masked_theta
            )

            inter_valid_mask = (
                self.instances_fv[self.cur_frame - 1, instance_mask]
                & self.instances_fv[self.cur_frame + 1, instance_mask]
            )
            masked_theta = torch.where(
                inter_valid_mask[:, None, None], interpolated_theta, _cur_masked_theta
            )
        else:
            theta = torch.cat(
                (self.instances_quats[self.cur_frame], self.smpl_qauts[self.cur_frame]),
                dim=1,
            )
            masked_theta = theta[instance_mask]
        masked_theta = self.quat_act(masked_theta)
        W, A = self.template(
            masked_theta=masked_theta,
            instances_mask=instance_mask,
            xyz_canonical=(
                means.reshape(self.num_instances, self.smpl_points_num, 3)
                if self.use_voxel_deformer
                else None
            ),
        )
        T = torch.einsum("bnj, bjrc -> bnrc", W, A)
        R = T[:, :, :3, :3]  # [N, 3, 3]
        t = T[:, :, :3, 3]  # [N, 3]

        reshaped_means = means.reshape(self.num_instances, self.smpl_points_num, 3)
        deformed_means = (
            torch.einsum("bnij,bnj->bni", R, reshaped_means[instance_mask]) + t
        )  # [N, 6890, 3]

        means_container = torch.zeros_like(reshaped_means)
        means_container.index_add_(0, instance_mask.nonzero().squeeze(), deformed_means)
        means_container = means_container.reshape(-1, 3)

        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_ins_trans = self.instances_trans[self.cur_frame - 1]
            _next_ins_trans = self.instances_trans[self.cur_frame + 1]
            _cur_ins_trans = self.instances_trans[self.cur_frame]
            interpolated_trans = (_prev_ins_trans + _next_ins_trans) * 0.5

            inter_valid_mask = (
                self.instances_fv[self.cur_frame - 1]
                & self.instances_fv[self.cur_frame + 1]
            )
            trans_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_trans, _cur_ins_trans
            )
        else:
            trans_cur_frame = self.instances_trans[self.cur_frame]  # (num_instances, 3)
        trans_per_pts = trans_cur_frame[self.point_ids[..., 0]]

        # transform the means to world space
        means_container += trans_per_pts
        return means_container

    def transform_means_and_quats(
        self, means: torch.Tensor, quats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        transform the means and quats of gaussians to world space
        according to the pose at the current frame
        """
        assert (
            means.shape[0] == self.point_ids.shape[0]
        ), "its a bug here, we need to pass the mask for points_ids"
        instance_mask = self.instances_fv[self.cur_frame]
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_masked_theta = torch.cat(
                (
                    self.instances_quats[self.cur_frame - 1],
                    self.smpl_qauts[self.cur_frame - 1],
                ),
                dim=1,
            )[instance_mask]
            _next_masked_theta = torch.cat(
                (
                    self.instances_quats[self.cur_frame + 1],
                    self.smpl_qauts[self.cur_frame + 1],
                ),
                dim=1,
            )[instance_mask]
            _cur_masked_theta = torch.cat(
                (self.instances_quats[self.cur_frame], self.smpl_qauts[self.cur_frame]),
                dim=1,
            )[instance_mask]
            interpolated_theta = interpolate_quats(
                _prev_masked_theta, _next_masked_theta
            )

            inter_valid_mask = (
                self.instances_fv[self.cur_frame - 1, instance_mask]
                & self.instances_fv[self.cur_frame + 1, instance_mask]
            )
            masked_theta = torch.where(
                inter_valid_mask[:, None, None], interpolated_theta, _cur_masked_theta
            )
        else:
            theta = torch.cat(
                (self.instances_quats[self.cur_frame], self.smpl_qauts[self.cur_frame]),
                dim=1,
            )
            masked_theta = theta[instance_mask]
        masked_theta = self.quat_act(masked_theta)
        W, A = self.template(
            masked_theta=masked_theta,
            instances_mask=instance_mask,
            xyz_canonical=(
                means.reshape(self.num_instances, self.smpl_points_num, 3)
                if self.use_voxel_deformer
                else None
            ),
        )
        T = torch.einsum("bnj, bjrc -> bnrc", W, A)
        R = T[:, :, :3, :3]  # [N, 3, 3]
        t = T[:, :, :3, 3]  # [N, 3]

        reshaped_means = means.reshape(self.num_instances, self.smpl_points_num, 3)
        deformed_means = (
            torch.einsum("bnij,bnj->bni", R, reshaped_means[instance_mask]) + t
        )  # [N, 6890, 3]

        means_container = torch.zeros_like(reshaped_means)
        means_container.index_add_(0, instance_mask.nonzero().squeeze(), deformed_means)
        means_container = means_container.reshape(-1, 3)

        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_ins_trans = self.instances_trans[self.cur_frame - 1]
            _next_ins_trans = self.instances_trans[self.cur_frame + 1]
            _cur_ins_trans = self.instances_trans[self.cur_frame]
            interpolated_trans = (_prev_ins_trans + _next_ins_trans) * 0.5

            inter_valid_mask = (
                self.instances_fv[self.cur_frame - 1]
                & self.instances_fv[self.cur_frame + 1]
            )
            trans_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_trans, _cur_ins_trans
            )
        else:
            trans_cur_frame = self.instances_trans[self.cur_frame]  # (num_instances, 3)
        trans_per_pts = trans_cur_frame[self.point_ids[..., 0]]

        # transform the means to world space
        means_container += trans_per_pts

        reshaped_quats = quats.reshape(self.num_instances, self.smpl_points_num, 4)
        R_quats = matrix_to_quaternion(R)
        deformed_quats = quat_mult(
            self.quat_act(R_quats), self.quat_act(reshaped_quats[instance_mask])
        )
        quats_container = torch.zeros_like(reshaped_quats)
        quats_container.index_add_(0, instance_mask.nonzero().squeeze(), deformed_quats)
        # fill other with [1, 0, 0, 0]
        quats_container.index_add_(
            0,
            (~instance_mask).nonzero().squeeze(),
            torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=self.device).repeat(
                (~instance_mask).sum(), 6890, 1
            ),
        )
        quats_container = quats_container.reshape(-1, 4)
        return means_container, quats_container

    def get_gaussians(self, cam: dataclass_camera) -> Dict[str, torch.Tensor]:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask

        instance_mask = self.instances_fv[self.cur_frame]
        if instance_mask.sum() == 0:
            return None

        if self.ball_gaussians:
            world_means = self.transform_means(self._means)
            world_quats = self._quats
        else:
            world_means, world_quats = self.transform_means_and_quats(
                self._means, self._quats
            )

        # 检查world_means是否有NaN
        if torch.isnan(world_means).any():
            logger.warning("NaN detected in world_means")
            world_means = torch.nan_to_num(world_means, 0.0)

        # 获取颜色
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)

        # 添加安全检查
        if self.sh_degree > 0:
            try:
                viewdirs = (
                    world_means.detach() - cam.camtoworlds.data[..., :3, 3]
                )  # (N, 3)
                # 添加eps避免除零
                viewdirs = viewdirs / (viewdirs.norm(dim=-1, keepdim=True) + 1e-10)
                n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
                rgbs = spherical_harmonics(n, viewdirs, colors)
                # 使用clamp确保值在有效范围内
                rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
            except Exception as e:
                logger.error(f"Error in computing spherical harmonics: {e}")
                # 如果计算出错，使用默认颜色
                rgbs = torch.ones_like(colors[:, 0, :]) * 0.5
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        # 检查rgbs是否有NaN
        if torch.isnan(rgbs).any():
            logger.warning("NaN detected in rgbs, replacing with default values")
            rgbs = torch.nan_to_num(rgbs, 0.5)

        valid_mask = self.get_pts_valid_mask()

        activated_opacities = self.get_opacity * valid_mask.float().unsqueeze(-1)
        if self.ball_gaussians:
            activated_scales = torch.exp(self._scales.repeat(1, 3))
        else:
            activated_scales = torch.exp(self._scales)
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs

        # 收集高斯体信息
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )

        # 检查所有值
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                logger.error(f"NaN detected in gaussian {k} at step {self.step}")
                # 替换NaN值而不是抛出错误
                if k == "_rgbs":
                    gs_dict[k] = torch.nan_to_num(v, 0.5)  # 使用灰色
                else:
                    gs_dict[k] = torch.nan_to_num(v, 0.0)

        self._gs_cache = {
            "_scales": activated_scales[filter_mask],
        }
        return gs_dict

    def compute_reg_loss(self):
        loss_dict = super().compute_reg_loss()

        instance_mask = self.instances_fv[self.cur_frame]
        if instance_mask.sum() == 0:
            return loss_dict

        # temporal smooth regularization
        temporal_smooth_reg = self.reg_cfg.get("temporal_smooth_reg", None)
        if temporal_smooth_reg is not None:
            joint_smooth_reg = temporal_smooth_reg.get("joint_smooth", None)
            if joint_smooth_reg is not None:
                if self.cur_frame >= 1 and self.cur_frame < self.num_frames - 1:
                    valid_mask = (
                        self.instances_fv[self.cur_frame - 1]
                        & self.instances_fv[self.cur_frame + 1]
                        & self.instances_fv[self.cur_frame]
                    )
                    cur_theta = torch.cat(
                        (
                            self.instances_quats[self.cur_frame],
                            self.smpl_qauts[self.cur_frame],
                        ),
                        dim=1,
                    )[valid_mask]
                    next_theta = torch.cat(
                        (
                            self.instances_quats[self.cur_frame + 1],
                            self.smpl_qauts[self.cur_frame + 1],
                        ),
                        dim=1,
                    )[valid_mask]
                    prev_theta = torch.cat(
                        (
                            self.instances_quats[self.cur_frame - 1],
                            self.smpl_qauts[self.cur_frame - 1],
                        ),
                        dim=1,
                    )[valid_mask]
                    thetas = torch.vstack([prev_theta, cur_theta, next_theta])
                    thetas = self.quat_act(thetas)
                    J_transformed, _ = batch_rigid_transform(
                        quaternion_to_matrix(thetas),
                        self.template.J_canonical[valid_mask].repeat(3, 1, 1),
                        self.template._template_layer.parents,
                    )

                    cur_trans = self.instances_trans[self.cur_frame, valid_mask]
                    next_trans = self.instances_trans[self.cur_frame + 1, valid_mask]
                    prev_trans = self.instances_trans[self.cur_frame - 1, valid_mask]
                    trans = torch.vstack([prev_trans, cur_trans, next_trans])
                    J_transformed += trans.unsqueeze(-2)
                    J_transformed = J_transformed.reshape(3, -1, 24, 3)

                    velocity_prev = J_transformed[1] - J_transformed[0]
                    velocity_next = J_transformed[2] - J_transformed[1]
                    # l2 loss
                    loss_dict["smpl_temporal_smooth"] = (
                        velocity_next - velocity_prev
                    ).abs().mean() * joint_smooth_reg.w

        # voxel deformer regularization
        voxel_deformer_reg = self.reg_cfg.get("voxel_deformer_reg", None)
        if voxel_deformer_reg is not None and self.use_voxel_deformer:
            w_std = self.template.voxel_deformer.get_tv("dc")
            w_rest_std = self.template.voxel_deformer.get_tv("rest")
            w_norm = self.template.voxel_deformer.get_mag("dc")
            w_rest_norm = self.template.voxel_deformer.get_mag("rest")

            loss_dict["voxel_deformer_reg"] = (
                voxel_deformer_reg.lambda_std_w * w_std
                + voxel_deformer_reg.lambda_std_w_rest * w_rest_std
                + voxel_deformer_reg.lambda_w_norm * w_norm
                + voxel_deformer_reg.lambda_w_rest_norm * w_rest_norm
            )

        # knn regularization
        knn_reg = self.reg_cfg.get("knn_reg", None)
        if knn_reg is not None:
            K = self.ctrl_cfg.knn_neighbors
            instances_mask = self.instances_fv[self.cur_frame]
            nn_ind = self.nn_ind[
                instances_mask
            ]  # (num_instances, smpl_points_num, knn_neighbors)

            if not self.ctrl_cfg.freeze_shs_dc:
                valid_shs_dc = self._features_dc.reshape(
                    self.num_instances, self.smpl_points_num, 3
                )[
                    instances_mask
                ]  # (num_instances, smpl_points_num, 3)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, 3)
                knn_shs_dc = torch.gather(
                    valid_shs_dc.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded
                )  # (num_instances, smpl_points_num, knn_neighbors, 3)
                shs_dc_std = knn_shs_dc.std(dim=2).mean()
                loss_dict["knn_reg_dc"] = shs_dc_std * knn_reg.lambda_std_shs_dc

            if not self.ctrl_cfg.freeze_shs_rest and self.sh_degree > 0:
                dim_sh = num_sh_bases(self.sh_degree)
                valid_shs_rest = self._features_rest.reshape(
                    self.num_instances, self.smpl_points_num, -1
                )[
                    instances_mask
                ]  # (num_instances, smpl_points_num, (dim_sh-1)*3)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(
                    -1, -1, -1, (dim_sh - 1) * 3
                )
                knn_shs_rest = torch.gather(
                    valid_shs_rest.unsqueeze(2).expand(-1, -1, K, -1),
                    1,
                    nn_ind_expanded,
                )  # (num_instances, smpl_points_num, knn_neighbors, (dim_sh-1)*3)
                shs_rest_std = knn_shs_rest.std(dim=2).mean()
                loss_dict["knn_reg_rest"] = shs_rest_std * knn_reg.lambda_std_shs_rest

            if not self.ctrl_cfg.freeze_o:
                valid_o = self.get_opacity.reshape(
                    self.num_instances, self.smpl_points_num, 1
                )[
                    instances_mask
                ]  # (num_instances, smpl_points_num, 1)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, 1)
                knn_o = torch.gather(
                    valid_o.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded
                )
                o_std = knn_o.std(dim=2).mean()
                loss_dict["knn_reg_o"] = o_std * knn_reg.lambda_std_o

            if not self.ctrl_cfg.freeze_s:
                scale_dim = 1 if self.ball_gaussians else 3
                valid_s = self.get_scaling.reshape(
                    self.num_instances, self.smpl_points_num, scale_dim
                )[
                    instances_mask
                ]  # (num_instances, smpl_points_num, 1)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, scale_dim)
                knn_s = torch.gather(
                    valid_s.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded
                )
                s_std = knn_s.std(dim=2).mean()
                loss_dict["knn_reg_s"] = s_std * knn_reg.lambda_std_s

            if not self.ctrl_cfg.freeze_q:
                valid_q = self._quats.reshape(
                    self.num_instances, self.smpl_points_num, 4
                )[
                    instances_mask
                ]  # (num_instances, smpl_points_num, 4)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, 4)
                knn_q = torch.gather(
                    valid_q.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded
                )
                q_std = knn_q.std(dim=2).mean()
                loss_dict["knn_reg_q"] = q_std * knn_reg.lambda_std_q

            # valid_x = self._means.reshape(self.num_instances, self.smpl_points_num, 3)[instances_mask] # (num_instances, smpl_points_num, 3)
            # nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, 3)
            # knn_x = torch.gather(valid_x.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded)
            # x_std = knn_x.std(dim=2).mean()
            # loss_dict["knn_reg_x"] = x_std * knn_reg.lambda_std_x

        x_offset_reg = self.reg_cfg.get("x_offset", None)
        if (
            x_offset_reg is not None
            and self.ctrl_cfg.constrain_xyz_offset
            and not self.ctrl_cfg.freeze_x
        ):
            instances_mask = self.instances_fv[self.cur_frame]
            valid_x = self._means.reshape(self.num_instances, self.smpl_points_num, 3)[
                instances_mask
            ]  # (num_instances, smpl_points_num, 3)
            valid_x_on_mesh = self.on_mesh_x.reshape(
                self.num_instances, self.smpl_points_num, 3
            )[instances_mask]
            x_offset = (valid_x - valid_x_on_mesh).norm(dim=-1).mean()

            loss_dict["x_offset"] = x_offset * x_offset_reg.w

        return loss_dict

    def state_dict(self) -> Dict:
        state_dict = VanillaGaussians.state_dict(self)
        state_dict.update(
            {
                "points_ids": self.point_ids,
                "instances_size": self.instances_size,
                "instances_fv": self.instances_fv,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        self.point_ids = state_dict.pop("points_ids")
        self.instances_size = state_dict.pop("instances_size")
        self.instances_fv = state_dict.pop("instances_fv")
        self.instances_trans = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 3, device=self.device)
        )
        self.instances_quats = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 1, 4, device=self.device)
        )
        self.smpl_qauts = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 23, 4, device=self.device)
        )
        self.template = SMPLTemplate(
            smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl",
            num_human=self.num_instances,
            init_beta=torch.zeros(self.num_instances, 10, device=self.device),
            cano_pose_type="da_pose",
            use_voxel_deformer=self.use_voxel_deformer,
            is_resume=True,
        ).to(self.device)
        if self.use_voxel_deformer:
            self.template.voxel_deformer.enable_voxel_correction()
        msg = VanillaGaussians.load_state_dict(self, state_dict, **kwargs)
        return msg

    def get_instance_activated_gs_dict(self, ins_id: int) -> Dict[str, torch.Tensor]:
        pts_mask = self.point_ids[..., 0] == ins_id
        if pts_mask.sum() < 100:
            return None
        local_means = self._means[pts_mask]
        activated_opacities = torch.sigmoid(self._opacities[pts_mask])
        activated_scales = torch.exp(
            self._scales[pts_mask].repeat(1, 3)
            if self.ball_gaussians
            else self._scales[pts_mask]
        )
        activated_local_rotations = self.quat_act(self._quats[pts_mask])
        gaussian_dict = {
            "means": local_means,
            "opacities": activated_opacities,
            "scales": activated_scales,
            "quats": activated_local_rotations,
            "sh_dcs": self._features_dc[pts_mask],
            "sh_rests": self._features_rest[pts_mask],
            "ids": self.point_ids[pts_mask],
        }
        return gaussian_dict

    def deform_gaussian_points(
        self,
        gaussian_dict: Dict[str, torch.Tensor],
        cur_normalized_time: float,
    ) -> Dict[str, torch.Tensor]:
        """
        deform the points
        """
        means = gaussian_dict["means"]
        cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - cur_normalized_time)
        )
        ins_id = gaussian_dict["ids"].flatten()[0]
        if not self.instances_fv[cur_frame, ins_id]:
            # find the nearest frame that has the instance
            for i in range(1, self.num_frames):
                if cur_frame - i >= 0:
                    if self.instances_fv[cur_frame - i, ins_id]:
                        cur_frame = cur_frame - i
                        break
                if cur_frame + i < self.num_frames:
                    if self.instances_fv[cur_frame + i, ins_id]:
                        cur_frame = cur_frame + i
                        break
        instance_mask = torch.zeros(self.num_instances, device=self.device)
        instance_mask[ins_id] = 1
        instance_mask = instance_mask.bool()
        masked_theta = torch.cat(
            [
                torch.tensor([1.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0),
                self.smpl_qauts[cur_frame, ins_id],
            ],
            dim=0,
        ).unsqueeze(0)

        W, A = self.template(
            masked_theta=self.quat_act(masked_theta),
            instances_mask=instance_mask,
            xyz_canonical=(
                means.reshape(1, self.smpl_points_num, 3).repeat(
                    self.num_instances, 1, 1
                )
                if self.use_voxel_deformer
                else None
            ),
        )
        T = torch.einsum("bnj, bjrc -> bnrc", W, A)
        R = T[..., :3, :3].squeeze()  # [N, 3, 3]
        t = T[..., :3, 3].squeeze()  # [N, 3]

        deformed_means = torch.einsum("nij,nj->ni", R, means) + t  # [6890, 3]
        gaussian_dict["means"] = deformed_means

        # placeholder for quats rotating: TODO
        gaussian_dict["quats"] = gaussian_dict["quats"]
        return gaussian_dict

    def collect_gaussians_from_ids(self, ids: List[int]) -> Dict:
        gaussian_dict = {}
        for id in ids:
            if id not in gaussian_dict:
                instance_raw_dict = {
                    "_means": self._means[self.point_ids[..., 0] == id],
                    "_scales": self._scales[self.point_ids[..., 0] == id],
                    "_quats": self._quats[self.point_ids[..., 0] == id],
                    "_features_dc": self._features_dc[self.point_ids[..., 0] == id],
                    "_features_rest": self._features_rest[self.point_ids[..., 0] == id],
                    "_opacities": self._opacities[self.point_ids[..., 0] == id],
                    "point_ids": self.point_ids[self.point_ids[..., 0] == id],
                    "instances_fv": self.instances_fv[:, id],
                    "instances_quats": self.instances_quats.data[:, id],
                    "smpl_qauts": self.smpl_qauts.data[:, id],
                    "smpl_template": {
                        "J_canonical": self.template.J_canonical[id],
                        "W": self.template.W[id],
                        "voxel_deformer": {
                            "lbs_voxel_base": self.template.voxel_deformer.lbs_voxel_base[
                                id
                            ],
                            "voxel_w_correction": self.template.voxel_deformer.voxel_w_correction[
                                id
                            ],
                            "offset": self.template.voxel_deformer.offset[id],
                            "scale": self.template.voxel_deformer.scale[id],
                        },
                    },
                }
                gaussian_dict[id] = instance_raw_dict
        return gaussian_dict

    def remove_instances(self, remove_id_list: List[int]) -> None:
        """
        一次性移除多个实例，确保所有相关参数都被正确更新

        Args:
            remove_id_list: 要移除的实例ID列表
        """
        # 创建保留实例的掩码
        keep_mask = torch.ones(self.num_instances, dtype=torch.bool, device=self.device)
        keep_mask[remove_id_list] = False

        # 创建点的掩码
        points_keep_mask = ~torch.isin(
            self.point_ids[..., 0], torch.tensor(remove_id_list, device=self.device)
        )

        # 更新高斯体参数
        self._means = Parameter(self._means[points_keep_mask])
        self._scales = Parameter(self._scales[points_keep_mask])
        self._quats = Parameter(self._quats[points_keep_mask])
        self._features_dc = Parameter(self._features_dc[points_keep_mask])
        self._features_rest = Parameter(self._features_rest[points_keep_mask])
        self._opacities = Parameter(self._opacities[points_keep_mask])
        self.point_ids = self.point_ids[points_keep_mask]

        # 更新实例相关参数
        self.instances_fv = self.instances_fv[:, keep_mask]
        self.instances_trans = Parameter(self.instances_trans[:, keep_mask])
        self.instances_quats = Parameter(self.instances_quats[:, keep_mask])
        self.smpl_qauts = Parameter(self.smpl_qauts[:, keep_mask])

        # 更新SMPL模板
        self.template = SMPLTemplate(
            smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl",
            num_human=keep_mask.sum().item(),  # 更新实例数量
            init_beta=torch.zeros(keep_mask.sum().item(), 10, device=self.device),
            cano_pose_type="da_pose",
            use_voxel_deformer=self.use_voxel_deformer,
            is_resume=True,
        ).to(self.device)

        # 如果使用体素变形器，启用它
        if self.use_voxel_deformer:
            self.template.voxel_deformer.enable_voxel_correction()

        # 确保所有尺寸匹配
        assert self.instances_fv.shape[1] == keep_mask.sum().item(), "实例数量不匹配"
        assert (
            self.instances_trans.shape[1] == keep_mask.sum().item()
        ), "变换参数数量不匹配"
        assert self.smpl_qauts.shape[1] == keep_mask.sum().item(), "SMPL参数数量不匹配"

    def set_smpl_params(
        self,
        instance_id: int,
        frame_idx: int = None,
        theta: torch.Tensor = None,  # [24, 4] 四元数形式的姿态参数，第 0 行为全局、后 23 行为关节旋转
        beta: torch.Tensor = None,  # [10] 形状参数
    ) -> None:
        """
        设置指定实例的 SMPL 参数（姿态和形状）。
        当 theta 或 beta 为 None 时，保持原有参数不变。
        更新前后会打印原本的参数和修改后的参数。

        Args:
            instance_id: 实例ID
            frame_idx: 帧索引，若为 None 则使用当前帧
            theta: 姿态参数，形状为 [24, 4]
            beta: 形状参数，形状为 [10]
        """
        if frame_idx is None:
            frame_idx = self.cur_frame

        with torch.no_grad():
            # 如果传入 theta 非空，则更新姿态参数
            if theta is not None:
                assert theta.shape == (
                    24,
                    4,
                ), f"theta shape should be (24, 4), got {theta.shape}"
                # 获取原来的参数
                orig_global = (
                    self.instances_quats[frame_idx, instance_id].detach().clone()
                )
                orig_joints = self.smpl_qauts[frame_idx, instance_id].detach().clone()
                print(
                    f"[set_smpl_params] Instance {instance_id} (frame {frame_idx}) original theta: global {orig_global}, joints {orig_joints}"
                )

                # 分离全局旋转和关节旋转参数
                global_rot = theta[:1]  # [1, 4]
                joint_rots = theta[1:]  # [23, 4]
                self.instances_quats[frame_idx, instance_id] = global_rot
                self.smpl_qauts[frame_idx, instance_id] = joint_rots

                print(
                    f"[set_smpl_params] Instance {instance_id} (frame {frame_idx}) updated theta: global {global_rot}, joints {joint_rots}"
                )

            # 如果传入 beta 非空，则更新形状参数
            if beta is not None:
                assert beta.shape == (
                    10,
                ), f"beta shape should be (10,), got {beta.shape}"
                orig_beta = self.template.init_beta[instance_id].detach().clone()
                print(
                    f"[set_smpl_params] Instance {instance_id} original beta: {orig_beta}"
                )

                # 更新模板中的形状参数（采用 in-place 更新）
                self.template.init_beta[instance_id] = beta

                # 手动更新该实例在模板中的输出参数
                # 使用模板中定义的 canonical_pose 作为标准零姿态输入
                canonical_pose = self.template.canonical_pose  # [24, 3, 3]
                body_pose = canonical_pose[None, 1:].repeat(1, 1, 1, 1)  # [1,23,3,3]
                global_orient = canonical_pose[None, 0].repeat(1, 1, 1)  # [1,3,3]

                # 获取该实例的 beta 参数，形状 [1, 10]
                instance_beta = self.template.init_beta[instance_id : instance_id + 1]
                smpl_output = self.template._template_layer(
                    betas=instance_beta,
                    body_pose=body_pose,
                    global_orient=global_orient,
                    return_full_pose=True,
                )
                # 更新该实例对应的 J_canonical 和 A0_inv
                self.template.J_canonical[instance_id] = smpl_output.J[0]
                A0 = smpl_output.A
                A0_inv = torch.inverse(A0)
                self.template.A0_inv[instance_id] = A0_inv

                print(
                    f"[set_smpl_params] Instance {instance_id} updated beta: {self.template.init_beta[instance_id]}"
                )

    def replace_instances(self, replace_dict: Dict[int, int]) -> None:
        """
        替换实例，包括几何形状和运动参数

        Args:
            replace_dict: {
                target_id: source_id,  # 用source_id的实例替换target_id的实例
                ...
            }
        """
        new_gaussians_dict = self.collect_gaussians_from_ids(
            list(replace_dict.values())
        )

        for target_id, source_id in replace_dict.items():
            # 1. 移除目标实例的几何体
            self.remove_instances([target_id])

            # 2. 复制源实例的几何体
            new_gaussian = new_gaussians_dict[source_id]
            self._means = Parameter(
                torch.cat([self._means, new_gaussian["_means"]], dim=0)
            )
            self._scales = Parameter(
                torch.cat([self._scales, new_gaussian["_scales"]], dim=0)
            )
            self._quats = Parameter(
                torch.cat([self._quats, new_gaussian["_quats"]], dim=0)
            )
            self._features_dc = Parameter(
                torch.cat([self._features_dc, new_gaussian["_features_dc"]], dim=0)
            )
            self._features_rest = Parameter(
                torch.cat([self._features_rest, new_gaussian["_features_rest"]], dim=0)
            )
            self._opacities = Parameter(
                torch.cat([self._opacities, new_gaussian["_opacities"]], dim=0)
            )

            # 3. 复制运动参数
            self.instances_quats[:, target_id] = self.instances_quats[
                :, source_id
            ].clone()
            self.smpl_qauts[:, target_id] = self.smpl_qauts[:, source_id].clone()
            self.instances_trans[:, target_id] = self.instances_trans[
                :, source_id
            ].clone()

            # 4. 复制形状参数
            if hasattr(self.template, "init_beta"):
                self.template.init_beta[target_id] = self.template.init_beta[
                    source_id
                ].clone()

            # 5. 更新point_ids
            self.point_ids = torch.cat(
                [self.point_ids, torch.full_like(new_gaussian["point_ids"], target_id)],
                dim=0,
            )

            # 6. 更新实例可见性
            self.instances_fv[:, target_id] = self.instances_fv[:, source_id].clone()

    def update_template(self):
        """
        更新SMPL模板
        在修改beta参数后调用此函数以更新模板
        """
        # 使用当前的beta参数重新计算模板
        with torch.no_grad():
            smpl_output = self.template._template_layer(
                betas=self.init_beta,
                body_pose=self.canonical_pose[None, 1:],
                global_orient=self.canonical_pose[None, 0],
                return_full_pose=True,
            )
            self.J_canonical = smpl_output.J
            self.A0_inv = torch.inverse(smpl_output.A[0])

    def export_gaussians_to_ply(
        self,
        alpha_thresh: float,
        instance_id: List[int] = None,
        specific_frame: int = 0,
    ) -> Dict[str, torch.Tensor]:
        self.cur_frame = specific_frame
        cur_frame = 10
        ins_id = instance_id

        means = self._means[self.point_ids[..., 0] == ins_id]

        if not self.instances_fv[cur_frame, ins_id]:
            # find the nearest frame that has the instance
            for i in range(1, self.num_frames):
                if cur_frame - i >= 0:
                    if self.instances_fv[cur_frame - i, ins_id]:
                        cur_frame = cur_frame - i
                        break
                if cur_frame + i < self.num_frames:
                    if self.instances_fv[cur_frame + i, ins_id]:
                        cur_frame = cur_frame + i
                        break
        instance_mask = torch.zeros(self.num_instances, device=self.device)
        instance_mask[ins_id] = 1
        instance_mask = instance_mask.bool()
        masked_theta = torch.cat(
            [
                torch.tensor([1.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0),
                self.smpl_qauts[cur_frame, ins_id],
            ],
            dim=0,
        ).unsqueeze(0)

        W, A = self.template(
            masked_theta=self.quat_act(masked_theta),
            instances_mask=instance_mask,
            xyz_canonical=(
                means.reshape(1, self.smpl_points_num, 3).repeat(
                    self.num_instances, 1, 1
                )
                if self.use_voxel_deformer
                else None
            ),
        )
        T = torch.einsum("bnj, bjrc -> bnrc", W, A)
        R = T[..., :3, :3].squeeze()  # [N, 3, 3]
        t = T[..., :3, 3].squeeze()  # [N, 3]

        deformed_means = torch.einsum("nij,nj->ni", R, means) + t  # [6890, 3]

        color = self.colors[self.point_ids[..., 0] == ins_id]
        return {
            "positions": deformed_means,
            "colors": color,
        }

    def add_transform_offset(
        self,
        instance_id: int,
        frame_idx: int,
        rotation_offset: torch.Tensor = None,
        translation_offset: torch.Tensor = None,
        smpl_rotation_offset: torch.Tensor = None,  # 新增SMPL姿态偏移
    ) -> None:
        """添加变换偏移"""
        with torch.no_grad():
            if translation_offset is not None:
                self.instances_trans[frame_idx, instance_id] = (
                    self.instances_trans[frame_idx, instance_id] + translation_offset
                )

            if rotation_offset is not None:
                current_rotation = self.instances_quats[frame_idx, instance_id]
                new_rotation = quaternion_multiply(rotation_offset, current_rotation)
                self.instances_quats[frame_idx, instance_id] = new_rotation

            if smpl_rotation_offset is not None:
                # 对SMPL姿态进行编辑
                current_smpl_rotation = self.smpl_qauts[frame_idx, instance_id]
                new_smpl_rotation = quaternion_multiply(
                    smpl_rotation_offset.unsqueeze(0).repeat(23, 1),
                    current_smpl_rotation,
                )
                self.smpl_qauts[frame_idx, instance_id] = new_smpl_rotation
