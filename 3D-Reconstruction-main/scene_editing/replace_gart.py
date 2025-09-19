import os
import logging
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
from utils.misc import import_str
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import numpy as np
from torch.nn import functional as F

# 导入共享函数
from scene_editing import (
    replace_smpl_instance_improved,
    batch_render_with_eval
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_gart_to_smpl(solver, num_frames):
    """按照viz_human_all数据格式提取参数，并修正参数范围和形状匹配"""
    model = solver.load_saved_model(os.path.join(solver.log_dir, "model.pth"))
    model.eval()
    
    # 使用与viz_human_all类似的方式准备姿态数据
    tpose = torch.zeros((1, 24, 3), dtype=torch.float32, device=solver.device)
    if hasattr(solver, 'viz_base_R'):
        tpose[:, 0] = matrix_to_axis_angle(solver.viz_base_R[None])[0]
    
    # 设置默认变换
    trans = torch.zeros(num_frames, 3, device=solver.device)
    trans[:, 2] = 3.0  # 默认z偏移
    
    # 获取点云和其他属性 - 使用属性访问器获取正确激活的值
    pts = model.get_x.detach()
    
    # 1. 居中和缩放点云处理
    pts_center = pts.mean(dim=0)
    pts_centered = pts - pts_center
    pts_scale = pts_centered.abs().max()
    pts_normalized = pts_centered / pts_scale  # 缩放到更合理的范围
    
    # 2. 获取特征，保持原始值而不是激活后的值
    features_dc = model._features_dc.detach()
    features_rest = model._features_rest.detach()
    
    # 3. 修正特征形状问题
    # 如果特征休息形状是[6890, 9]，重塑为[6890, 3, 3]
    if features_rest.shape[1] == 9:
        features_rest = features_rest.reshape(6890, 3, 3)
    
    # 4. 获取缩放和不透明度的原始logit值，而不是激活后的值
    # 定义逆激活函数
    s_inv_act = lambda x: torch.log(x + 1e-8)  # 添加小值避免log(0)
    o_inv_act = lambda x: torch.logit(torch.clamp(x, 0.001, 0.999))  # 裁剪避免0和1
    
    # 获取激活后的值
    scaling_activated = model.get_s.detach()
    opacity_activated = model.get_o.detach()
    
    # 转换回logit/log空间，匹配SMPL的预期格式
    scaling = s_inv_act(scaling_activated)
    opacity = o_inv_act(opacity_activated)
    
    # 确保形状正确
    if scaling.shape[1] == 1 and model._scaling.shape[1] == 3:
        scaling = scaling.repeat(1, 3)
    
    # 保持旋转参数为四元数格式
    rotation = model._rotation.detach()
    
    # 5. 创建有效的映射关系
    mapping_dist = torch.ones(6890, 1, device=pts.device)
    mapping_face = torch.zeros(6890, 1, dtype=torch.int64, device=pts.device)
    mapping_uvw = torch.ones(6890, 3, device=pts.device) / 3.0
    
    # 6. 转换姿态为四元数形式，确保单位四元数
    from pytorch3d.transforms import axis_angle_to_quaternion
    global_orient = axis_angle_to_quaternion(tpose[:, 0:1]).repeat(num_frames, 1, 1)
    body_pose = axis_angle_to_quaternion(tpose[:, 1:]).repeat(num_frames, 1, 1)
    
    # 确保四元数已归一化
    global_orient = F.normalize(global_orient, dim=-1)
    body_pose = F.normalize(body_pose, dim=-1)
    
    # 创建实例数据
    instance_data = {
        "pts": pts_normalized,  # 使用居中和重新缩放的点云
        "colors": torch.sigmoid(features_dc),  # 颜色需要在sigmoid空间
        "scales": scaling,      # 在log空间，匹配SMPL期望
        "rotations": rotation,  # 四元数表示
        "opacities": opacity,   # 在logit空间，匹配SMPL期望
        "features_dc": features_dc,
        "features_rest": features_rest,
        "global_orient": global_orient,  # [num_frames, 1, 4]
        "body_pose": body_pose,         # [num_frames, 23, 4]
        "transl": trans,                # [num_frames, 3]
        "betas": torch.zeros(10, device=solver.device),  # 默认形状参数
        "mapping": {
            "dist": mapping_dist,
            "face": mapping_face,
            "uvw": mapping_uvw,
        },
        "size": torch.tensor([1.0, 1.0, 1.0], device=solver.device),
        "frame_info": torch.ones(num_frames, dtype=torch.bool, device=solver.device),
        "num_pts": 6890
    }
    
    # 添加详细的调试信息
    logger.info(f"GART点云统计: 原始={pts.shape[0]}, 范围={pts.min(dim=0).values} 到 {pts.max(dim=0).values}")
    logger.info(f"处理后点云: 范围={pts_normalized.min(dim=0).values} 到 {pts_normalized.max(dim=0).values}, 中心={pts_normalized.mean(dim=0)}")
    logger.info(f"缩放参数: 原范围={scaling_activated.min():.6f} 到 {scaling_activated.max():.6f}, 转换后={scaling.min():.6f} 到 {scaling.max():.6f}")
    logger.info(f"不透明度: 原范围={opacity_activated.min():.6f} 到 {opacity_activated.max():.6f}, 转换后={opacity.min():.6f} 到 {opacity.max():.6f}")
    logger.info(f"全局旋转形状: {global_orient.shape}, 关节旋转形状: {body_pose.shape}")
    
    return instance_data

def main(args):
    # 1. 初始化SMPL训练器 
    config_dir = os.path.dirname(args.resume_from)
    config_path = os.path.join(config_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.merge_with(OmegaConf.from_dotlist(args.opts))

    log_dir = os.path.dirname(args.resume_from)
    output_dir = os.path.join(log_dir, "replaced_gart_output")
    os.makedirs(output_dir, exist_ok=True)

    from datasets.driving_dataset import DrivingDataset
    dataset = DrivingDataset(data_cfg=cfg.data)
    
    from models.trainers import BasicTrainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device="cuda",
    )
    trainer.resume_from_checkpoint(args.resume_from, load_only_model=True)
    trainer.eval()

    # 2. 加载GART模型 (viz_human_all方式)
    from third_party.GART.solver import TGFitter
    logger.info(f"Loading GART model from {args.gart_model_dir}")
    
    gart_solver = TGFitter(
        log_dir=args.gart_model_dir,
        profile_fn=os.path.join(args.gart_model_dir, "gen.yaml"),
        mode="human",
        device="cuda",
        NO_TB=True
    )
    model=gart_solver.load_saved_model(os.path.join(args.gart_model_dir, "model.pth"))

    # 3. 提取GART参数并替换SMPL节点
    smpl_nodes = trainer.models["SMPLNodes"]
    num_frames = smpl_nodes.instances_quats.shape[0]
    gart_instance = convert_gart_to_smpl(gart_solver, num_frames)

    # 4. 执行替换 (使用通用模块中的函数)
    with torch.no_grad():
        replace_smpl_instance_improved(
            trainer=trainer,
            instance_id=args.instance_id,
            new_instance=gart_instance,
            keep_translation=args.keep_translation,
            keep_global_rot=args.keep_global_rot
        )

    # 5. 批量渲染 (使用通用模块中的函数)
    logger.info(f"Rendering to {output_dir}")
    batch_render_with_eval(cfg, trainer, dataset, output_dir, log_dir)
    logger.info(f"Rendering completed at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Replace SMPL with GART model")
    parser.add_argument("--resume_from", type=str, required=True, help="Path to SMPL checkpoint")
    parser.add_argument("--gart_model_dir", type=str, required=True, help="Path to GART model directory")
    parser.add_argument("--instance_id", type=int, default=0, help="SMPL instance ID to replace")
    parser.add_argument("--keep_translation", action="store_true", help="Keep original translations")
    parser.add_argument("--keep_global_rot", action="store_true", help="Keep original global rotations")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Config overrides")
    args = parser.parse_args()
    
    main(args)