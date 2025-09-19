import os
import json
import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from typing import Dict, List, Optional, Tuple

from datasets.scene_base import SceneCamera, ScenePixelSource
from utils.camera_utils import undistort_image
from utils.geometry import (
    get_rays_np,
    get_rays_torch,
    get_ray_directions,
    get_ray_directions_torch,
    get_ndc_rays_fx_fy,
)

logger = logging.getLogger(__name__)

# 坐标系转换矩阵 (根据实际传感器配置调整)
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
)

# 可用相机列表 (根据实际相机配置调整)
AVAILABLE_CAM_LIST = ["0", "1"]  # 双目相机配置
OBJECT_CLASS_NODE_MAPPING = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}


class CustomKITTICameraData(SceneCamera):
    """Custom KITTI相机数据加载器"""

    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        cam_id: str,
        start_timestep: int,
        end_timestep: int,
        load_dynamic_mask: bool = False,
        load_sky_mask: bool = False,
        downscale_when_loading: float = 1.0,
        undistort: bool = False,
        buffer_downscale: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            dataset_name=dataset_name,
            cam_id=cam_id,
            cam_name=f"custom_{cam_id}",
            device=device,
        )

        # 初始化参数
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.load_dynamic_mask = load_dynamic_mask
        self.load_sky_mask = load_sky_mask
        self.downscale = downscale_when_loading
        self.undistort = undistort
        self.buffer_downscale = buffer_downscale

        # 验证路径
        self._validate_paths()

        # 加载数据
        self.load_camera_parameters()
        self.load_images()
        if load_dynamic_mask or load_sky_mask:
            self.load_masks()

    def _validate_paths(self):
        """验证必要路径是否存在"""
        required_dirs = [
            os.path.join(self.data_path, "images"),
            os.path.join(self.data_path, "ego_pose"),
            os.path.join(self.data_path, "extrinsics"),
            os.path.join(self.data_path, "intrinsics"),
        ]

        for d in required_dirs:
            if not os.path.exists(d):
                raise FileNotFoundError(f"Missing required directory: {d}")

    def load_camera_parameters(self):
        """加载相机参数"""
        # 加载外参
        cam_to_ego = np.loadtxt(
            os.path.join(self.data_path, "extrinsics", f"{self.cam_id}.txt")
        )
        self.cam_to_ego = cam_to_ego @ OPENCV2DATASET

        # 加载内参
        K = np.loadtxt(os.path.join(self.data_path, "intrinsics", f"{self.cam_id}.txt"))
        self._intrinsics = np.array(
            [[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]], dtype=np.float32
        )

        # 加载畸变参数
        self._distortions = K[4:8] if len(K) >= 8 else np.zeros(4)

    def load_images(self):
        """加载图像数据"""
        image_dir = os.path.join(self.data_path, "images", self.cam_id)
        self.image_paths = [
            os.path.join(image_dir, f"{t:06d}.png")
            for t in range(self.start_timestep, self.end_timestep)
        ]

        # 初始化存储
        self.images = []
        self.H, self.W = None, None

        # 加载并处理图像
        for path in self.image_paths:
            img = Image.open(path)
            if self.downscale != 1.0:
                new_size = (
                    int(img.width * self.downscale),
                    int(img.height * self.downscale),
                )
                img = img.resize(new_size, Image.LANCZOS)

            if self.undistort:
                img = undistort_image(
                    np.array(img), self._intrinsics, self._distortions
                )
                img = Image.fromarray(img)

            self.images.append(img)

            if self.H is None:
                self.H, self.W = img.height, img.width

        # 转换为tensor
        self.images = torch.stack(
            [torch.from_numpy(np.array(img)).float() / 255.0 for img in self.images]
        ).to(self.device)

    def load_masks(self):
        """加载掩码数据"""
        # 动态物体掩码
        if self.load_dynamic_mask:
            mask_dir = os.path.join(self.data_path, "fine_dynamic_mask")
            self.dynamic_masks = self._load_mask_type(mask_dir)

        # 天空掩码
        if self.load_sky_mask:
            sky_dir = os.path.join(self.data_path, "sky_masks")
            self.sky_masks = self._load_mask_type(sky_dir)

    def _load_mask_type(self, mask_dir: str) -> torch.Tensor:
        """通用掩码加载方法"""
        mask_paths = [
            os.path.join(mask_dir, f"{t:06d}_{self.cam_id}.png")
            for t in range(self.start_timestep, self.end_timestep)
        ]

        masks = []
        for path in mask_paths:
            mask = Image.open(path).convert("L")
            if self.downscale != 1.0:
                new_size = (
                    int(mask.width * self.downscale),
                    int(mask.height * self.downscale),
                )
                mask = mask.resize(new_size, Image.NEAREST)

            masks.append(torch.from_numpy(np.array(mask)).float() / 255.0)

        return torch.stack(masks).to(self.device)


class CustomKITTIPixelSource(ScenePixelSource):
    """Custom KITTI像素数据源"""

    def __init__(
        self,
        dataset_name: str,
        pixel_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.load_data()

    def load_cameras(self):
        """加载所有相机数据"""
        self._timesteps = torch.arange(self.start_timestep, self.end_timestep)
        self.register_normalized_timestamps()

        for idx, cam_id in enumerate(AVAILABLE_CAM_LIST):
            logger.info(f"加载相机 {cam_id}")
            camera = CustomKITTICameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                load_dynamic_mask=self.data_cfg.load_dynamic_mask,
                load_sky_mask=self.data_cfg.load_sky_mask,
                downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                undistort=self.data_cfg.undistort,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = (
                torch.arange(len(camera), device=self.device) * len(AVAILABLE_CAM_LIST)
                + idx
            )
            camera.set_unique_ids(unique_cam_idx=idx, unique_img_idx=unique_img_idx)
            logger.info(f"相机 {camera.cam_name} 加载完成")
            self.camera_data[cam_id] = camera

    def load_objects(self):
        """加载物体实例数据"""
        instances_info_path = os.path.join(
            self.data_path, "instances", "instances_info.json"
        )
        frame_instances_path = os.path.join(
            self.data_path, "instances", "frame_instances.json"
        )

        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)
        with open(frame_instances_path, "r") as f:
            frame_instances = json.load(f)

        # 处理实例姿态数据
        num_instances = len(instances_info["instances"])
        num_frames = self.end_timestep - self.start_timestep

        # 初始化存储
        instances_pose = torch.zeros((num_frames, num_instances, 4, 4))
        instances_size = torch.zeros((num_frames, num_instances, 3))
        per_frame_instance_mask = torch.zeros(
            (num_frames, num_instances), dtype=torch.bool
        )

        # 加载初始位姿
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )

        # 处理每个实例
        for ins_id, ins_data in instances_info["instances"].items():
            ins_idx = int(ins_id)
            class_name = ins_data["class_name"]

            # 对齐时间戳
            for frame_idx, obj_to_world, box_size in zip(
                ins_data["frame_annotations"]["frame_idx"],
                ins_data["frame_annotations"]["obj_to_world"],
                ins_data["frame_annotations"]["box_size"],
            ):
                if frame_idx < self.start_timestep or frame_idx >= self.end_timestep:
                    continue

                # 坐标系转换
                rel_frame_idx = frame_idx - self.start_timestep
                obj_to_world = np.array(obj_to_world).reshape(4, 4)
                obj_to_world = np.linalg.inv(ego_to_world_start) @ obj_to_world

                # 存储数据
                instances_pose[rel_frame_idx, ins_idx] = torch.from_numpy(
                    obj_to_world
                ).float()
                instances_size[rel_frame_idx, ins_idx] = torch.tensor(box_size).float()
                per_frame_instance_mask[rel_frame_idx, ins_idx] = True

        # 过滤不可见实例
        ins_frame_cnt = per_frame_instance_mask.sum(dim=0)
        valid_ins = ins_frame_cnt > 0

        self.instances_pose = instances_pose[:, valid_ins]
        self.instances_size = instances_size[:, valid_ins].sum(0) / ins_frame_cnt[
            valid_ins
        ].unsqueeze(-1)
        self.per_frame_instance_mask = per_frame_instance_mask[:, valid_ins]
        self.instances_true_id = torch.arange(num_instances)[valid_ins]
        self.instances_model_types = torch.tensor(
            [
                OBJECT_CLASS_NODE_MAPPING[
                    instances_info["instances"][str(i)]["class_name"]
                ]
                for i in self.instances_true_id.tolist()
            ]
        )
