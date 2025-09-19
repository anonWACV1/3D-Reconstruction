from typing import List
import os
import cv2
import json
import logging
import numpy as np
from tqdm import tqdm

from utils.geometry import get_corners, project_camera_points_to_image

from .kitti_sourceloader import (
    SMPLNODE_CLASSES,
    OPENCV2DATASET,
    AVAILABLE_CAM_LIST,
)

# 配置日志，仅输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

CAMERA_LIST = AVAILABLE_CAM_LIST

def project_human_boxes(
    scene_dir: str,
    camera_list: List[int],
    save_temp=True,
    verbose=False,
    narrow_width_ratio=0.2,
    fps=12,
):
    """处理Custom KITTI数据集的人体框投影"""
    logger.info(f"开始处理场景: {scene_dir}")
    logger.info(f"使用的相机列表: {camera_list}")
    logger.info(f"保存临时文件: {save_temp}")
    logger.info(f"详细模式: {verbose}")

    # 验证路径结构
    required_dirs = {
        "images": f"{scene_dir}/images",
        "ego_pose": f"{scene_dir}/ego_pose",
        "extrinsics": f"{scene_dir}/extrinsics",
        "intrinsics": f"{scene_dir}/intrinsics",
        "instances": f"{scene_dir}/instances",
    }

    logger.info("验证路径是否存在:")
    for name, path in required_dirs.items():
        exists = os.path.exists(path)
        logger.info(f"- {name}: {path} => {'存在' if exists else '不存在'}")
        if not exists:
            raise FileNotFoundError(f"Missing {name} directory: {path}")

    # 初始化保存目录
    save_dir = f"{scene_dir}/humanpose/temp/Pedes_GTTracks"
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"创建保存目录: {save_dir}")

    # 可视化相关目录
    if verbose:
        video_dir = f"{save_dir}/vis"
        per_human_img_dir = f"{video_dir}/images"
        os.makedirs(per_human_img_dir, exist_ok=True)
        logger.info(f"创建可视化目录: {video_dir}")

    # 加载实例元数据
    logger.info("加载实例元数据")
    try:
        with open(f'{required_dirs["instances"]}/frame_instances.json') as f:
            frame_infos = json.load(f)
        with open(f'{required_dirs["instances"]}/instances_info.json') as f:
            instances_meta = json.load(f)
        logger.info("实例元数据加载成功")
    except Exception as e:
        logger.error(f"加载实例元数据失败: {str(e)}")
        raise

    collector_all = {}

    # 处理每个相机
    for cam_id in tqdm(camera_list, desc="Processing cameras"):
        logger.info(f"开始处理相机 {cam_id}")
        pkl_path = os.path.join(save_dir, f"{cam_id}.pkl")

        if os.path.exists(pkl_path):
            logger.info(f"找到已处理的相机 {cam_id} 数据，跳过")
            with open(pkl_path) as f:
                collector_all[cam_id] = json.load(f)
            continue

        collector = {}
        frames = []

        # 加载相机参数
        logger.debug(f"加载相机 {cam_id} 的参数")
        extrinsic_path = f'{required_dirs["extrinsics"]}/{cam_id}.txt'
        extrinsic = np.loadtxt(extrinsic_path)
        cam2world = np.linalg.inv(extrinsic @ OPENCV2DATASET)

        intrinsic_path = f'{required_dirs["intrinsics"]}/{cam_id}.txt'
        K = np.loadtxt(intrinsic_path)

        # 处理每帧数据
        for frame_id, frame_ins_list in frame_infos.items():
            logger.debug(f"处理帧 {frame_id}")
            frame_id = int(frame_id)
            frame_collector = {
                "gt_bbox": [],
                "extra_data": {
                    "gt_track_id": [],
                    "gt_class": [],
                },
            }

            # 加载位姿
            ego_pose_path = f'{required_dirs["ego_pose"]}/{frame_id:03d}.txt'
            ego_pose = np.loadtxt(ego_pose_path)
            cam2world = np.linalg.inv(ego_pose) @ cam2world

            # 处理每个实例
            for instance_id in frame_ins_list:
                logger.debug(f"处理实例 {instance_id}")
                ins = instances_meta[str(instance_id)]
                
                if ins["class_name"] not in SMPLNODE_CLASSES:
                    logger.debug(f"跳过非人体实例 {instance_id} ({ins['class_name']})")
                    continue

                # 获取实例变换参数
                idx = ins["frame_annotations"]["frame_idx"].index(frame_id)
                obj_to_world = np.array(ins["frame_annotations"]["obj_to_world"][idx])
                l, w, h = ins["frame_annotations"]["box_size"][idx]

                # 3D框投影
                corners = get_corners(l, w, h)
                corners_world = obj_to_world[:3, :3] @ corners + obj_to_world[:3, 3:4]
                world2cam = np.linalg.inv(cam2world)
                corners_cam = world2cam[:3, :3] @ corners_world + world2cam[:3, 3:4]
                cam_points, _ = project_camera_points_to_image(corners_cam.T, K)

                # 生成2D边界框
                x_min, y_min = np.min(cam_points, axis=0)
                x_max, y_max = np.max(cam_points, axis=0)
                if narrow_width_ratio > 0:
                    center_x = (x_min + x_max) / 2
                    new_width = (x_max - x_min) * narrow_width_ratio
                    x_min = center_x - new_width / 2
                    x_max = center_x + new_width / 2

                # 保存结果
                frame_collector["gt_bbox"].append([x_min, y_min, x_max, y_max])
                frame_collector["extra_data"]["gt_track_id"].append(instance_id)
                frame_collector["extra_data"]["gt_class"].append(ins["class_name"])

            frames.append(frame_collector)

        # 保存相机结果
        collector[cam_id] = frames
        if save_temp:
            logger.info(f"保存相机 {cam_id} 的处理结果到 {pkl_path}")
            with open(pkl_path, "w") as f:
                json.dump(collector, f)
        collector_all.update(collector)

    logger.info(f"场景 {scene_dir} 处理完成")
    return collector_all