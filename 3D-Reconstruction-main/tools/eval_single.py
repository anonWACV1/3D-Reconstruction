from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import argparse
import cv2
import numpy as np
import torch

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import render_images, save_videos, render_novel_views

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


# Info printing functions
def print_rigid_info(trainer, image_idx):
    """打印刚体节点信息"""
    rigid_info = trainer.get_rigid_info(frame_idx=image_idx)
    if rigid_info is not None:
        print("\n=== Rigid Objects Information ===")
        print(f"Frame: {image_idx}")

        # 打印位置信息
        positions = rigid_info["positions"]
        print(f"\nNumber of points: {len(positions)}")
        print(f"Points shape: {positions.shape}")
        print("Sample positions:")
        print(positions[:5])  # 打印前5个点的位置

        # 打印实例ID信息
        if rigid_info["instance_ids"] is not None:
            unique_ids = torch.unique(rigid_info["instance_ids"])
            print(f"\nUnique instance IDs: {unique_ids.cpu().numpy()}")

            # 打印每个实例的统计信息
            for id in unique_ids:
                mask = rigid_info["instance_ids"] == id
                pts = positions[mask]
                center = pts.mean(dim=0)
                print(f"\nInstance {id}:")
                print(f"  Number of points: {mask.sum().item()}")
                print(f"  Center position: {center.cpu().numpy()}")

                # 打印该实例的变换信息
                if "transforms" in rigid_info:
                    print(f"  Transforms:")
                    transforms = rigid_info["transforms"]
                    print(f"    Rotation matrix:")
                    print(transforms["rotations"][id].cpu().numpy())
                    print(f"    Translation vector:")
                    print(transforms["translations"][id].cpu().numpy())
                    print(f"    Quaternion (wxyz):")
                    print(transforms["quaternions"][id].cpu().numpy())


def print_smpl_info(trainer, image_idx):
    """打印SMPL节点信息"""
    smpl_info = trainer.get_smpl_info(frame_idx=image_idx)
    if smpl_info is not None:
        print("\n=== SMPL Information ===")
        print(f"Frame: {image_idx}")

        # 打印位置信息
        positions = smpl_info["positions"]
        print(f"\nNumber of points: {len(positions)}")
        print(f"Points shape: {positions.shape}")

        # 打印实例ID信息
        if smpl_info["instance_ids"] is not None:
            unique_ids = torch.unique(smpl_info["instance_ids"])
            print(f"\nUnique instance IDs: {unique_ids.cpu().numpy()}")

            # 打印每个实例的统计信息
            for id in unique_ids:
                mask = smpl_info["instance_ids"] == id
                pts = positions[mask]
                center = pts.mean(dim=0)
                print(f"\nInstance {id}:")
                print(f"  Number of points: {mask.sum().item()}")
                print(f"  Center position: {center.cpu().numpy()}")

                # 打印该实例的变换信息
                if "transforms" in smpl_info:
                    print(f"  Transforms:")
                    transforms = smpl_info["transforms"]
                    print(f"    SMPL joint rotations shape:")
                    print(transforms["smpl_quats"][id].shape)
                    print(f"    Global rotation:")
                    print(transforms["global_quats"][id].cpu().numpy())
                    print(f"    Translation vector:")
                    print(transforms["translations"][id].cpu().numpy())


# Node editing functions
def edit_rigid_nodes(trainer, args):
    """编辑刚体节点"""
    # 获取RigidNodes模型
    rigid_nodes = trainer.models["RigidNodes"]

    # 定义编辑配置
    edit_config = {
        "remove": [0],  # 移除ID为0的实例
        "replace": {1: 2},  # 用ID为2的实例替换ID为1的实例
    }

    # 执行编辑操作
    rigid_nodes.remove_instances(edit_config["remove"])
    # rigid_nodes.replace_instances(edit_config["replace"])

    # 添加平移偏移
    translation_offset = torch.tensor([5.0, -5.0, 0], device=trainer.device)
    rigid_nodes.add_transform_offset(
        instance_id=1, frame_idx=args.image_idx, translation_offset=translation_offset
    )

    return rigid_nodes


def edit_smpl_nodes(trainer, args):
    """编辑 SMPL 节点，并修改 SMPL 参数"""
    # 获取 SMPLNodes 模型
    smpl_nodes = trainer.models["SMPLNodes"]

    # # 打印当前可用实例ID和模板实例数
    # instance_ids = smpl_nodes.point_ids[..., 0].unique()
    # print(f"Available instance IDs: {instance_ids}")
    # template_instances = smpl_nodes.template.A0_inv.shape[0]
    # print(f"Template instances: {template_instances}")

    # 例如，移除实例 [5]（根据需要选择是否删除）
    edit_config = {"remove": [5]}
    print(f"Removing instances: {edit_config['remove']}")
    # 若需要删除实例，则执行：
    # smpl_nodes.remove_instances(edit_config["remove"])

    # 为实例 0 添加平移偏移
    translation_offset = torch.tensor([2.0, 0, 0], device=trainer.device)
    smpl_nodes.add_transform_offset(
        instance_id=0,
        frame_idx=args.image_idx,
        translation_offset=translation_offset,
    )

    # # 修改实例 0 的 SMPL 姿态
    # theta = torch.zeros(24, 4, device=trainer.device)  # 初始化全部为零（T-pose）
    # theta[:, 0] = 1  # 将四元数的 w 分量设为 1
    # smpl_nodes.set_smpl_params(instance_id=0, frame_idx=args.image_idx, theta=theta)

    # # 修改实例 0 的 SMPL 形状参数（beta）
    # beta = torch.randn(10, device=trainer.device) * 0.1  # 随机形状参数
    # smpl_nodes.set_smpl_params(instance_id=0, frame_idx=args.image_idx, beta=beta)

    return smpl_nodes


# Rendering functions
@torch.no_grad()
def render_single_image(
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    image_idx: int = 0,
    checkpoint_dir: str = None,
    use_test_set: bool = True,
):
    """渲染单张图像"""
    # 选择数据集
    if use_test_set and dataset.test_image_set is not None:
        image_data = dataset.test_image_set
    else:
        image_data = dataset.full_image_set

    if image_data is None:
        raise ValueError("Neither test_image_set nor full_image_set is available")

    # 确保索引在有效范围内
    if image_idx >= len(image_data):
        raise ValueError(
            f"Image index {image_idx} out of range. Total images: {len(image_data)}"
        )

    # 创建输出目录
    output_dir = os.path.join(checkpoint_dir, "outputImage", str(image_idx))
    os.makedirs(output_dir, exist_ok=True)

    # 创建一个只包含这一张图片的数据集
    from datasets.base.split_wrapper import SplitWrapper

    single_image_dataset = SplitWrapper(
        datasource=image_data.datasource,
        split_indices=[image_data.split_indices[image_idx]],
    )

    # 打印节点信息
    # print_rigid_info(trainer, image_idx)
    # print_smpl_info(trainer, args.image_idx)

    # 渲染图像
    render_results = render_images(
        trainer=trainer,
        dataset=single_image_dataset,
        compute_metrics=False,
        compute_error_map=False,
    )

    # 保存渲染结果
    save_render_results(render_results, output_dir)


def save_render_results(render_results, output_dir):
    """保存渲染结果"""
    # 保存RGB图像
    rgb_image = render_results["rgbs"][0]
    rgb_path = os.path.join(output_dir, "rgb.png")
    cv2.imwrite(
        rgb_path, cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    logger.info(f"Rendered RGB image saved to {rgb_path}")

    # 保存深度图
    if "depths" in render_results:
        save_depth_image(
            render_results["depths"][0],
            os.path.join(output_dir, "depth_raw.png"),
            os.path.join(output_dir, "depth_vis.png"),
        )

    # 保存其他深度图
    for key in render_results.keys():
        if key in ["rgbs", "depths"] or not key.endswith("_depths"):
            continue
        save_depth_image(
            render_results[key][0],
            os.path.join(output_dir, f"{key}_raw.png"),
            os.path.join(output_dir, f"{key}_vis.png"),
            key,
        )


def save_depth_image(depth_image, raw_path, vis_path, prefix="Depth"):
    """保存深度图"""
    depth_min = np.min(depth_image)
    depth_max = np.max(depth_image)
    depth_norm = (depth_image - depth_min) / (depth_max - depth_min)

    # 保存原始深度图
    cv2.imwrite(raw_path, (depth_norm * 65535).astype(np.uint16))

    # 保存可视化的深度图
    depth_colormap = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    cv2.imwrite(vis_path, depth_colormap)

    logger.info(f"{prefix} maps saved to {raw_path} and {vis_path}")
    logger.info(f"{prefix} range: {depth_min:.4f} - {depth_max:.4f}")


def inspect_instances(trainer):
    """查看实例信息"""
    smpl_nodes = trainer.models["SMPLNodes"]
    instance_ids = smpl_nodes.point_ids[..., 0].unique()

    for id in instance_ids:
        pts_mask = smpl_nodes.point_ids[..., 0] == id
        num_points = pts_mask.sum().item()

        points = smpl_nodes._means[pts_mask]
        center = points.mean(dim=0)

        print(f"Instance {id}:")
        print(f"  Points: {num_points}")
        print(f"  Center: {center}")
    return smpl_nodes


def main(args):
    """主函数"""
    # 加载配置
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    args.enable_wandb = False

    # 创建输出目录
    for folder in ["videos_eval", "metrics_eval"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建数据集
    dataset = DrivingDataset(data_cfg=cfg.data)

    # 设置训练器
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )

    # 从检查点恢复
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)
    logger.info(f"Resuming from {args.resume_from}")

    # 检查实例信息
    # smpl_nodes = inspect_instances(trainer)

    # 根据模型类型执行相应的编辑操作
    if "RigidNodes" in trainer.models and args.edit_rigid:
        rigid_nodes = edit_rigid_nodes(trainer, args)

    if "SMPLNodes" in trainer.models and args.edit_smpl:
        smpl_nodes = edit_smpl_nodes(trainer, args)

    # 渲染图像
    render_single_image(
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        image_idx=args.image_idx,
        checkpoint_dir=log_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    # eval
    parser.add_argument(
        "--resume_from",
        default=None,
        help="path to checkpoint to resume from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--image_idx", type=int, default=0, help="index of the image to render"
    )

    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")

    # misc
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # 编辑相关参数
    parser.add_argument(
        "--edit_rigid", action="store_true", default=False, help="是否编辑刚体节点"
    )
    parser.add_argument(
        "--edit_smpl", action="store_true", default=False, help="是否编辑SMPL节点"
    )

    args = parser.parse_args()
    main(args)
