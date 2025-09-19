import os
import argparse
import logging
from omegaconf import OmegaConf
import torch

from utils.misc import export_points_to_ply, import_str
from datasets.driving_dataset import DrivingDataset

logger = logging.getLogger()


def export_gaussians_to_ply(
    model,
    save_dir: str,
    alpha_thresh: float = 0.01,
    instance_id: int = None,
    frame_idx: int = 0,
    normalize: bool = False,
    export_all_frames: bool = False,
):
    """
    将高斯体导出为PLY文件

    Args:
        model: 高斯模型
        save_dir: 保存目录
        alpha_thresh: 透明度阈值，用于过滤点
        instance_id: 实例ID，如果指定则只导出该实例
        frame_idx: 帧索引，用于SMPL模型
        normalize: 是否将点云归一化到[0,1]范围
        export_all_frames: 是否导出所有帧的点云
    """
    os.makedirs(save_dir, exist_ok=True)

    if hasattr(model, "smpl_qauts"):  # SMPLNodes
        if export_all_frames:
            # 导出所有帧
            for frame in range(model.num_frames):
                if instance_id is not None:
                    if not model.instances_fv[frame, instance_id]:
                        continue
                    point_data = model.export_gaussians_to_ply(
                        alpha_thresh=alpha_thresh,
                        instance_id=instance_id,
                        specific_frame=frame,
                    )
                    save_path = os.path.join(
                        save_dir, f"frame_{frame}_instance_{instance_id}.ply"
                    )
                    export_points_to_ply(
                        positions=point_data["positions"],
                        colors=point_data["colors"],
                        save_path=save_path,
                        normalize=normalize,
                    )
                    logger.info(
                        f"Exported frame {frame} instance {instance_id} to {save_path}"
                    )
        else:
            # 只导出指定帧
            point_data = model.export_gaussians_to_ply(
                alpha_thresh=alpha_thresh,
                instance_id=instance_id,
                specific_frame=frame_idx,
            )
            save_path = os.path.join(
                save_dir, f"frame_{frame_idx}_instance_{instance_id}.ply"
            )
            export_points_to_ply(
                positions=point_data["positions"],
                colors=point_data["colors"],
                save_path=save_path,
                normalize=normalize,
            )
            logger.info(
                f"Exported frame {frame_idx} instance {instance_id} to {save_path}"
            )

    elif hasattr(model, "point_ids"):  # RigidNodes
        point_data = model.export_gaussians_to_ply(
            alpha_thresh=alpha_thresh, instance_id=instance_id
        )
        if instance_id is not None:
            save_path = os.path.join(save_dir, f"instance_{instance_id}.ply")
        else:
            save_path = os.path.join(save_dir, "all_instances.ply")
        export_points_to_ply(
            positions=point_data["positions"],
            colors=point_data["colors"],
            save_path=save_path,
            normalize=normalize,
        )
        logger.info(f"Exported to {save_path}")

    else:  # VanillaGaussians
        point_data = model.export_gaussians_to_ply(alpha_thresh=alpha_thresh)
        save_path = os.path.join(save_dir, "gaussians.ply")
        export_points_to_ply(
            positions=point_data["positions"],
            colors=point_data["colors"],
            save_path=save_path,
            normalize=normalize,
        )
        logger.info(f"Exported to {save_path}")


def main(args):
    # 加载配置
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建数据集
    dataset = DrivingDataset(data_cfg=cfg.data)

    # 设置trainer
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

    # 从checkpoint恢复
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)
    logger.info(f"Resumed from checkpoint: {args.resume_from}")

    # 获取要导出的模型
    if args.model_name is not None:
        model = trainer.models[args.model_name]
    else:
        model = trainer.models[list(trainer.models.keys())[0]]

    # 导出PLY文件
    export_gaussians_to_ply(
        model=model,
        save_dir=args.save_dir,
        alpha_thresh=args.alpha_thresh,
        instance_id=args.instance_id,
        frame_idx=args.frame_idx,
        normalize=args.normalize,
        export_all_frames=args.export_all_frames,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Gaussian model to PLY file")

    # 必需参数
    parser.add_argument(
        "--resume_from",
        type=str,
        required=True,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save PLY files"
    )

    # 可选参数
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of model to export (e.g., 'SMPLNodes', 'RigidNodes')",
    )
    parser.add_argument(
        "--alpha_thresh",
        type=float,
        default=0.01,
        help="Opacity threshold for filtering points",
    )
    parser.add_argument(
        "--instance_id", type=int, default=None, help="ID of instance to export"
    )
    parser.add_argument(
        "--frame_idx", type=int, default=0, help="Frame index for SMPL model export"
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize point cloud to [0,1] range"
    )
    parser.add_argument(
        "--export_all_frames",
        action="store_true",
        help="Export all frames for SMPL model",
    )

    # 其他配置选项
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    args = parser.parse_args()
    main(args)
