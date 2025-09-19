from typing import List, Dict, Optional
from omegaconf import OmegaConf
import os
import time
import logging
import argparse
import torch
import numpy as np
from tqdm import tqdm

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import render_images, save_videos

# 导入新的通用模块
from scene_editing.scene_editing import (
    get_full_smpl_sequence,
    load_pose_sequence,
    replace_smpl_pose,
    reverse_sequence,
    apply_reversed_sequence,
    batch_render_with_eval
)

logger = logging.getLogger()

def main(args):
    # 初始化配置
    config_dir = os.path.dirname(args.resume_from)
    config_path = os.path.join(config_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.merge_with(OmegaConf.from_dotlist(args.opts))

    # 设置日志目录（不依赖cfg.logging.log_dir）
    log_dir = os.path.dirname(args.resume_from)
    output_dir = os.path.join(log_dir, "reversed_motion_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化数据集和训练器
    dataset = DrivingDataset(data_cfg=cfg.data)
    
    # 初始化训练器
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
    
    # 设置为评估模式，禁用梯度计算
    trainer.eval()
    # 加载新姿势序列
    if args.new_pose_path:
        logger.info(f"从 {args.new_pose_path} 加载新姿势序列")
        new_poses = load_pose_sequence(args.new_pose_path)
    else:
        new_poses = None
    # 处理运动序列
    with torch.no_grad():
        if new_poses is not None:
            logger.info(f"替换实例 {args.instance_id} 的SMPL姿势")
            replace_smpl_pose(
                trainer, 
                args.instance_id, 
                new_poses,
                keep_translation=args.keep_translation,
                keep_global_rot=args.keep_global_rot
            )
        else:
            logger.info(f"获取实例 {args.instance_id} 的SMPL序列")
            original_seq = get_full_smpl_sequence(trainer, args.instance_id)
            logger.info("反转序列")
            reversed_seq = reverse_sequence(original_seq)
            logger.info("应用反转序列到模型")
            apply_reversed_sequence(trainer, args.instance_id, reversed_seq)

    # 批量渲染
    logger.info(f"开始渲染到: {output_dir}")
    batch_render_with_eval(cfg, trainer, dataset, output_dir, log_dir, post_fix="_reversed")
    logger.info(f"渲染完成，保存到: {output_dir}")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    parser = argparse.ArgumentParser("Replace or reverse SMPL motion sequence")
    parser.add_argument(
        "--resume_from", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--instance_id", type=int, default=0, help="SMPL instance ID to edit"
    )
    parser.add_argument(
        "--new_pose_path", type=str, default="", help="Path to .npy file with new poses"
    )
    parser.add_argument(
        "--keep_translation", action="store_true", help="Keep original translations"
    )
    parser.add_argument(
        "--keep_global_rot", action="store_true", help="Keep original global rotations"
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Config overrides")

    args = parser.parse_args()
    main(args)