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

logger = logging.getLogger()


def get_full_smpl_sequence(trainer, instance_id: int) -> Dict:
    """获取完整的SMPL运动序列"""
    smpl_nodes = trainer.models["SMPLNodes"]
    sequence = {
        "global_quats": smpl_nodes.instances_quats[:, instance_id].clone(),
        "translations": smpl_nodes.instances_trans[:, instance_id].clone(),
        "smpl_quats": smpl_nodes.smpl_qauts[:, instance_id].clone(),
        "betas": (
            smpl_nodes.betas[instance_id].clone()
            if hasattr(smpl_nodes, "betas")
            else None
        ),
    }
    return sequence


def reverse_sequence(sequence: Dict) -> Dict:
    """反转运动序列"""
    return {
        "global_quats": sequence["global_quats"].flip(0),
        "translations": sequence["translations"].flip(0),
        "smpl_quats": sequence["smpl_quats"].flip(0),
        "betas": sequence["betas"] if sequence["betas"] is not None else None,
    }


def apply_reversed_sequence(trainer, instance_id: int, reversed_seq: Dict):
    """应用反转后的序列到模型"""
    smpl_nodes = trainer.models["SMPLNodes"]

    # 更新模型参数，使用copy_方法避免in-place操作
    with torch.no_grad():
        smpl_nodes.instances_quats[:, instance_id].copy_(reversed_seq["global_quats"])
        smpl_nodes.instances_trans[:, instance_id].copy_(reversed_seq["translations"])
        smpl_nodes.smpl_qauts[:, instance_id].copy_(reversed_seq["smpl_quats"])

        if reversed_seq["betas"] is not None:
            smpl_nodes.betas[instance_id].copy_(reversed_seq["betas"])


def batch_render_with_eval(cfg, trainer, dataset, output_dir: str, log_dir: str):
    """使用eval.py的渲染逻辑进行批量渲染"""
    from tools.eval import do_evaluation

    # 创建伪args对象
    class Args:
        def __init__(self):
            self.save_catted_videos = True
            self.enable_viewer = False
            self.render_video_postfix = "_reversed"

    fake_args = Args()
    
    # 确保视频目录存在
    videos_dir = os.path.join(log_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # 创建reversed目录
    videos_reversed_dir = os.path.join(log_dir, "videos_reversed")
    os.makedirs(videos_reversed_dir, exist_ok=True)
    
    # 为其他渲染键创建目录
    for key in ["rgbs", "depths"]:
        os.makedirs(os.path.join(log_dir, key), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"{key}_reversed"), exist_ok=True)

    # 修改cfg以确保log_dir正确设置
    if not hasattr(cfg, "logging"):
        cfg.logging = OmegaConf.create({})
    cfg.logging.log_dir = log_dir

    # 执行渲染
    try:
        do_evaluation(
            step=trainer.step,
            cfg=cfg,
            trainer=trainer,
            dataset=dataset,
            args=fake_args,
            render_keys=["rgbs", "depths"],
            post_fix="_reversed",
            log_metrics=False,
        )

        # 移动渲染结果到指定目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查videos目录中的文件
        if os.path.exists(videos_dir):
            for f in os.listdir(videos_dir):
                if "reversed" in f:
                    src_file = os.path.join(videos_dir, f)
                    dst_file = os.path.join(output_dir, f)
                    logger.info(f"移动文件: {src_file} -> {dst_file}")
                    os.rename(src_file, dst_file)
        
        # 检查videos_reversed目录中的文件
        if os.path.exists(videos_reversed_dir):
            for f in os.listdir(videos_reversed_dir):
                dst_file = os.path.join(output_dir, f)
                src_file = os.path.join(videos_reversed_dir, f)
                logger.info(f"移动文件: {src_file} -> {dst_file}")
                os.rename(src_file, dst_file)
                
    except Exception as e:
        logger.error(f"渲染过程中发生错误: {str(e)}")
        # 尝试直接渲染
        try:
            logger.info("尝试直接渲染...")
            # 获取测试视图的渲染
            views = dataset.test_timesteps
            all_rgbs = []
            all_depths = []
            
            for view_id in tqdm(views, desc="渲染视图"):
                with torch.no_grad():
                    outputs = trainer.render_view(view_id)
                    all_rgbs.append(outputs["rgb"].cpu().numpy())
                    if "depth" in outputs:
                        all_depths.append(outputs["depth"].cpu().numpy())
            
            # 保存视频
            os.makedirs(output_dir, exist_ok=True)
            if all_rgbs:
                rgb_video_path = os.path.join(output_dir, "test_rgbs_reversed.mp4")
                save_videos(all_rgbs, rgb_video_path)
                logger.info(f"RGB视频保存到: {rgb_video_path}")
            
            if all_depths:
                depth_video_path = os.path.join(output_dir, "test_depths_reversed.mp4")
                save_videos(all_depths, depth_video_path)
                logger.info(f"深度视频保存到: {depth_video_path}")
        except Exception as e2:
            logger.error(f"直接渲染也失败: {str(e2)}")


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
    
    # 处理运动序列
    with torch.no_grad():
        logger.info(f"获取实例 {args.instance_id} 的SMPL序列")
        original_seq = get_full_smpl_sequence(trainer, args.instance_id)
        logger.info("反转序列")
        reversed_seq = reverse_sequence(original_seq)
        logger.info("应用反转序列到模型")
        apply_reversed_sequence(trainer, args.instance_id, reversed_seq)

    # 批量渲染
    logger.info(f"开始渲染到: {output_dir}")
    batch_render_with_eval(cfg, trainer, dataset, output_dir, log_dir)
    logger.info(f"反转运动渲染完成，保存到: {output_dir}")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser("Reverse SMPL motion sequence and batch render")
    parser.add_argument(
        "--resume_from", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--instance_id", type=int, default=0, help="SMPL instance ID to edit"
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Config overrides")

    args = parser.parse_args()
    main(args)