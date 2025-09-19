#!/usr/bin/env python3
"""
行人提前出现编辑工具，实现让行人提前出现并按原有轨迹运动。

使用示例:
export PYTHONPATH=$(pwd)
python scene_editing/human_translation.py \
    --resume_from output/waymo_3cams/dataset=waymo/5cams_788/checkpoint_final.pth \
    --instance_id 35 \
    --translation_offset 0.0 20.0 0.0 \
    --early_start_frame 50 \
    --output_dir output/human_early_appear_output
"""

from typing import List, Dict, Tuple, Optional
from omegaconf import OmegaConf
import os
import time
import logging
import argparse
import torch
import numpy as np
from tqdm import tqdm
import math
import json

# 导入项目特定模块
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import render_images, save_videos

# 从scene_editing.py导入所需函数
try:
    from scene_editing.scene_editing import (
        batch_render_with_eval,
        print_node_info,
        save_node_info,
    )
except ImportError:
    from scene_editing import (
        batch_render_with_eval,
        print_node_info,
        save_node_info,
    )

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_valid_frames(positions, threshold=1e-5):
    """
    查找位置张量中有效的帧（非零位置）
    
    Args:
        positions: 位置张量 [num_frames, 3]
        threshold: 判断为非零的阈值
        
    Returns:
        有效帧的起始和结束索引
    """
    # 计算每一帧位置的平方和
    norms = torch.sum(positions**2, dim=1)
    # 找出非零帧的索引
    valid_indices = torch.where(norms > threshold)[0]
    
    if len(valid_indices) == 0:
        return None, None
    
    # 返回起始和结束索引
    return valid_indices[0].item(), valid_indices[-1].item()


def apply_human_early_appear(trainer, args):
    """
    应用提前出现操作到指定行人实例，将有效运动轨迹前移到指定帧并添加偏移
    
    Args:
        trainer: 训练器实例
        args: 命令行参数，包含编辑相关参数

    Returns:
        tuple: (更新后的SMPLNodes对象, 编辑参数字典)
    """
    # 检查是否存在SMPLNodes模型
    if "SMPLNodes" not in trainer.models:
        raise ValueError("模型中没有找到SMPLNodes")

    # 获取SMPLNodes模型
    smpl_nodes = trainer.models["SMPLNodes"]
    logger.info(f"找到SMPLNodes模型，实例数量: {smpl_nodes.num_instances}")

    # 保存编辑参数
    edit_params = {
        "operation": "lane_change",
        "instance_id": args.instance_id,
        "translation_offset": args.translation_offset,
        "early_start_frame": args.early_start_frame,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }

    try:
        # 获取所有实例ID
        all_ids = smpl_nodes.point_ids[..., 0].unique().cpu().numpy()
        logger.info(f"当前所有实例ID: {all_ids}")

        # 检查目标实例ID是否存在
        if args.instance_id not in all_ids:
            logger.error(f"指定的实例ID {args.instance_id} 不存在")
            return smpl_nodes, edit_params

        # 获取设备信息
        device = trainer.device

        # 将平移向量转换为张量
        translation_offset = torch.tensor(
            args.translation_offset, device=device, dtype=torch.float32
        )
        logger.info(f"平移向量: {translation_offset.cpu().numpy()}")

        # 获取帧数
        num_frames = (
            smpl_nodes.num_frames
            if hasattr(smpl_nodes, "num_frames")
            else smpl_nodes.instances_trans.shape[0]
        )
        
        # 提前出现的起始帧
        early_start_frame = max(0, min(args.early_start_frame, num_frames - 1))

        # 应用提前出现和平移偏移
        with torch.no_grad():
            # 获取目标实例的所有帧位置
            instance_id = args.instance_id
            instance_positions = smpl_nodes.instances_trans[:, instance_id].clone()
            
            # 查找有效帧范围
            valid_start, valid_end = find_valid_frames(instance_positions)
            
            if valid_start is None or valid_end is None:
                logger.warning(f"实例 {instance_id} 没有有效位置，无法编辑")
                return smpl_nodes, edit_params
            
            logger.info(f"实例 {instance_id} 原有效帧范围: {valid_start} - {valid_end}")
            logger.info(f"将提前到第 {early_start_frame} 帧出现")
            
            # 计算需要前移的帧数
            frame_shift = valid_start - early_start_frame
            
            # 如果前移为负数（即要求的起始帧晚于实际起始帧），则警告并使用实际起始帧
            if frame_shift <= 0:
                logger.warning(f"指定的提前出现帧 {early_start_frame} 晚于或等于实例实际出现帧 {valid_start}，将使用实际出现帧")
                return smpl_nodes, edit_params
            
            # 获取有效轨迹
            valid_trajectory = instance_positions[valid_start:valid_end+1].clone()
            
            # 打印原始有效轨迹
            logger.info(f"\n===== 实例 {instance_id} 的原始有效轨迹 =====")
            for i in range(0, len(valid_trajectory), 10):  # 每10帧打印一次
                pos = valid_trajectory[i].cpu().numpy()
                frame_idx = valid_start + i
                logger.info(f"帧 {frame_idx}: {pos}")
            
            # 为轨迹添加偏移
            valid_trajectory_with_offset = valid_trajectory + translation_offset
            
            # 创建提前出现的轨迹
            # 保持有效轨迹的长度不变，但起始帧提前
            for i in range(len(valid_trajectory)):
                target_frame = early_start_frame + i
                # 确保不超出最大帧数
                if target_frame >= num_frames:
                    break
                # 应用新轨迹（带偏移）
                smpl_nodes.instances_trans[target_frame, instance_id] = valid_trajectory_with_offset[i]
            
            # 打印编辑后的轨迹
            logger.info(f"\n===== 实例 {instance_id} 提前出现后的轨迹 =====")
            for frame_idx in range(early_start_frame, min(early_start_frame + len(valid_trajectory), num_frames), 10):
                pos = smpl_nodes.instances_trans[frame_idx, instance_id].cpu().numpy()
                logger.info(f"帧 {frame_idx}: {pos}")
            
            # 确保原有效帧范围之后的帧保持不变（可选，取决于需求）
            # 如果要保持原有效帧之后的帧不变，可以取消下面的注释
            '''
            for frame_idx in range(valid_start, num_frames):
                if frame_idx < valid_start + len(valid_trajectory):
                    # 这些帧已经在前面被处理，跳过
                    continue
                # 将原轨迹平移到对应位置
                orig_pos = instance_positions[frame_idx]
                if torch.sum(orig_pos**2) > 1e-5:  # 如果是有效位置
                    smpl_nodes.instances_trans[frame_idx, instance_id] = orig_pos + translation_offset
            '''

        logger.info(f"实例 {args.instance_id} 提前出现编辑已完成")

    except Exception as e:
        logger.error(f"编辑操作过程中出错: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())

    # 返回更新后的模型和编辑参数
    return smpl_nodes, edit_params


def main(args):
    """主函数"""
    # 加载配置
    log_dir = os.path.dirname(args.resume_from)
    config_path = os.path.join(log_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg.merge_with(OmegaConf.from_dotlist(args.opts))

    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(log_dir, f"human_early_appear_{args.instance_id}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # 初始化数据集和训练器
    dataset = DrivingDataset(data_cfg=cfg.data)

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

    # 加载检查点
    trainer.resume_from_checkpoint(args.resume_from, load_only_model=True)
    trainer.eval()
    logger.info(f"已从检查点加载模型: {args.resume_from}")

    # 打印节点信息
    nodes_info = print_node_info(trainer)

    # 执行编辑操作
    with torch.no_grad():
        smpl_nodes, edit_params = apply_human_early_appear(trainer, args)

        # 保存编辑参数
        params_file = os.path.join(output_dir, "human_early_appear_params.json")
        with open(params_file, "w", encoding="utf-8") as f:
            # 处理不可序列化的对象
            def make_serializable(obj):
                if isinstance(obj, (np.ndarray, torch.Tensor)):
                    return obj.tolist() if hasattr(obj, "tolist") else str(obj)
                elif (
                    isinstance(obj, (list, tuple))
                    and len(obj) > 0
                    and isinstance(obj[0], (np.ndarray, torch.Tensor))
                ):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                else:
                    return obj

            serializable_params = make_serializable(edit_params)
            json.dump(serializable_params, f, ensure_ascii=False, indent=2)

        logger.info(f"编辑参数已保存到 {params_file}")

        # # 重置渲染缓存（如果有）
        # if hasattr(trainer, "reset_renderer_cache"):
        #     trainer.reset_renderer_cache()

        # 编辑后再次打印节点信息（用于比较）
        logger.info("\n编辑后的节点信息:")
        nodes_info_after = print_node_info(trainer)

        # 渲染完整视频
        logger.info("开始渲染...")
        translation_str = "_".join([f"{abs(x):.1f}" for x in args.translation_offset])
        post_fix = f"_early_{args.instance_id}_{args.early_start_frame}_{translation_str}"
        batch_render_with_eval(
            cfg=cfg,
            trainer=trainer,
            dataset=dataset,
            output_dir=output_dir,
            log_dir=log_dir,
            post_fix=post_fix,
        )

        logger.info(f"渲染完成，结果已保存到: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("行人提前出现编辑工具")

    # 基本参数
    parser.add_argument(
        "--resume_from",
        type=str,
        required=True,
        help="检查点路径，如 output/dataset=waymo/1cams/checkpoint_final.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，默认为原检查点目录的子目录",
    )

    # 编辑操作参数
    parser.add_argument(
        "--instance_id", type=int, required=True, help="要编辑的行人实例ID"
    )
    parser.add_argument(
        "--translation_offset",
        type=float,
        nargs=3,
        default=[0.0, 20.0, 0.0],
        help="位置偏移量向量 [x, y, z]，默认为 [0.0, 20.0, 0.0]",
    )
    parser.add_argument(
        "--early_start_frame",
        type=int,
        default=50,
        help="行人提前出现的帧索引",
    )

    # 其他参数
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="配置覆盖选项，如 dataset.num_val=1"
    )

    args = parser.parse_args()
    main(args)