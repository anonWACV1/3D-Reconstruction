#!/usr/bin/env python3
"""
刚体节点变道编辑工具，实现平滑的变道操作。

使用示例:
export PYTHONPATH=$(pwd)
python scene_editing/lane_change.py \
    --resume_from output/waymo_3cams/dataset=waymo/3cams_788/checkpoint_final.pth \
    --instance_id 0 \
    --lane_offset -3.2 \
    --offset_vector 0.0 1.0 0.0 \
    --start_frame 20 \
    --end_frame 50 \
    --output_dir output/lane_change_output
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

from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_multiply

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


def apply_lane_change(trainer, args):
    """
    应用变道操作到指定的刚体实例，包括位置平移和航向角变化

    Args:
        trainer: 训练器实例
        args: 命令行参数，包含变道相关参数

    Returns:
        tuple: (更新后的RigidNodes对象, 编辑参数字典)
    """
    # 检查是否存在RigidNodes模型
    if "RigidNodes" not in trainer.models:
        raise ValueError("模型中没有找到RigidNodes")

    # 获取RigidNodes模型
    rigid_nodes = trainer.models["RigidNodes"]
    logger.info(f"找到RigidNodes模型，实例数量: {rigid_nodes.num_instances}")

    # 设置最大航向角变化为0.5弧度(约28.6度)
    max_yaw_angle = 0.25

    # 保存编辑参数
    edit_params = {
        "operation": "lane_change",
        "instance_id": args.instance_id,
        "lane_offset": args.lane_offset,
        "offset_vector": args.offset_vector,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "max_yaw_angle": max_yaw_angle,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }

    try:
        # 检查实例ID是否有效
        all_ids = rigid_nodes.point_ids[..., 0].unique().cpu().numpy()
        logger.info(f"当前所有实例ID: {all_ids}")

        if args.instance_id not in all_ids:
            logger.warning(f"实例ID {args.instance_id} 不存在")
            return rigid_nodes, edit_params

        # 获取设备信息
        device = trainer.device

        # 检查旋转属性
        # 首先检查 instances_quats
        has_rotation = hasattr(rigid_nodes, "instances_quats")
        logger.info(f"刚体节点是否有 instances_quats 属性: {has_rotation}")
        
        if has_rotation:
            logger.info(f"旋转属性形状: {rigid_nodes.instances_quats.shape}")
        else:
            # 如果没有 instances_quats，检查 instances_rot
            has_rotation = hasattr(rigid_nodes, "instances_rot")
            logger.info(f"刚体节点是否有 instances_rot 属性: {has_rotation}")
            
            if has_rotation:
                logger.info(f"旋转属性形状: {rigid_nodes.instances_rot.shape}")
            else:
                logger.warning("刚体节点没有旋转属性，将只应用位置变化")

        # 将偏移向量转换为张量并归一化
        offset_vector = torch.tensor(
            args.offset_vector, device=device, dtype=torch.float32
        )
        if torch.norm(offset_vector) > 0:
            offset_vector = offset_vector / torch.norm(offset_vector)
        else:
            logger.warning("偏移向量模长为0，使用默认偏移向量 [0, 1, 0]")
            offset_vector = torch.tensor(
                [0.0, 1.0, 0.0], device=device, dtype=torch.float32
            )

        # 计算完整偏移量
        full_offset = offset_vector * args.lane_offset
        logger.info(f"变道偏移向量: {full_offset.cpu().numpy()}")

        # 确保帧范围有效
        num_frames = (
            rigid_nodes.num_frames
            if hasattr(rigid_nodes, "num_frames")
            else rigid_nodes.instances_trans.shape[0]
        )
        start_frame = max(0, min(args.start_frame, num_frames - 1))
        end_frame = max(start_frame + 1, min(args.end_frame, num_frames - 1))

        logger.info(
            f"应用变道: 从第 {start_frame} 帧开始，到第 {end_frame} 帧结束，共 {num_frames} 帧"
        )

        # 应用变道偏移和航向角变化
        with torch.no_grad():
            # 记录原始变换
            original_trans = rigid_nodes.instances_trans[
                start_frame, args.instance_id
            ].clone()
            original_rot = None
            
            # 根据属性名选择正确的旋转属性
            if hasattr(rigid_nodes, "instances_quats"):
                original_rot = rigid_nodes.instances_quats[start_frame, args.instance_id].clone()
                logger.info(f"原始旋转(四元数): {original_rot.cpu().numpy()}")
            elif hasattr(rigid_nodes, "instances_rot"):
                original_rot = rigid_nodes.instances_rot[start_frame, args.instance_id].clone()
                logger.info(f"原始旋转(四元数): {original_rot.cpu().numpy()}")

            # 对每一帧应用平滑过渡的偏移
            for frame_idx in range(num_frames):
                if frame_idx < start_frame:
                    # 起始帧之前保持原始位置和旋转
                    continue
                elif frame_idx > end_frame:
                    # 结束帧之后应用完整偏移，但不改变航向角
                    rigid_nodes.instances_trans[
                        frame_idx, args.instance_id
                    ] += full_offset
                else:
                    # 在起始帧到结束帧之间平滑过渡
                    # 计算进度
                    progress = (frame_idx - start_frame) / (end_frame - start_frame)
                    
                    # 应用余弦平滑因子(0->1)，使变道更加自然
                    smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
                    current_offset = full_offset * smooth_factor
                    
                    # 应用位置偏移
                    rigid_nodes.instances_trans[
                        frame_idx, args.instance_id
                    ] += current_offset
                    
                    # 计算航向角变化
                    # 使用正弦函数模拟转向过程：开始时转向目标车道，结束时回正
                    yaw_angle = math.sin(math.pi * progress) * max_yaw_angle
                    
                    # 确定转向方向：根据车道偏移的正负决定
                    if args.lane_offset > 0:
                        yaw_angle = -yaw_angle  # 向左变道，右侧为正方向
                    
                    # 根据模型中实际的旋转属性名称应用旋转
                    if hasattr(rigid_nodes, "instances_quats"):
                        # 详细记录当前帧的旋转变化
                        logger.info(f"Frame {frame_idx}: 计算航向角 = {yaw_angle} rad ({yaw_angle * 180 / math.pi} deg)")
                        
                        # 创建表示航向角变化的四元数
                        rotation_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
                        rotation_quat = axis_angle_to_quaternion(rotation_axis * yaw_angle)
                        
                        # 记录旋转前四元数
                        before_rot = rigid_nodes.instances_quats[frame_idx, args.instance_id].clone()
                        logger.info(f"Frame {frame_idx} 旋转前: {before_rot.cpu().numpy()}")
                        logger.info(f"应用旋转四元数: {rotation_quat.cpu().numpy()}")
                        
                        # 尝试应用旋转
                        try:
                            # 使用PyTorch3D的四元数乘法
                            new_rot = quaternion_multiply(
                                rotation_quat,
                                rigid_nodes.instances_quats[frame_idx, args.instance_id]
                            )
                            
                            # 应用新旋转
                            rigid_nodes.instances_quats[frame_idx, args.instance_id] = new_rot
                            
                            # 记录旋转后四元数
                            after_rot = rigid_nodes.instances_quats[frame_idx, args.instance_id].clone()
                            logger.info(f"Frame {frame_idx} 旋转后: {after_rot.cpu().numpy()}")
                            logger.info(f"旋转变化: {(after_rot - before_rot).cpu().numpy()}")
                            
                        except Exception as e:
                            logger.error(f"四元数乘法失败: {str(e)}")
                            
                            # 备用方法：直接混合四元数
                            w1, x1, y1, z1 = before_rot.unbind(-1)
                            w2, x2, y2, z2 = rotation_quat.unbind(-1)
                            
                            # 平滑混合
                            blend_factor = 0.2 * smooth_factor
                            w = w1 * (1 - blend_factor) + w2 * blend_factor
                            x = x1 * (1 - blend_factor) + x2 * blend_factor
                            y = y1 * (1 - blend_factor) + y2 * blend_factor
                            z = z1 * (1 - blend_factor) + z2 * blend_factor
                            
                            # 归一化
                            norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
                            w = w / norm
                            x = x / norm
                            y = y / norm
                            z = z / norm
                            
                            # 合并回四元数
                            rigid_nodes.instances_quats[frame_idx, args.instance_id] = torch.stack([w, x, y, z], dim=-1)
                            
                            # 记录
                            after_rot = rigid_nodes.instances_quats[frame_idx, args.instance_id].clone()
                            logger.info(f"Frame {frame_idx} 旋转后(方法2): {after_rot.cpu().numpy()}")
                    
                    elif hasattr(rigid_nodes, "instances_rot"):
                        # 与上面类似的处理，但使用 instances_rot 属性
                        logger.info(f"Frame {frame_idx}: 计算航向角 = {yaw_angle} rad ({yaw_angle * 180 / math.pi} deg)")
                        
                        rotation_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
                        rotation_quat = axis_angle_to_quaternion(rotation_axis * yaw_angle)
                        
                        before_rot = rigid_nodes.instances_rot[frame_idx, args.instance_id].clone()
                        logger.info(f"Frame {frame_idx} 旋转前: {before_rot.cpu().numpy()}")
                        logger.info(f"应用旋转四元数: {rotation_quat.cpu().numpy()}")
                        
                        try:
                            new_rot = quaternion_multiply(
                                rotation_quat,
                                rigid_nodes.instances_rot[frame_idx, args.instance_id]
                            )
                            
                            rigid_nodes.instances_rot[frame_idx, args.instance_id] = new_rot
                            
                            after_rot = rigid_nodes.instances_rot[frame_idx, args.instance_id].clone()
                            logger.info(f"Frame {frame_idx} 旋转后: {after_rot.cpu().numpy()}")
                            logger.info(f"旋转变化: {(after_rot - before_rot).cpu().numpy()}")
                            
                        except Exception as e:
                            logger.error(f"四元数乘法失败: {str(e)}")
                            
                            # 备用方法
                            w1, x1, y1, z1 = before_rot.unbind(-1)
                            w2, x2, y2, z2 = rotation_quat.unbind(-1)
                            
                            blend_factor = 0.2 * smooth_factor
                            w = w1 * (1 - blend_factor) + w2 * blend_factor
                            x = x1 * (1 - blend_factor) + x2 * blend_factor
                            y = y1 * (1 - blend_factor) + y2 * blend_factor
                            z = z1 * (1 - blend_factor) + z2 * blend_factor
                            
                            norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
                            w = w / norm
                            x = x / norm
                            y = y / norm
                            z = z / norm
                            
                            rigid_nodes.instances_rot[frame_idx, args.instance_id] = torch.stack([w, x, y, z], dim=-1)
                            
                            after_rot = rigid_nodes.instances_rot[frame_idx, args.instance_id].clone()
                            logger.info(f"Frame {frame_idx} 旋转后(方法2): {after_rot.cpu().numpy()}")

            # 记录修改后的变换以便比较
            modified_trans = rigid_nodes.instances_trans[
                end_frame, args.instance_id
            ].clone()
            modified_rot = None
            
            logger.info(f"变道前位置: {original_trans.cpu().numpy()}")
            logger.info(f"变道后位置: {modified_trans.cpu().numpy()}")
            logger.info(f"总位置变化量: {(modified_trans - original_trans).cpu().numpy()}")
            
            # 记录旋转变化
            if hasattr(rigid_nodes, "instances_quats") and original_rot is not None:
                modified_rot = rigid_nodes.instances_quats[end_frame, args.instance_id].clone()
                logger.info(f"变道前旋转(四元数): {original_rot.cpu().numpy()}")
                logger.info(f"变道后旋转(四元数): {modified_rot.cpu().numpy()}")
                logger.info(f"旋转变化: {(modified_rot - original_rot).cpu().numpy()}")
            elif hasattr(rigid_nodes, "instances_rot") and original_rot is not None:
                modified_rot = rigid_nodes.instances_rot[end_frame, args.instance_id].clone()
                logger.info(f"变道前旋转(四元数): {original_rot.cpu().numpy()}")
                logger.info(f"变道后旋转(四元数): {modified_rot.cpu().numpy()}")
                logger.info(f"旋转变化: {(modified_rot - original_rot).cpu().numpy()}")

        logger.info(f"实例 {args.instance_id} 的变道操作已完成")

    except Exception as e:
        logger.error(f"变道操作过程中出错: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())

    # 返回更新后的模型和编辑参数
    return rigid_nodes, edit_params


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
        output_dir = os.path.join(log_dir, f"lane_change_output_{args.instance_id}")

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

    # 执行变道操作
    with torch.no_grad():
        rigid_nodes, edit_params = apply_lane_change(trainer, args)

        # 保存编辑参数
        params_file = os.path.join(output_dir, "lane_change_params.json")
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

        logger.info(f"变道参数已保存到 {params_file}")

        # # 重置渲染缓存（如果有）
        # if hasattr(trainer, "reset_renderer_cache"):
        #     trainer.reset_renderer_cache()

        # 编辑后再次打印节点信息（用于比较）
        logger.info("\n变道后的节点信息:")
        nodes_info_after = print_node_info(trainer)

        # 渲染完整视频
        logger.info("开始渲染...")
        post_fix = f"_lane_{args.instance_id}_{abs(args.lane_offset):.1f}"
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
    parser = argparse.ArgumentParser("刚体节点变道编辑工具")

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

    # 变道操作参数
    parser.add_argument(
        "--instance_id", type=int, required=True, help="要进行变道的实例ID"
    )
    parser.add_argument(
        "--lane_offset",
        type=float,
        default=-3.2,
        help="变道的总位移量（米），负值表示向左变道，正值表示向右变道",
    )
    parser.add_argument(
        "--offset_vector",
        type=float,
        nargs=3,
        default=[0.0, 1.0, 0.0],
        help="变道的方向向量，默认为Y轴方向 [0.0, 1.0, 0.0]，会被自动归一化",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=20,
        help="开始变道的帧索引",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=50,
        help="结束变道的帧索引",
    )

    # 其他参数
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="配置覆盖选项，如 dataset.num_val=1"
    )

    args = parser.parse_args()
    main(args)