#!/usr/bin/env python3
"""
用于同时替换GART点云和SMPL姿势的组合脚本。
此脚本结合了replace_gart.py和replace_smpl_pose.py的功能。

使用示例:
export PYTHONPATH=$(pwd)
python scene_editing/replace_combined.py \
    --resume_from output/waymo_1cam_edit/dataset=waymo/1cams/checkpoint_final.pth \
    --gart_model_dir $(pwd)/GART_DATA/skywalker/ \
    --new_pose_path $(pwd)/motion/20_out.npy \
    --instance_id 0 \   
    --keep_translation \
    --keep_global_rot
"""

import os
import logging
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from torch.nn import functional as F

from utils.misc import import_str
from datasets.driving_dataset import DrivingDataset
from models.trainers import BasicTrainer

# 导入函数可能需要根据您的实际包结构进行调整
try:
    from scene_editing.scene_editing import (
        replace_smpl_instance_improved,
        batch_render_with_eval,
        load_pose_sequence,
        replace_smpl_pose
    )
except ImportError:
    # 如果上面的导入失败，尝试直接导入
    from scene_editing import (
        replace_smpl_instance_improved,
        batch_render_with_eval,
        load_pose_sequence,
        replace_smpl_pose
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
    # 初始化配置
    config_dir = os.path.dirname(args.resume_from)
    config_path = os.path.join(config_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.merge_with(OmegaConf.from_dotlist(args.opts))

    # 设置输出目录
    log_dir = os.path.dirname(args.resume_from)
    output_dir = os.path.join(log_dir, "combined_replace_output")
    os.makedirs(output_dir, exist_ok=True)
    
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
    trainer.resume_from_checkpoint(args.resume_from, load_only_model=True)
    trainer.eval()
    
    # 检查并执行GART点云替换
    if args.gart_model_dir:
        logger.info(f"加载GART模型: {args.gart_model_dir}")
        
        # 1. 加载GART模型
        from third_party.GART.solver import TGFitter
        gart_solver = TGFitter(
            log_dir=args.gart_model_dir,
            profile_fn=os.path.join(args.gart_model_dir, "gen.yaml"),
            mode="human",
            device="cuda",
            NO_TB=True
        )
        model = gart_solver.load_saved_model(os.path.join(args.gart_model_dir, "model.pth"))
        
        # 2. 提取GART参数并准备替换数据
        smpl_nodes = trainer.models["SMPLNodes"]
        num_frames = smpl_nodes.instances_quats.shape[0]
        logger.info(f"为{num_frames}帧提取GART参数")
        gart_instance = convert_gart_to_smpl(gart_solver, num_frames)
        
        # 3. 执行GART点云替换
        logger.info(f"替换实例 {args.instance_id} 的点云数据")
        with torch.no_grad():
            replace_smpl_instance_improved(
                trainer=trainer,
                instance_id=args.instance_id,
                new_instance=gart_instance,
                keep_translation=args.keep_translation,
                keep_global_rot=args.keep_global_rot
            )
    
    # 检查并执行姿势替换
    if args.new_pose_path:
        logger.info(f"加载姿势数据: {args.new_pose_path}")
        
        # 1. 加载姿势数据
        new_poses = load_pose_sequence(args.new_pose_path)
        
        # 2. 替换姿势
        logger.info(f"替换实例 {args.instance_id} 的姿势数据")
        with torch.no_grad():
            replace_smpl_pose(
                trainer=trainer,
                instance_id=args.instance_id,
                new_poses=new_poses,
                keep_translation=args.keep_translation,
                keep_global_rot=args.keep_global_rot
            )
    
    # 批量渲染
    logger.info(f"正在渲染到: {output_dir}")
    batch_render_with_eval(
        cfg=cfg, 
        trainer=trainer, 
        dataset=dataset, 
        output_dir=output_dir, 
        log_dir=log_dir,
        post_fix="_combined"
    )
    logger.info(f"渲染完成，结果保存到: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("替换SMPL实例点云和姿势的组合工具")
    
    # 基本参数
    parser.add_argument(
        "--resume_from", type=str, required=True, 
        help="检查点路径，如 output/Kitti/dataset=Kitti/1cams/checkpoint_final.pth"
    )
    parser.add_argument(
        "--instance_id", type=int, default=1,
        help="要替换的SMPL实例ID"
    )
    
    # GART点云相关参数
    parser.add_argument(
        "--gart_model_dir", type=str, default="",
        help="GART模型目录，如果提供则替换点云"
    )
    
    # 姿势替换相关参数
    parser.add_argument(
        "--new_pose_path", type=str, default="",
        help="新姿势数据的路径(.npy文件)，如果提供则替换姿势"
    )
    
    # 通用选项
    parser.add_argument(
        "--keep_translation", action="store_true",
        help="保留原始平移数据"
    )
    parser.add_argument(
        "--keep_global_rot", action="store_true",
        help="保留原始全局旋转"
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, 
        help="配置覆盖选项"
    )
    
    args = parser.parse_args()
    
    # 检查至少需要一个替换选项
    if not args.gart_model_dir and not args.new_pose_path:
        parser.error("至少需要提供--gart_model_dir或--new_pose_path中的一个")
    
    main(args)