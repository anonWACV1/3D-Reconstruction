#!/usr/bin/env python3
"""
批量修改多个SMPL实例并渲染到单一视频的脚本。

使用示例:
export PYTHONPATH=$(pwd)
python scene_editing/multi_instance_replace.py \
    --resume_from output/waymo_1cam_edit/dataset=waymo/1cams/checkpoint_final.pth \
    --config_file ./configs/scene_config.yaml \
    --output_dir ./multi_replace_output
"""

import os
import logging
import argparse
import torch
import json
import yaml
import glob
import traceback
from pathlib import Path
from omegaconf import OmegaConf

# 导入现有的替换函数
try:
    from scene_editing.scene_editing import (
        replace_smpl_instance_improved,
        batch_render_with_eval,
        load_pose_sequence,
        replace_smpl_pose
    )
except ImportError:
    # 如果上面的导入失败，尝试直接导入
    try:
        from replace_combined import (
            convert_gart_to_smpl,
            replace_smpl_instance_improved,
            batch_render_with_eval,
            load_pose_sequence,
            replace_smpl_pose
        )
    except ImportError:
        # 最后尝试从当前目录导入
        from scene_editing.replace_combined import (
            convert_gart_to_smpl,
            batch_render_with_eval,
            load_pose_sequence,
            replace_smpl_pose
        )

from utils.misc import import_str
from datasets.driving_dataset import DrivingDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_file):
    """加载配置文件，支持JSON和YAML格式"""
    if not os.path.exists(config_file):
        logger.error(f"配置文件不存在: {config_file}")
        return None
        
    file_ext = os.path.splitext(config_file)[1].lower()
    try:
        if file_ext in ['.json']:
            with open(config_file, 'r') as f:
                config = json.load(f)
        elif file_ext in ['.yaml', '.yml']:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                # 打印加载的配置，用于调试
                logger.info(f"加载的YAML配置结构: {type(config)}")
                if isinstance(config, dict):
                    for key, value in config.items():
                        logger.info(f"配置键: {key}, 类型: {type(value)}")
                        if key == "instances" and isinstance(value, list):
                            logger.info(f"实例列表长度: {len(value)}")
                            for i, instance in enumerate(value):
                                logger.info(f"实例 {i} 类型: {type(instance)}")
        else:
            logger.error(f"不支持的配置文件格式: {file_ext}")
            return None
            
        logger.info(f"已加载配置文件: {config_file}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def convert_gart_to_smpl(solver, num_frames):
    """按照viz_human_all数据格式提取参数，并修正参数范围和形状匹配"""
    model = solver.load_saved_model(os.path.join(solver.log_dir, "model.pth"))
    model.eval()
    
    # 使用与viz_human_all类似的方式准备姿态数据
    tpose = torch.zeros((1, 24, 3), dtype=torch.float32, device=solver.device)
    if hasattr(solver, 'viz_base_R'):
        from pytorch3d.transforms import matrix_to_axis_angle
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
    from torch.nn import functional as F
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
    
    logger.info(f"GART点云处理完成: 点数={pts.shape[0]}, 特征DC形状={features_dc.shape}")
    return instance_data

def process_instance(
    trainer, 
    instance_id, 
    instance_config,
    num_frames
):
    """处理单个实例的替换"""
    logger.info(f"处理实例 {instance_id}")
    
    # 确保instance_config是字典类型
    if not isinstance(instance_config, dict):
        logger.error(f"实例配置必须是字典类型，当前类型: {type(instance_config)}")
        return False
    
    # 提取配置项
    gart_path = instance_config.get("gart_model_dir", None)
    motion_path = instance_config.get("new_pose_path", None)
    keep_translation = instance_config.get("keep_translation", True)
    keep_global_rot = instance_config.get("keep_global_rot", True)
    
    if not gart_path and not motion_path:
        logger.warning(f"实例 {instance_id} 未指定GART模型或姿势路径，跳过")
        return False
    
    logger.info(f"GART路径: {gart_path if gart_path else '未指定'}")
    logger.info(f"姿势路径: {motion_path if motion_path else '未指定'}")
    
    # 处理GART替换
    try:
        with torch.no_grad():
            if gart_path and os.path.exists(gart_path):
                logger.info(f"正在加载GART模型: {gart_path}")
                
                # 导入GART模块
                try:
                    from third_party.GART.solver import TGFitter
                except ImportError:
                    logger.error("无法导入GART模块，请确保已安装third_party.GART")
                    return False
                
                # 初始化GART求解器
                gart_solver = TGFitter(
                    log_dir=gart_path,
                    profile_fn=os.path.join(gart_path, "gen.yaml"),
                    mode="human",
                    device="cuda",
                    NO_TB=True
                )
                
                # 加载模型
                try:
                    model = gart_solver.load_saved_model(os.path.join(gart_path, "model.pth"))
                    logger.info(f"GART模型加载成功")
                except Exception as e:
                    logger.error(f"GART模型加载失败: {str(e)}")
                    return False
                
                # 转换GART数据到SMPL格式
                logger.info(f"正在提取{num_frames}帧的GART参数")
                gart_instance = convert_gart_to_smpl(gart_solver, num_frames)
                
                # 执行GART点云替换
                logger.info(f"正在替换实例 {instance_id} 的点云数据")
                replace_smpl_instance_improved(
                    trainer=trainer,
                    instance_id=instance_id,
                    new_instance=gart_instance,
                    keep_translation=keep_translation,
                    keep_global_rot=keep_global_rot
                )
            
            # 处理姿势替换
            if motion_path and os.path.exists(motion_path):
                logger.info(f"正在加载姿势数据: {motion_path}")
                
                # 加载姿势数据
                new_poses = load_pose_sequence(motion_path)
                
                # 替换姿势
                logger.info(f"正在替换实例 {instance_id} 的姿势数据")
                replace_smpl_pose(
                    trainer=trainer,
                    instance_id=instance_id,
                    new_poses=new_poses,
                    keep_translation=keep_translation,
                    keep_global_rot=keep_global_rot
                )
                
            return True
            
    except Exception as e:
        logger.error(f"替换实例 {instance_id} 时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main(args):
    # 加载场景配置文件
    scene_config = load_config(args.config_file)
    if not scene_config:
        return
    
    # 提取实例配置
    instances = scene_config.get("instances", [])
    if not instances:
        logger.error("配置文件中未找到实例配置")
        return
    
    # 检查instances是否为列表
    if not isinstance(instances, list):
        logger.error(f"实例配置必须是列表类型，当前类型: {type(instances)}")
        return
    
    # 加载模型配置
    config_dir = os.path.dirname(args.resume_from)
    config_path = os.path.join(config_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"模型配置文件不存在: {config_path}")
        return
        
    cfg = OmegaConf.load(config_path)
    if args.opts:
        cfg.merge_with(OmegaConf.from_dotlist(args.opts))
    
    # 设置输出目录
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(args.resume_from), "multi_instance_output")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 初始化数据集和训练器
    logger.info("初始化数据集和训练器...")
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
    
    # 获取帧数
    if "SMPLNodes" not in trainer.models:
        logger.error("模型中没有SMPLNodes组件")
        return
        
    smpl_nodes = trainer.models["SMPLNodes"]
    num_frames = smpl_nodes.instances_quats.shape[0]
    logger.info(f"场景帧数: {num_frames}")
    
    # 处理每个实例
    success_count = 0
    for instance_config in instances:
        # 确保instance_config是字典类型
        if not isinstance(instance_config, dict):
            logger.error(f"实例配置必须是字典类型，当前类型: {type(instance_config)}")
            continue
            
        instance_id = instance_config.get("instance_id")
        if instance_id is None:
            logger.warning("实例配置中未指定instance_id，跳过")
            continue
            
        success = process_instance(
            trainer=trainer,
            instance_id=instance_id,
            instance_config=instance_config,
            num_frames=num_frames
        )
        
        if success:
            success_count += 1
    
    # 所有实例处理完毕后，执行一次渲染
    if success_count > 0:
        logger.info(f"成功处理了 {success_count}/{len(instances)} 个实例，开始渲染...")
        output_postfix = scene_config.get("output_postfix", "_multi")
        
        try:
            log_dir = os.path.dirname(args.resume_from)
            batch_render_with_eval(
                cfg=cfg, 
                trainer=trainer, 
                dataset=dataset, 
                output_dir=output_dir, 
                log_dir=log_dir,
                post_fix=output_postfix
            )
            logger.info(f"渲染完成，结果保存到: {output_dir}")
        except Exception as e:
            logger.error(f"渲染过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        logger.warning("没有成功处理任何实例，跳过渲染")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("修改多个SMPL实例并渲染到单一视频")
    
    # 基本参数
    parser.add_argument(
        "--resume_from", type=str, required=True, 
        help="检查点路径，如 output/waymo_1cam_edit/dataset=waymo/1cams/checkpoint_final.pth"
    )
    parser.add_argument(
        "--config_file", type=str, required=True,
        help="场景配置文件路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="输出目录路径"
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, 
        help="配置覆盖选项"
    )
    
    args = parser.parse_args()
    main(args)