from typing import List, Dict, Optional, Tuple, Any
from omegaconf import OmegaConf
import os
import time
import logging
import argparse
import torch
from torch.nn import Parameter
import numpy as np
from tqdm import tqdm
import trimesh
import math
import json
import cv2

# 特定的3D转换库
from pytorch3d.transforms import (
    matrix_to_quaternion, 
    quaternion_to_matrix,
    axis_angle_to_matrix, 
    axis_angle_to_quaternion
)

# 项目特定导入
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import render_images, save_videos
from pathlib import Path
from utils.simplify_loc2rot import joints2smpl
from utils.rotation_conversions import quaternion_multiply
from third_party.smplx.transfer_model.transfer_model import run_fitting
from utils.rotation_conversions import quaternion_to_euler_angles

from third_party.HumanGaussian.animation import Skeleton
# 在实际使用时，应根据项目结构调整此导入
from models.human_body import (
    phalp_colors,
    SMPLTemplate,
    get_on_mesh_init_geo_values,
    batch_rigid_transform
)

logger = logging.getLogger()

def improved_RGB2SH(rgb):
    """将RGB颜色转换为球谐系数，更强健的实现"""
    # 添加调试信息
    if hasattr(rgb, 'shape'):
        logger.debug(f"RGB2SH输入形状: {rgb.shape}")
    
    # 确保输入是正确的形状 [N, 3]
    if not isinstance(rgb, torch.Tensor):
        rgb = torch.tensor(rgb, dtype=torch.float32)
    
    # 强制转换为二维张量
    if len(rgb.shape) == 1:
        rgb = rgb.unsqueeze(0)  # 添加一个维度
    
    # 如果只有一个通道，扩展到三个通道
    if rgb.shape[-1] == 1:
        rgb = rgb.repeat(1, 3)
    
    # 确保值范围在[0,1]之间
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    
    # 转换为球谐系数 (简单转换)
    result = (rgb - 0.5) / 0.28209479177387814
    
    logger.debug(f"RGB2SH输出形状: {result.shape}")
    return result

def match_sequence_length(target_length: int, sequence: torch.Tensor) -> torch.Tensor:
    """调整序列长度以匹配目标长度"""
    logger.debug(f"输入序列形状: {sequence.shape}, 目标长度: {target_length}")
    current_length = sequence.shape[0]
    
    if current_length == target_length:
        logger.debug("序列长度已匹配，无需调整")
        return sequence
    
    if current_length < target_length:
        # 计算需要重复的次数
        repeat_times = (target_length // current_length) + 1
        logger.debug(f"序列过短，将重复 {repeat_times} 次")
        
        # 根据输入维度动态调整repeat参数
        repeat_dims = [repeat_times] + [1]*(sequence.dim()-1)
        logger.debug(f"使用repeat参数: {repeat_dims}")
        
        repeated = sequence.repeat(*repeat_dims)
        return repeated[:target_length]
    else:
        logger.debug(f"序列过长，将截断至 {target_length} 帧")
        return sequence[:target_length]

def inspect_smpl_node_structure(smpl_nodes):
    """检查SMPL节点的结构，打印所有属性"""
    logger.info("SMPL节点结构检查:")
    for attr_name in dir(smpl_nodes):
        if not attr_name.startswith('_') or attr_name in ['_means', '_features_dc', '_scales', '_rotations', '_opacity', '_features_rest']:
            if hasattr(smpl_nodes, attr_name):
                attr = getattr(smpl_nodes, attr_name)
                if isinstance(attr, torch.Tensor):
                    logger.info(f"  {attr_name}: Tensor, 形状 = {attr.shape}, 类型 = {attr.dtype}")
                elif attr is not None:
                    logger.info(f"  {attr_name}: {type(attr)}")


def compare_instance_info(smpl_nodes, instance_id, new_instance, print_stats=True):
    """比较SMPL节点中原有实例和新实例的详细信息
    
    Args:
        smpl_nodes: SMPL节点对象
        instance_id: 要替换的实例ID
        new_instance: 新实例的数据字典
        print_stats: 是否打印统计信息
    """
    logger.info(f"===== 实例 {instance_id} 替换前后对比 =====")
    
    # 获取原始实例信息
    points_per_instance = smpl_nodes.smpl_points_num
    start_idx = instance_id * points_per_instance
    end_idx = (instance_id + 1) * points_per_instance
    
    # 创建比较表格
    info_table = []
    
    # 检查几何属性
    geometric_attrs = {
        "_means/pts": (smpl_nodes._means[start_idx:end_idx], new_instance.get("pts")),
        "_scales/scales": (smpl_nodes._scales[start_idx:end_idx], new_instance.get("scales")),
        "_quats/rotations": (smpl_nodes._quats[start_idx:end_idx], new_instance.get("rotations")),
        "_opacity/opacities": (smpl_nodes._opacities[start_idx:end_idx], new_instance.get("opacities")),
        "_features_dc": (smpl_nodes._features_dc[start_idx:end_idx], new_instance.get("features_dc")),
        "_features_rest": (smpl_nodes._features_rest[start_idx:end_idx], new_instance.get("features_rest")),
    }
    
    for attr_name, (orig_val, new_val) in geometric_attrs.items():
        if orig_val is not None and new_val is not None:
            orig_name, new_name = attr_name.split("/") if "/" in attr_name else (attr_name, attr_name)
            
            # 检查形状
            orig_shape = orig_val.shape
            new_shape = new_val.shape
            shape_match = "✓" if orig_shape == new_shape else "✗"
            
            # 计算值范围
            orig_min, orig_max = orig_val.min().item(), orig_val.max().item()
            new_min, new_max = new_val.min().item(), new_val.max().item()
            
            # 计算均值和标准差
            orig_mean, orig_std = orig_val.mean().item(), orig_val.std().item()
            new_mean, new_std = new_val.mean().item(), new_val.std().item()
            
            info_table.append({
                "属性": f"{orig_name} → {new_name}",
                "原形状": str(orig_shape),
                "新形状": str(new_shape),
                "形状匹配": shape_match,
                "原范围": f"{orig_min:.4f} 到 {orig_max:.4f}",
                "新范围": f"{new_min:.4f} 到 {new_max:.4f}",
                "原均值/标准差": f"{orig_mean:.4f} / {orig_std:.4f}",
                "新均值/标准差": f"{new_mean:.4f} / {new_std:.4f}",
            })
    
    # 检查姿态和变换
    pose_attrs = {
        "global_orient": (smpl_nodes.instances_quats[0, instance_id], new_instance.get("global_orient")),
        "body_pose": (smpl_nodes.smpl_qauts[0, instance_id], new_instance.get("body_pose")),
        "translation": (smpl_nodes.instances_trans[0, instance_id], new_instance.get("transl")),
    }
    
    for attr_name, (orig_val, new_val) in pose_attrs.items():
        if orig_val is not None and new_val is not None:
            # 对形状进行调整以便比较
            if attr_name == "global_orient":
                if new_val.dim() > 2:
                    new_val = new_val[0, 0]  # 取第一帧的全局旋转
                orig_val = orig_val.squeeze()
            elif attr_name == "body_pose":
                if new_val.dim() > 2:
                    new_val = new_val[0]  # 取第一帧的姿态
            elif attr_name == "translation":
                if new_val.dim() > 1:
                    new_val = new_val[0]  # 取第一帧的平移
            
            # 检查形状
            orig_shape = orig_val.shape
            new_shape = new_val.shape
            shape_match = "✓" if orig_shape == new_shape else "✗"
            
            # 值范围分析
            orig_min, orig_max = orig_val.min().item(), orig_val.max().item()
            new_min, new_max = new_val.min().item(), new_val.max().item()
            
            info_table.append({
                "属性": attr_name,
                "原形状": str(orig_shape),
                "新形状": str(new_shape),
                "形状匹配": shape_match,
                "原范围": f"{orig_min:.4f} 到 {orig_max:.4f}",
                "新范围": f"{new_min:.4f} 到 {new_max:.4f}",
                "原内容": str(orig_val.flatten()[:4].tolist()) + "...",  # 只显示前几个值
                "新内容": str(new_val.flatten()[:4].tolist()) + "...",
            })
    
    # 检查映射关系
    if "mapping" in new_instance:
        mapping = new_instance["mapping"]
        logger.info(f"新实例映射信息:")
        for map_name, map_val in mapping.items():
            if map_val is not None:
                logger.info(f"  {map_name}: 形状={map_val.shape}, 范围=[{map_val.min().item():.4f}, {map_val.max().item():.4f}]")
            else:
                logger.info(f"  {map_name}: None")
    
    # 打印表格
    if print_stats:
        logger.info("\n" + "-" * 120)
        header = list(info_table[0].keys())
        row_format = "| {:<20} | {:<15} | {:<15} | {:<10} | {:<20} | {:<20} | {:<20} | {:<20} |"
        
        # 打印表头
        logger.info(row_format.format(*header))
        logger.info("-" * 120)
        
        # 打印数据行
        for row in info_table:
            logger.info(row_format.format(*[str(row.get(h, "")) for h in header]))
        
        logger.info("-" * 120)
    
    # 返回比较结果
    return info_table


def load_and_convert_using_skeleton(ply_path: str, motion_path: str, smplx_path: str, frame_count: int) -> Dict:
    """使用Skeleton类加载并转换数据为SMPLNodes格式"""
    logger.info(f"使用Skeleton加载数据: PLY={ply_path}, Motion={motion_path}, SMPLX={smplx_path}")
    try:
        # 创建Skeleton实例
        from argparse import Namespace
        opt = Namespace(
            ply=ply_path,
            motion=motion_path,
            smplx_path=smplx_path,
            gui=False,
            W=800,
            H=800,
            radius=2,
            fovy=50
        )
        
        skel = Skeleton(opt)
        
        # 加载SMPLX模型
        skel.load_smplx(opt.smplx_path)
        logger.info(f"SMPLX模型加载成功，顶点数: {skel.vertices.shape[0] if hasattr(skel, 'vertices') else 'N/A'}")
        
        # 获取点云数据
        gs_xyz = skel.gs.gaussians._xyz.detach().clone()
        gs_features_dc = skel.gs.gaussians._features_dc.detach().clone() 
        gs_features_rest = skel.gs.gaussians._features_rest.detach().clone()
        gs_opacity = skel.gs.gaussians._opacity.detach().clone()
        gs_scaling = skel.gs.gaussians._scaling.detach().clone()
        gs_rotation = skel.gs.gaussians._rotation.detach().clone()
        
        # 获取映射信息
        mapping_dist = torch.tensor(skel.mapping_dist, dtype=torch.float32) if hasattr(skel, 'mapping_dist') else None
        mapping_face = torch.tensor(skel.mapping_face, dtype=torch.int64) if hasattr(skel, 'mapping_face') else None
        mapping_uvw = torch.tensor(skel.mapping_uvw, dtype=torch.float32) if hasattr(skel, 'mapping_uvw') else None
        
        # 1. 动作数据读取 - 参照insert_smpl_copy.py的逻辑
        # 直接从NPZ文件加载SMPLX参数
        import numpy as np
        smplx_data = np.load(motion_path)
        
        # SMPLX到SMPL的关节映射
        smplx_to_smpl_joints = [
            0,   # 骨盆(根关节)
            1,2,3,   # 左腿
            4,5,6,   # 右腿
            7,8,9,   # 脊柱
            10,11,12,13,  # 左臂
            14,15,16,17,  # 右臂
            18,19,20,  # 头部
            21,22,23   # 手部(简化为SMPL)
        ]
        
        # 提取并转换姿态参数
        poses = smplx_data['poses']  # [nframe, 55, 3]
        trans = smplx_data['trans']  # [nframe, 3]
        betas = smplx_data['betas']  # [10]
        
        # 转换为SMPL关节(24个)
        smpl_poses = poses[:, smplx_to_smpl_joints, :]  # [nframe, 24, 3]
        logger.debug(f"转换后poses shape: {smpl_poses.shape}")
        
        # 转换为四元数
        smpl_quats = axis_angle_to_quaternion(torch.from_numpy(smpl_poses))  # [nframe, 24, 4]
        
        # # 2. 匹配帧数
        # smpl_quats = match_sequence_length(frame_count, smpl_quats)
        # trans = match_sequence_length(frame_count, torch.from_numpy(trans).float())
        
        # 3. 准备最终数据结构
        instance_data = {
            "pts": gs_xyz,
            "colors": torch.sigmoid(gs_features_dc),  # 将特征转换为颜色
            "scales": gs_scaling,
            "rotations": gs_rotation,
            "opacities": torch.sigmoid(gs_opacity),
            "features_dc": gs_features_dc,
            "features_rest": gs_features_rest,
            "global_orient": smpl_quats[:, 0],      # [num_frames, 4]
            "body_pose": smpl_quats[:, 1:],        # [num_frames, 23, 4]
            "transl":  torch.from_numpy(trans).float(),                       # [num_frames, 3]
            "betas": torch.from_numpy(betas).float(),  # 形状参数
            "mapping": {
                "dist": mapping_dist,
                "face": mapping_face,
                "uvw": mapping_uvw,
            },
            "size": torch.tensor([1.0, 1.0, 1.0]),
            "frame_info": torch.ones(frame_count, dtype=torch.bool),
            "num_pts": gs_xyz.shape[0]
        }
        
        logger.info(f"从Skeleton成功转换数据，点数: {gs_xyz.shape[0]}, 帧数: {frame_count}")
        return instance_data
        
    except Exception as e:
        logger.error(f"使用Skeleton加载数据失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def replace_smpl_instance_improved(trainer, instance_id, new_instance, keep_translation=True, keep_global_rot=True):
    """使用新实例数据替换指定SMPL实例，完整保留姿势和位移更新逻辑"""
    smpl_nodes = trainer.models["SMPLNodes"]
    compare_instance_info(smpl_nodes, instance_id, new_instance)
    try:
        with torch.no_grad():
            # 1. 基础信息获取
            device = smpl_nodes._means.device
            num_frames = smpl_nodes.instances_quats.shape[0]
            points_per_instance = smpl_nodes.smpl_points_num  # 6890
            start_idx = instance_id * points_per_instance
            end_idx = (instance_id + 1) * points_per_instance
            
            # 2. 准备新实例数据
            input_pts = new_instance["pts"].to(device)
            input_colors = new_instance["colors"].to(device)
            
            # 处理可选参数
            input_scales = new_instance.get("scales", None)
            if input_scales is not None:
                input_scales = input_scales.to(device)
            
            input_rotations = new_instance.get("rotations", None)
            if input_rotations is not None:
                input_rotations = input_rotations.to(device)
            
            input_opacities = new_instance.get("opacities", None)
            if input_opacities is not None:
                input_opacities = input_opacities.to(device)
            
            features_dc = new_instance.get("features_dc", None)
            if features_dc is not None:
                features_dc = features_dc.to(device)
            
            # 3. 点云数量校验与调整
            if len(input_pts) != points_per_instance:
                logger.warning(f"点云数量调整: {len(input_pts)} -> {points_per_instance}")
                if len(input_pts) < points_per_instance:
                    # 不足时重复数据
                    repeat_times = math.ceil(points_per_instance / len(input_pts))
                    input_pts = input_pts.repeat(repeat_times, 1)[:points_per_instance]
                    if input_rotations is not None:
                        input_rotations = input_rotations.repeat(repeat_times, 1)[:points_per_instance]
                    if features_dc is not None:
                        features_dc = features_dc.repeat(repeat_times, *[1]*(features_dc.dim()-1))[:points_per_instance]
                    if input_opacities is not None:
                        input_opacities = input_opacities.repeat(repeat_times, *[1]*(input_opacities.dim()-1))[:points_per_instance]
                else:
                    # 过多时随机采样
                    selected_indices = torch.randperm(len(input_pts))[:points_per_instance]
                    input_pts = input_pts[selected_indices]
                    if input_rotations is not None:
                        input_rotations = input_rotations[selected_indices]
                    if features_dc is not None:
                        features_dc = features_dc[selected_indices]
                    if input_opacities is not None:
                        input_opacities = input_opacities[selected_indices]
            
            # 4. 执行点云替换
            smpl_nodes._means.data[start_idx:end_idx] = input_pts
            
            # 处理各属性的默认值
            if input_scales is None:
                input_scales = torch.ones_like(smpl_nodes._scales[start_idx:end_idx]) * 0.01
            smpl_nodes._scales.data[start_idx:end_idx] = input_scales
            
            if input_rotations is None:
                input_rotations = torch.zeros(points_per_instance, 4, device=device)
                input_rotations[:, 3] = 1.0  # 单位四元数
            smpl_nodes._quats.data[start_idx:end_idx] = input_rotations
            
            # 处理features_dc维度问题
            if features_dc is None:
                features_dc = improved_RGB2SH(input_colors)
            
            # 确保features_dc是2D张量 [6890, 3]
            if features_dc.dim() == 3:
                if features_dc.shape[1] == 1:  # [6890, 1, 3] -> [6890, 3]
                    features_dc = features_dc.squeeze(1)
                else:
                    # 其他情况取第一个切片
                    features_dc = features_dc[:, 0, :]
            
            smpl_nodes._features_dc.data[start_idx:end_idx] = features_dc
            
            if input_opacities is None:
                input_opacities = torch.ones(points_per_instance, 1, device=device) * 5.0
            smpl_nodes._opacities.data[start_idx:end_idx] = input_opacities
            
            # # 更新点ID
            smpl_nodes.point_ids.data[start_idx:end_idx] = torch.full(
                (points_per_instance, 1), instance_id, device=device, dtype=torch.long
            )
            # keep_mask = smpl_nodes.point_ids[..., 0] != instance_id
            # kept_point_ids = smpl_nodes.point_ids[keep_mask]

            # new_point_ids = torch.full((len(input_pts), 1), instance_id, 
            #                       device=input_pts.device, dtype=torch.long)
            # smpl_nodes.point_ids = torch.cat([kept_point_ids, new_point_ids], dim=0)
            
            # 5. 更新姿势和位移参数 (参照insert_smpl_copy.py的逻辑)
            # 匹配全局旋转、关节旋转和位移
            global_orient = match_sequence_length(num_frames, new_instance["global_orient"]) if "global_orient" in new_instance else None
            body_pose = match_sequence_length(num_frames, new_instance["body_pose"]) if "body_pose" in new_instance else None
            trans = match_sequence_length(num_frames, new_instance["transl"])if "transl" in new_instance else None
            
            # # 更新姿态和平移
            # if not keep_global_rot:
            #     for i in range(num_frames):
            #         if i < len(global_orient):
            #             smpl_nodes.instances_quats[i, instance_id] = global_orient[i]
            
            # for i in range(num_frames):
            #     if i < len(body_pose):
            #         smpl_nodes.smpl_qauts[i, instance_id] = body_pose[i]
            
            # if not keep_translation:
            #     for i in range(num_frames):
            #         if i < len(trans):
            #             smpl_nodes.instances_trans[i, instance_id] = trans[i]
            
            # 6. 更新SMPL形状参数
            if "betas" in new_instance and hasattr(smpl_nodes.template, "init_beta"):
                smpl_nodes.template.init_beta[instance_id] = new_instance["betas"].to(device)
                if hasattr(smpl_nodes.template, "_template_layer"):  # 检查是否存在底层模型
                    # 手动触发模板更新
                    smpl_output = smpl_nodes.template._template_layer(
                        betas=smpl_nodes.template.init_beta[instance_id:instance_id+1],
                        body_pose=smpl_nodes.template.canonical_pose[None, 1:],
                        global_orient=smpl_nodes.template.canonical_pose[None, 0],
                        return_full_pose=True,
                    )
                    smpl_nodes.template.J_canonical[instance_id] = smpl_output.J[0]
            
            # 7. 重置渲染缓存
            if hasattr(trainer, "reset_renderer_cache"):
                trainer.reset_renderer_cache()
            
            logger.info(f"实例 {instance_id} 替换完成，点数: {points_per_instance}")
            
    except Exception as e:
        logger.error(f"实例替换失败: {str(e)}")
        raise

def batch_render_with_eval(cfg, trainer, dataset, output_dir: str, log_dir: str):
    """使用eval.py的渲染逻辑进行批量渲染，并确保输出到正确的目录"""
    import shutil
    from tools.eval import do_evaluation

    # 创建伪args对象
    class Args:
        def __init__(self):
            self.save_catted_videos = True
            self.enable_viewer = False
            self.render_video_postfix = "_new"  # 更改后缀，便于识别

    fake_args = Args()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保视频目录存在
    videos_dir = os.path.join(log_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # 创建新的视频目录
    videos_new_dir = os.path.join(log_dir, "videos_new")
    os.makedirs(videos_new_dir, exist_ok=True)
    
    # 为其他渲染键创建目录
    for key in ["rgbs", "depths"]:
        os.makedirs(os.path.join(log_dir, key), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"{key}_new"), exist_ok=True)

    # 直接使用输出目录作为渲染目标
    orig_log_dir = None
    if hasattr(cfg, "logging") and hasattr(cfg.logging, "log_dir"):
        orig_log_dir = cfg.logging.log_dir  # 保存原始log_dir
    
    if not hasattr(cfg, "logging"):
        cfg.logging = OmegaConf.create({})
    
    # 临时将log_dir设为output_dir，确保直接渲染到目标位置
    cfg.logging.log_dir = output_dir
    logger.info(f"将渲染输出设置为: {output_dir}")

    # 执行渲染
    try:
        logger.info("开始渲染...")
        do_evaluation(
            step=trainer.step,
            cfg=cfg,
            trainer=trainer,
            dataset=dataset,
            args=fake_args,
            render_keys=["rgbs"],
            post_fix="_new",
            log_metrics=False,
        )
        logger.info("渲染完成。")
        
        # 检查一下output_dir中是否有生成的文件
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            if files:
                logger.info(f"输出目录 {output_dir} 中的文件: {files}")
            else:
                logger.warning(f"输出目录 {output_dir} 是空的")
                
                # 寻找渲染结果并复制到输出目录
                logger.info("尝试从其他位置查找渲染结果...")
                potential_dirs = [
                    videos_dir, 
                    videos_new_dir,
                    os.path.join(log_dir, "rgbs"),
                    os.path.join(log_dir, "depths"),
                    os.path.join(log_dir, "rgbs_new"),
                    os.path.join(log_dir, "depths_new")
                ]
                
                for src_dir in potential_dirs:
                    if os.path.exists(src_dir):
                        logger.info(f"查找目录: {src_dir}")
                        for f in os.listdir(src_dir):
                            if f.endswith('.mp4') or f.endswith('.avi') or f.endswith('.png'):
                                src_file = os.path.join(src_dir, f)
                                dst_file = os.path.join(output_dir, f)
                                logger.info(f"复制文件: {src_file} -> {dst_file}")
                                shutil.copy2(src_file, dst_file)
                
        else:
            logger.error(f"输出目录 {output_dir} 不存在")
                
    except Exception as e:
        logger.error(f"使用eval.py渲染过程中发生错误: {str(e)}")
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
            
            # 保存视频直接到输出目录
            if all_rgbs:
                rgb_video_path = os.path.join(output_dir, "test_rgbs_new.mp4")
                save_videos(all_rgbs, rgb_video_path)
                logger.info(f"RGB视频保存到: {rgb_video_path}")
            
            if all_depths:
                depth_video_path = os.path.join(output_dir, "test_depths_new.mp4")
                save_videos(all_depths, depth_video_path)
                logger.info(f"深度视频保存到: {depth_video_path}")
        except Exception as e2:
            logger.error(f"直接渲染也失败: {str(e2)}")
    
    finally:
        # 恢复原始log_dir
        if orig_log_dir is not None:
            cfg.logging.log_dir = orig_log_dir
        
        # 再次检查输出目录
        logger.info(f"最终检查输出目录: {output_dir}")
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            if files:
                logger.info(f"输出目录中的文件: {files}")
            else:
                logger.warning("输出目录仍然为空")

def main(args):
    # 初始化配置
    config_dir = os.path.dirname(args.resume_from)
    config_path = os.path.join(config_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.merge_with(OmegaConf.from_dotlist(args.opts))

    # 设置输出目录
    log_dir = os.path.dirname(args.resume_from)
    output_dir = os.path.join(log_dir, "replaced_smpl_output")
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
    
    # 检查SMPL节点结构（调试用）
    inspect_smpl_node_structure(trainer.models["SMPLNodes"])
    
    # 加载并转换新实例
    if args.new_npz_path and args.new_ply_path:
        logger.info(f"从 {args.new_npz_path} 和 {args.new_ply_path} 加载新实例")
        
        # 获取当前模型的帧数
        smpl_nodes = trainer.models["SMPLNodes"]
        num_frames = smpl_nodes.instances_quats.shape[0]
        logger.info(f"当前模型帧数: {num_frames}")
        
        # 使用改进的Skeleton加载方法
        new_instance = load_and_convert_using_skeleton(
            args.new_ply_path, 
            args.new_npz_path,
            args.smplx_path if hasattr(args, 'smplx_path') else "smpl_models", 
            num_frames
        )
        
        # 替换实例
        with torch.no_grad():
            replace_smpl_instance_improved(
                trainer, 
                args.instance_id, 
                new_instance,
                keep_translation=args.keep_translation,
                keep_global_rot=args.keep_global_rot
            )
    
    # 批量渲染
    logger.info(f"开始渲染到: {output_dir}")
    batch_render_with_eval(cfg, trainer, dataset, output_dir, log_dir)
    logger.info(f"渲染完成，保存到: {output_dir}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    parser = argparse.ArgumentParser("Replace SMPL instance with new data")
    parser.add_argument(
        "--resume_from", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--instance_id", type=int, default=1, help="SMPL instance ID to replace"
    )
    parser.add_argument(
        "--new_npz_path", type=str, default="", help="Path to motion data npz file"
    )
    parser.add_argument(
        "--new_ply_path", type=str, default="", help="Path to .ply file with point cloud"
    )
    parser.add_argument(
        "--smplx_path", type=str, default="smpl_models", help="Path to SMPLX models folder"
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