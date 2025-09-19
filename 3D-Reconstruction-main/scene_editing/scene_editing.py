from typing import List, Dict, Optional, Tuple, Any
from omegaconf import OmegaConf
import os
import time
import logging
import torch
import numpy as np
import math

# 特定的3D转换库
from pytorch3d.transforms import (
    matrix_to_quaternion, 
    quaternion_to_matrix,
    axis_angle_to_matrix, 
    axis_angle_to_quaternion
)

# 项目特定导入
from utils.simplify_loc2rot import joints2smpl
logger = logging.getLogger()
# 在现有的 scene_editing.py 文件中添加以下函数

import pickle
import json
import time
from pathlib import Path
from typing import Union, List, Dict, Optional, Any
import torch
import numpy as np
from torch.nn import Parameter

# 导入必要的函数，添加错误处理
try:
    from models.gaussians.basics import RGB2SH, num_sh_bases, random_quat_tensor, k_nearest_sklearn
except ImportError as e:
    logger.warning(f"部分gaussians.basics函数导入失败: {e}")
    # 提供备用实现
    try:
        from models.gaussians.basics import RGB2SH, num_sh_bases, random_quat_tensor
    except ImportError:
        logger.error("无法导入基础gaussian函数")


# k_nearest_sklearn的备用实现
def k_nearest_sklearn(points, k=3):
    """k近邻的简单备用实现"""
    try:
        from sklearn.neighbors import NearestNeighbors
        points_np = points.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points_np)
        distances, indices = nbrs.kneighbors(points_np)
        return distances[:, 1:], indices[:, 1:]  # 排除自己
    except ImportError:
        # 如果sklearn也不可用，返回固定值
        n_points = points.shape[0]
        distances = np.ones((n_points, k)) * 0.01
        indices = np.zeros((n_points, k), dtype=int)
        return distances, indices
# ------------------- RGB2SH 转换函数 -------------------

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

# ------------------- 序列处理函数 -------------------

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

# ------------------- SMPL节点检查和替换函数 -------------------

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

# ------------------- 姿势加载与替换函数 -------------------

def load_pose_sequence(npy_path: str) -> torch.Tensor:
    """从npy文件加载姿势序列（优先使用缓存，避免重复转换）"""
    print(f"\n=== 加载姿势序列 ===")
    print(f"源文件路径: {os.path.abspath(npy_path)}")
    
    # 生成缓存文件路径（在原文件名后添加_smpl）
    cache_path = os.path.join(
        os.path.dirname(npy_path),
        os.path.basename(npy_path).replace(".npy", "_smpl.npy")
    )

    # 1. 优先尝试加载缓存
    if os.path.exists(cache_path):
        print(f"检测到缓存文件，直接加载: {cache_path}")
        try:
            cached_data = np.load(cache_path)
            print(f"成功加载缓存数据，形状: {cached_data.shape}")
            return torch.from_numpy(cached_data).float().to("cuda")
        except Exception as e:
            print(f"缓存加载失败（将重新转换）: {str(e)}")

    # 2. 无缓存时执行完整转换流程
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"错误：原始文件不存在 {npy_path}")

    print("执行原始数据转换...")
    # 原始数据加载和转换流程
    loc_data = np.load(npy_path)
    print(f"原始数据形状: {loc_data.shape}")

    # SMPL转换
    converter = joints2smpl(num_frames=loc_data.shape[1], device_id=0)
    result = converter.joint2smpl(loc_data[0])  # 输入(nframe,22,3)
    
    # 获取旋转数据
    rot_data = result[1] if isinstance(result, tuple) else result
    print(f"转换后数据形状: {rot_data.shape}")

    # 格式转换
    rot_data = rot_data.reshape(-1, 24, 3)
    quat_data = axis_angle_to_quaternion(torch.tensor(rot_data)).numpy()
    print(f"最终四元数形状: {quat_data.shape}")

    # 保存缓存
    try:
        np.save(cache_path, quat_data)
        print(f"转换结果已缓存至: {cache_path}")
    except Exception as e:
        print(f"警告：缓存保存失败（不影响使用）: {str(e)}")

    return torch.from_numpy(quat_data).float().to("cuda")

def replace_smpl_pose(
    trainer, 
    instance_id: int, 
    new_poses: torch.Tensor,
    keep_translation: bool = True,
    keep_global_rot: bool = True
):
    """替换SMPL姿势，处理转换后的四元数数据"""
    smpl_nodes = trainer.models["SMPLNodes"]
    original_seq = get_full_smpl_sequence(trainer, instance_id)
    
    print(f"\n原始SMPL关节旋转形状: {original_seq['smpl_quats'].shape}")
    print(f"输入四元数形状: {new_poses.shape}")

    print(f"\nSMPL全局旋转形状: {original_seq['global_quats'].shape}")
    
    # 打印模型参数形状
    logger.info(f"模型参数形状:")
    logger.info(f"instances_quats: {smpl_nodes.instances_quats.shape}")
    logger.info(f"instances_trans: {smpl_nodes.instances_trans.shape}")
    logger.info(f"smpl_qauts: {smpl_nodes.smpl_qauts.shape}")

    # 调整长度匹配
    num_frames = original_seq["smpl_quats"].shape[0]
    new_poses = match_sequence_length(num_frames, new_poses)
    
    # 确保输入是(nframe,24,4)
    assert new_poses.shape[1] == 24, f"输入应有24个关节，实际: {new_poses.shape[1]}"
    assert new_poses.shape[2] == 4, f"四元数应为4维，实际: {new_poses.shape[2]}"
    
    # 更新模型参数
    with torch.no_grad():
        # 保留原始的第一个旋转（全局旋转）
        if keep_global_rot:
            smpl_nodes.instances_quats[:, instance_id] = original_seq["global_quats"]
        else:
            smpl_nodes.instances_quats[:, instance_id] = new_poses[:, 0].unsqueeze(1)


        # 替换后23个关节旋转（索引1-23对应SMPL的1-23关节）
        smpl_nodes.smpl_qauts[:, instance_id, :] = new_poses[:, 1:24]  # 使用转换后的1-23关节
        
        # 保留原始平移
        if keep_translation:
            smpl_nodes.instances_trans[:, instance_id].copy_(original_seq["translations"])
    
    # 验证更新
    updated_seq = get_full_smpl_sequence(trainer, instance_id)
    print(f"\n更新后SMPL关节旋转形状: {updated_seq['smpl_quats'].shape}")


# ------------------- 渲染函数 -------------------

def batch_render_with_eval(cfg, trainer, dataset, output_dir: str, log_dir: str, post_fix="_new"):
    """使用eval.py的渲染逻辑进行批量渲染，并确保输出到正确的目录"""
    from tools.eval import do_evaluation

    # 创建伪args对象
    class Args:
        def __init__(self):
            self.save_catted_videos = True
            self.enable_viewer = False
            self.render_video_postfix = post_fix

    fake_args = Args()
    

    
    # 创建新的视频目录
    videos_new_dir = os.path.join(log_dir, f"videos{post_fix}")
    os.makedirs(videos_new_dir, exist_ok=True)


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
            post_fix=post_fix,
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
                      
        else:
            logger.error(f"输出目录 {output_dir} 不存在")
                
    except Exception as e:
        logger.error(f"使用eval.py渲染过程中发生错误: {str(e)}")


# 以下是需要添加到scene_editing.py的新函数

def edit_rigid_nodes(trainer, args):
    """
    编辑刚体节点，执行删除、替换或添加位移操作
    
    Args:
        trainer: 训练器实例
        args: 命令行参数，包含operation等信息
        
    Returns:
        tuple: (更新后的RigidNodes对象, 编辑参数字典)
    """
    # 检查是否存在RigidNodes模型
    if "RigidNodes" not in trainer.models:
        raise ValueError("模型中没有找到RigidNodes")
    
    # 获取RigidNodes模型
    rigid_nodes = trainer.models["RigidNodes"]
    logger.info(f"找到RigidNodes模型，实例数量: {rigid_nodes.num_instances}")
    
    operation = args.operation.lower()
    
    # 保存编辑参数
    edit_params = {
        "operation": operation,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
    }
    
    try:
        # 根据操作类型执行不同的编辑
        if operation == "remove":
            # 删除指定ID的实例
            instance_ids = args.instance_ids
            logger.info(f"执行删除操作，实例IDs: {instance_ids}")
            edit_params["instance_ids"] = instance_ids
            
            # 检查实例ID是否有效
            all_ids = rigid_nodes.point_ids[..., 0].unique().cpu().numpy()
            logger.info(f"当前所有实例ID: {all_ids}")
            
            # 筛选有效ID
            valid_ids = [id for id in instance_ids if id in all_ids]
            if len(valid_ids) < len(instance_ids):
                logger.warning(f"部分实例ID无效: {set(instance_ids) - set(valid_ids)}")
            
            if valid_ids:
                rigid_nodes.remove_instances(valid_ids)
                logger.info(f"已删除实例: {valid_ids}")
            else:
                logger.warning("没有有效的实例ID可删除")
        
        elif operation == "replace":
            # 替换实例，格式为 "源ID:目标ID" 或 "源ID1:目标ID1,源ID2:目标ID2"
            instance_map_str = args.instance_map
            logger.info(f"执行替换操作，实例映射: {instance_map_str}")
            
            # 解析映射字符串
            replace_map = {}
            for pair in instance_map_str.split(','):
                if ':' not in pair:
                    logger.warning(f"忽略无效的映射: {pair}")
                    continue
                    
                source_id, target_id = map(int, pair.split(':'))
                replace_map[source_id] = target_id
            
            edit_params["replace_map"] = replace_map
            
            # 检查实例ID是否有效
            all_ids = rigid_nodes.point_ids[..., 0].unique().cpu().numpy()
            logger.info(f"当前所有实例ID: {all_ids}")
            
            # 筛选有效映射
            valid_map = {}
            for src, tgt in replace_map.items():
                if src in all_ids and tgt in all_ids:
                    valid_map[src] = tgt
                else:
                    logger.warning(f"无效的映射 {src}:{tgt}，实例ID不存在")
            
            if valid_map:
                rigid_nodes.replace_instances(valid_map)
                logger.info(f"已替换实例: {valid_map}")
            else:
                logger.warning("没有有效的实例映射可执行")
        
        elif operation == "offset":
            # 添加位移偏移
            instance_id = args.instance_id
            offset = args.offset
            logger.info(f"执行位移操作，实例ID: {instance_id}，偏移量: {offset}")
            
            edit_params["instance_id"] = instance_id
            edit_params["offset"] = offset
            
            # 检查实例ID是否有效
            all_ids = rigid_nodes.point_ids[..., 0].unique().cpu().numpy()
            if instance_id not in all_ids:
                logger.warning(f"实例ID {instance_id} 不存在")
                return rigid_nodes, edit_params
            
            # 将偏移转换为张量
            translation_offset = torch.tensor(offset, device=trainer.device)
            
            # 确定帧数范围
            if hasattr(args, "frame_range") and args.frame_range:
                try:
                    if "-" in args.frame_range:
                        start_frame, end_frame = map(int, args.frame_range.split("-"))
                        frame_indices = list(range(start_frame, end_frame + 1))
                    else:
                        frame_indices = [int(args.frame_range)]
                except ValueError:
                    logger.warning(f"无效的帧范围格式: {args.frame_range}，将应用于所有帧")
                    frame_indices = list(range(rigid_nodes.num_frames))
            else:
                # 如果未指定，应用于所有帧
                frame_indices = list(range(rigid_nodes.num_frames))
            
            edit_params["frame_indices"] = frame_indices
            
            # 应用偏移
            for frame_idx in frame_indices:
                rigid_nodes.add_transform_offset(
                    instance_id=instance_id,
                    frame_idx=frame_idx,
                    translation_offset=translation_offset
                )
            
            logger.info(f"已为实例 {instance_id} 的 {len(frame_indices)} 帧添加位移偏移: {offset}")
        
        elif operation == "rotate":
            # 添加旋转偏移
            instance_id = args.instance_id
            rotation_axis = args.rotation
            angle_degrees = args.angle
            logger.info(f"执行旋转操作，实例ID: {instance_id}，旋转轴: {rotation_axis}，角度: {angle_degrees}度")
            
            edit_params["instance_id"] = instance_id
            edit_params["rotation_axis"] = rotation_axis
            edit_params["angle_degrees"] = angle_degrees
            
            # 检查实例ID是否有效
            all_ids = rigid_nodes.point_ids[..., 0].unique().cpu().numpy()
            if instance_id not in all_ids:
                logger.warning(f"实例ID {instance_id} 不存在")
                return rigid_nodes, edit_params
            
            # 将角度转换为弧度
            angle_radians = math.radians(angle_degrees)
            
            # 归一化旋转轴
            rotation_axis = torch.tensor(rotation_axis, device=trainer.device)
            rotation_axis = rotation_axis / torch.norm(rotation_axis)
            
            # 创建轴角表示
            axis_angle = rotation_axis * angle_radians
            
            # 转换为四元数(PyTorch3D中四元数格式为xyzw)
            rotation_quaternion = axis_angle_to_quaternion(axis_angle)
            
            # 确定帧数范围
            if hasattr(args, "frame_range") and args.frame_range:
                try:
                    if "-" in args.frame_range:
                        start_frame, end_frame = map(int, args.frame_range.split("-"))
                        frame_indices = list(range(start_frame, end_frame + 1))
                    else:
                        frame_indices = [int(args.frame_range)]
                except ValueError:
                    logger.warning(f"无效的帧范围格式: {args.frame_range}，将应用于所有帧")
                    frame_indices = list(range(rigid_nodes.num_frames))
            else:
                # 如果未指定，应用于所有帧
                frame_indices = list(range(rigid_nodes.num_frames))
            
            edit_params["frame_indices"] = frame_indices
            
            # 应用旋转
            for frame_idx in frame_indices:
                rigid_nodes.add_transform_offset(
                    instance_id=instance_id,
                    frame_idx=frame_idx,
                    rotation_offset=rotation_quaternion
                )
            
            logger.info(f"已为实例 {instance_id} 的 {len(frame_indices)} 帧添加旋转: 轴={rotation_axis.cpu().numpy()}, 角度={angle_degrees}度")
        
        else:
            logger.error(f"不支持的操作类型: {operation}")
            logger.info("支持的操作: remove, replace, offset, rotate")
    
    except Exception as e:
        logger.error(f"编辑过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 返回更新后的模型和编辑参数
    return rigid_nodes, edit_params


def print_node_info(trainer):
    """
    打印场景中的各类节点（Rigid, SMPL等）信息
    
    Args:
        trainer: 训练器实例
        
    Returns:
        dict: 包含节点信息的字典
    """
    logger.info("=" * 50)
    logger.info("场景节点信息摘要")
    logger.info("=" * 50)
    
    nodes_info = {}
    
    # 检查并打印所有可用的节点类型
    available_node_types = list(trainer.models.keys())
    logger.info(f"可用节点类型: {available_node_types}")
    
    # 依次处理每种节点类型
    for node_type in available_node_types:
        node = trainer.models[node_type]
        logger.info(f"\n--- {node_type} 信息 ---")
        
        # 获取基本属性
        node_attrs = {}
        
        # 记录公共属性
        common_attrs = ["num_points", "num_instances", "num_frames"]
        for attr in common_attrs:
            if hasattr(node, attr):
                try:
                    if callable(getattr(node, attr)):
                        value = getattr(node, attr)()
                    else:
                        value = getattr(node, attr)
                    
                    if isinstance(value, torch.Tensor):
                        value = value.item() if value.numel() == 1 else value.shape
                    
                    node_attrs[attr] = value
                    logger.info(f"  {attr}: {value}")
                except Exception as e:
                    logger.warning(f"  无法获取 {attr}: {str(e)}")
        
        # 特定类型的节点处理
        if node_type == "RigidNodes":
            try:
                # 获取实例ID信息
                if hasattr(node, "point_ids"):
                    unique_ids = node.point_ids[..., 0].unique().cpu().numpy()
                    logger.info(f"  实例ID: {unique_ids}")
                    node_attrs["instance_ids"] = unique_ids.tolist()
                    
                    # 统计每个实例的点数
                    instance_points = {}
                    for id in unique_ids:
                        mask = node.point_ids[..., 0] == id
                        count = mask.sum().item()
                        instance_points[int(id)] = count
                    
                    logger.info(f"  每个实例的点数: {instance_points}")
                    node_attrs["instance_points"] = instance_points
                
                # 获取变换信息
                if hasattr(node, "instances_quats") and hasattr(node, "instances_trans"):
                    logger.info(f"  变换形状:")
                    logger.info(f"    四元数: {node.instances_quats.shape}")
                    logger.info(f"    平移: {node.instances_trans.shape}")
                    node_attrs["transform_shapes"] = {
                        "quaternions": tuple(node.instances_quats.shape),
                        "translations": tuple(node.instances_trans.shape)
                    }
            except Exception as e:
                logger.warning(f"  处理RigidNodes时出错: {str(e)}")
        
        elif node_type == "SMPLNodes":
            try:
                # 获取实例ID信息
                if hasattr(node, "point_ids"):
                    unique_ids = node.point_ids[..., 0].unique().cpu().numpy()
                    logger.info(f"  实例ID: {unique_ids}")
                    node_attrs["instance_ids"] = unique_ids.tolist()
                    
                    # 统计每个实例的点数
                    instance_points = {}
                    for id in unique_ids:
                        mask = node.point_ids[..., 0] == id
                        count = mask.sum().item()
                        instance_points[int(id)] = count
                    
                    logger.info(f"  每个实例的点数: {instance_points}")
                    node_attrs["instance_points"] = instance_points
                
                # SMPL特有属性
                if hasattr(node, "smpl_points_num"):
                    logger.info(f"  每个SMPL实例的点数: {node.smpl_points_num}")
                    node_attrs["smpl_points_num"] = node.smpl_points_num
                
                # 获取姿态信息
                if hasattr(node, "smpl_qauts") and hasattr(node, "instances_quats"):
                    logger.info(f"  姿态形状:")
                    logger.info(f"    全局旋转: {node.instances_quats.shape}")
                    logger.info(f"    关节旋转: {node.smpl_qauts.shape}")
                    node_attrs["pose_shapes"] = {
                        "global_quats": tuple(node.instances_quats.shape),
                        "joint_quats": tuple(node.smpl_qauts.shape)
                    }
                
                # 获取可见性信息
                if hasattr(node, "instances_fv"):
                    visible_frames = torch.sum(node.instances_fv, dim=0).cpu().numpy()
                    logger.info(f"  各实例的可见帧数: {visible_frames}")
                    node_attrs["visible_frames"] = visible_frames.tolist()
            except Exception as e:
                logger.warning(f"  处理SMPLNodes时出错: {str(e)}")
        
        elif node_type == "DeformableNodes":
            try:
                # 特定于DeformableNodes的属性
                if hasattr(node, "def_points_num"):
                    logger.info(f"  可变形点数: {node.def_points_num}")
                    node_attrs["def_points_num"] = node.def_points_num
                
                # 获取变形信息
                if hasattr(node, "instances_def"):
                    logger.info(f"  变形参数形状: {node.instances_def.shape}")
                    node_attrs["deform_shape"] = tuple(node.instances_def.shape)
            except Exception as e:
                logger.warning(f"  处理DeformableNodes时出错: {str(e)}")
        
        nodes_info[node_type] = node_attrs
    
    logger.info("=" * 50)
    
    return nodes_info

def get_model_key(instance_type):
    """根据实例类型获取对应的模型键名"""
    if instance_type.lower() == "smpl":
        return "SMPLNodes"
    elif instance_type.lower() == "rigid":
        return "RigidNodes"
    else:
        return f"{instance_type.capitalize()}Nodes"

# 保存场景节点信息为JSON文件
def save_node_info(nodes_info, output_dir):
    """
    将节点信息保存为JSON文件
    
    Args:
        nodes_info: 节点信息字典
        output_dir: 输出目录
    """
    import json
    
    # 处理不可序列化的对象
    def make_serializable(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], (np.ndarray, torch.Tensor)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    serializable_info = make_serializable(nodes_info)
    
    # 添加时间戳
    serializable_info["_metadata"] = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "description": "场景节点信息"
    }
    
    info_file = os.path.join(output_dir, "scene_nodes_info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"节点信息已保存到 {info_file}")

# ------------------- 实例保存和加载函数 -------------------

def save_smpl_instance(
    trainer, 
    instance_id: int, 
    save_path: str,
    save_metadata: bool = True
) -> Dict[str, Any]:
    """
    保存单个SMPL实例的所有数据到文件
    
    Args:
        trainer: 训练器实例
        instance_id: 要保存的实例ID
        save_path: 保存路径
        save_metadata: 是否保存元数据信息
        
    Returns:
        保存的数据字典
    """
    logger.info(f"=== 开始保存SMPL实例 {instance_id} ===")
    
    if "SMPLNodes" not in trainer.models:
        raise ValueError("模型中没有找到SMPLNodes")
    
    smpl_nodes = trainer.models["SMPLNodes"]
    
    # 检查实例ID是否存在
    all_ids = smpl_nodes.point_ids[..., 0].unique().cpu().numpy()
    if instance_id not in all_ids:
        raise ValueError(f"实例ID {instance_id} 不存在，可用ID: {all_ids}")
    
    # 获取该实例的点掩码
    points_per_instance = smpl_nodes.smpl_points_num  # 6890
    start_idx = instance_id * points_per_instance
    end_idx = (instance_id + 1) * points_per_instance
    pts_mask = smpl_nodes.point_ids[..., 0] == instance_id
    
    logger.info(f"实例 {instance_id} 的点数: {pts_mask.sum().item()}")
    
    # 收集实例数据
    instance_data = {}
    
    with torch.no_grad():
        # 1. 基础几何数据 - 确保数据类型和设备一致性
        instance_data["geometry"] = {
            "_means": smpl_nodes._means[pts_mask].detach().clone().cpu(),
            "_scales": smpl_nodes._scales[pts_mask].detach().clone().cpu(),
            "_quats": smpl_nodes._quats[pts_mask].detach().clone().cpu(),
            "_features_dc": smpl_nodes._features_dc[pts_mask].detach().clone().cpu(),
            "_features_rest": smpl_nodes._features_rest[pts_mask].detach().clone().cpu(),
            "_opacities": smpl_nodes._opacities[pts_mask].detach().clone().cpu(),
            # 添加：点ID数据
            "point_ids": smpl_nodes.point_ids[pts_mask].detach().clone().cpu(),
        }
        
        # 2. 姿态和变换数据 - 确保完整复制，注意instances_size的正确保存
        instance_data["motion"] = {
            "instances_quats": smpl_nodes.instances_quats[:, instance_id].detach().clone().cpu(),
            "instances_trans": smpl_nodes.instances_trans[:, instance_id].detach().clone().cpu(),
            "smpl_qauts": smpl_nodes.smpl_qauts[:, instance_id].detach().clone().cpu(),
            "instances_fv": smpl_nodes.instances_fv[:, instance_id].detach().clone().cpu(),
        }
        
        # 单独保存实例尺寸信息，不放在motion里避免被序列处理
        if hasattr(smpl_nodes, 'instances_size'):
            instance_data["instances_size"] = smpl_nodes.instances_size[instance_id].detach().clone().cpu()
        
        # 3. SMPL模板数据 - 保证精度
        instance_data["smpl_template"] = {}
        if hasattr(smpl_nodes.template, "init_beta"):
            instance_data["smpl_template"]["betas"] = smpl_nodes.template.init_beta[instance_id].detach().clone().cpu()
        
        if hasattr(smpl_nodes.template, "J_canonical"):
            instance_data["smpl_template"]["J_canonical"] = smpl_nodes.template.J_canonical[instance_id].detach().clone().cpu()
            
        if hasattr(smpl_nodes.template, "W"):
            instance_data["smpl_template"]["W"] = smpl_nodes.template.W[instance_id].detach().clone().cpu()
        
        # 添加：SMPL模板的其他重要数据
        if hasattr(smpl_nodes.template, "A0_inv"):
            instance_data["smpl_template"]["A0_inv"] = smpl_nodes.template.A0_inv[instance_id].detach().clone().cpu()
        
        # 添加：canonical_pose 如果存在
        if hasattr(smpl_nodes.template, "canonical_pose"):
            instance_data["smpl_template"]["canonical_pose"] = smpl_nodes.template.canonical_pose.detach().clone().cpu()
        
        # 添加：其他可能的模板数据
        if hasattr(smpl_nodes.template, "_template_layer") and hasattr(smpl_nodes.template._template_layer, "parents"):
            instance_data["smpl_template"]["parents"] = smpl_nodes.template._template_layer.parents.detach().clone().cpu()
        
        # 4. 体素变形器数据（如果存在）
        if hasattr(smpl_nodes.template, "voxel_deformer") and smpl_nodes.use_voxel_deformer:
            instance_data["voxel_deformer"] = {
                "lbs_voxel_base": smpl_nodes.template.voxel_deformer.lbs_voxel_base[instance_id].detach().clone().cpu(),
                "offset": smpl_nodes.template.voxel_deformer.offset[instance_id].detach().clone().cpu(),
                "scale": smpl_nodes.template.voxel_deformer.scale[instance_id].detach().clone().cpu(),
            }
            if hasattr(smpl_nodes.template.voxel_deformer, "voxel_w_correction"):
                instance_data["voxel_deformer"]["voxel_w_correction"] = smpl_nodes.template.voxel_deformer.voxel_w_correction[instance_id].detach().clone().cpu()
            
            # 添加：体素变形器的全局参数（这些可能是所有实例共享的）
            if hasattr(smpl_nodes.template.voxel_deformer, "global_scale"):
                instance_data["voxel_deformer"]["global_scale"] = smpl_nodes.template.voxel_deformer.global_scale.detach().clone().cpu()
            if hasattr(smpl_nodes.template.voxel_deformer, "ratio"):
                instance_data["voxel_deformer"]["ratio"] = smpl_nodes.template.voxel_deformer.ratio.detach().clone().cpu()
            if hasattr(smpl_nodes.template.voxel_deformer, "grid_denorm"):
                instance_data["voxel_deformer"]["grid_denorm"] = smpl_nodes.template.voxel_deformer.grid_denorm[instance_id].detach().clone().cpu()
            if hasattr(smpl_nodes.template.voxel_deformer, "resolution_dhw"):
                instance_data["voxel_deformer"]["resolution_dhw"] = smpl_nodes.template.voxel_deformer.resolution_dhw
            if hasattr(smpl_nodes.template.voxel_deformer, "bbox"):
                instance_data["voxel_deformer"]["bbox"] = smpl_nodes.template.voxel_deformer.bbox[instance_id].detach().clone().cpu()
        
        # 5. 额外的约束数据（如果存在）
        if hasattr(smpl_nodes, "on_mesh_x") and smpl_nodes.ctrl_cfg.get("constrain_xyz_offset", False):
            instance_data["constraints"] = {
                "on_mesh_x": smpl_nodes.on_mesh_x[pts_mask].detach().clone().cpu(),
            }
        
        # 6. KNN数据（如果存在且配置需要）
        if hasattr(smpl_nodes, "nn_ind"):
            instance_data["knn_data"] = {
                "nn_ind": smpl_nodes.nn_ind[instance_id].detach().clone().cpu(),
            }
        
        # 7. 控制配置数据
        instance_data["control_config"] = {
            "ball_gaussians": smpl_nodes.ball_gaussians if hasattr(smpl_nodes, "ball_gaussians") else False,
            "use_voxel_deformer": smpl_nodes.use_voxel_deformer,
            "smpl_points_num": smpl_nodes.smpl_points_num,
            "sh_degree": smpl_nodes.sh_degree,
            # 保存重要的控制配置
            "ctrl_cfg": {
                "opacity_init_value": smpl_nodes.ctrl_cfg.get("opacity_init_value", 0.1),
                "constrain_xyz_offset": smpl_nodes.ctrl_cfg.get("constrain_xyz_offset", False),
                "knn_neighbors": smpl_nodes.ctrl_cfg.get("knn_neighbors", 50),
                "freeze_x": smpl_nodes.ctrl_cfg.get("freeze_x", False),
                "freeze_s": smpl_nodes.ctrl_cfg.get("freeze_s", False),
                "freeze_q": smpl_nodes.ctrl_cfg.get("freeze_q", False),
                "freeze_o": smpl_nodes.ctrl_cfg.get("freeze_o", False),
                "freeze_shs_dc": smpl_nodes.ctrl_cfg.get("freeze_shs_dc", False),
                "freeze_shs_rest": smpl_nodes.ctrl_cfg.get("freeze_shs_rest", False),
            }
        }
        
        # 8. 渲染相关的状态数据
        if hasattr(smpl_nodes, "normalized_timestamps"):
            instance_data["render_state"] = {
                "normalized_timestamps": smpl_nodes.normalized_timestamps.detach().clone().cpu(),
            }
        
        # 9. 元数据
        if save_metadata:
            instance_data["metadata"] = {
                "instance_id": instance_id,
                "num_points": pts_mask.sum().item(),
                "num_frames": smpl_nodes.instances_quats.shape[0],
                "smpl_points_num": smpl_nodes.smpl_points_num,
                "use_voxel_deformer": smpl_nodes.use_voxel_deformer,
                "sh_degree": smpl_nodes.sh_degree,
                "save_timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
                "device": str(smpl_nodes._means.device),
                # 添加：SMPLNodes类的属性信息
                "class_prefix": smpl_nodes.class_prefix,
                "num_instances": smpl_nodes.num_instances,
                "num_frames": smpl_nodes.num_frames,
                # 体素变形器状态
                "voxel_deformer_enabled": hasattr(smpl_nodes.template, "voxel_deformer") and smpl_nodes.use_voxel_deformer,
            }

        # ========== 新增：单独保存轨迹数据 ==========
        # 提取轨迹数据
        frame_count = smpl_nodes.instances_quats.shape[0]
        frame_indices = list(range(frame_count))
        obj_to_world_matrices = []
        
        # 计算每一帧的obj_to_world变换矩阵
        for frame_idx in range(frame_count):
            # 获取该帧的位置和旋转
            translation = smpl_nodes.instances_trans[frame_idx, instance_id].detach().clone().cpu().numpy()
            rotation_quat = smpl_nodes.instances_quats[frame_idx, instance_id].detach().clone().cpu().numpy()
            
            # 四元数转旋转矩阵 (假设四元数格式为 [w, x, y, z])
            def quat_to_rotation_matrix(quat):
                """将四元数转换为3x3旋转矩阵"""
                if len(quat) == 4:
                    w, x, y, z = quat
                    # 归一化四元数
                    norm = np.sqrt(w*w + x*x + y*y + z*z)
                    if norm > 1e-8:
                        w, x, y, z = w/norm, x/norm, y/norm, z/norm
                    
                    # 计算旋转矩阵
                    return np.array([
                        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
                    ])
                else:
                    # 如果不是4元素，返回单位矩阵
                    return np.eye(3)
            
            # 构建4x4变换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = quat_to_rotation_matrix(rotation_quat)
            transform_matrix[:3, 3] = translation
            
            # 转换为列表格式，保持与目标格式一致
            matrix_list = transform_matrix.tolist()
            obj_to_world_matrices.append(matrix_list)
        
        # 构建轨迹数据字典，格式与目标JSON完全一致
        trajectory_data = {
            str(instance_id): {
                "id": instance_id,
                "class_name": "Pedestrian",  # 或者根据实际情况设置
                "frame_annotations": {
                    "frame_idx": frame_indices,
                    "obj_to_world": obj_to_world_matrices
                }
            }
        }
        
        # 保存轨迹数据到单独的JSON文件
        trajectory_save_path = Path(save_path).parent / f"trajectory_{Path(save_path).stem}.json"
        with open(trajectory_save_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"轨迹数据已单独保存到: {trajectory_save_path}")
        logger.info(f"轨迹包含 {len(frame_indices)} 帧数据")
    # 保存到文件
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用pickle保存，保持tensor格式
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(instance_data, f)
    
    logger.info(f"SMPL实例 {instance_id} 已保存到: {save_path}")
    logger.info(f"保存的数据大小: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 打印保存的数据统计
    for category, data in instance_data.items():
        if category == "metadata":
            continue
        logger.info(f"  {category} 包含 {len(data)} 个属性" if isinstance(data, dict) else f"  {category}: {type(data)}")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"    {key}: {value.shape} {value.dtype}")
                else:
                    logger.info(f"    {key}: {type(value)}")
        elif isinstance(data, torch.Tensor):
            logger.info(f"    形状: {data.shape} {data.dtype}")
    
    return instance_data



def save_rigid_instance(
    trainer, 
    instance_id: int, 
    save_path: str,
    save_metadata: bool = True
) -> Dict[str, Any]:
    """
    保存单个Rigid实例的所有数据到文件
    
    Args:
        trainer: 训练器实例
        instance_id: 要保存的实例ID
        save_path: 保存路径
        save_metadata: 是否保存元数据信息
        
    Returns:
        保存的数据字典
    """
    logger.info(f"=== 开始保存Rigid实例 {instance_id} ===")
    
    if "RigidNodes" not in trainer.models:
        raise ValueError("模型中没有找到RigidNodes")
    
    rigid_nodes = trainer.models["RigidNodes"]
    
    # 检查实例ID是否存在
    all_ids = rigid_nodes.point_ids[..., 0].unique().cpu().numpy()
    if instance_id not in all_ids:
        raise ValueError(f"实例ID {instance_id} 不存在，可用ID: {all_ids}")
    
    # 获取该实例的点掩码
    pts_mask = rigid_nodes.point_ids[..., 0] == instance_id
    
    logger.info(f"实例 {instance_id} 的点数: {pts_mask.sum().item()}")
    
    # 收集实例数据
    instance_data = {}
    
    with torch.no_grad():
        # 1. 基础几何数据
        instance_data["geometry"] = {
            "_means": rigid_nodes._means[pts_mask].detach().cpu(),
            "_scales": rigid_nodes._scales[pts_mask].detach().cpu(),
            "_quats": rigid_nodes._quats[pts_mask].detach().cpu(),
            "_features_dc": rigid_nodes._features_dc[pts_mask].detach().cpu(),
            "_features_rest": rigid_nodes._features_rest[pts_mask].detach().cpu(),
            "_opacities": rigid_nodes._opacities[pts_mask].detach().cpu(),
        }
        
        # 2. 姿态和变换数据
        instance_data["motion"] = {
            "instances_quats": rigid_nodes.instances_quats[:, instance_id].detach().cpu(),
            "instances_trans": rigid_nodes.instances_trans[:, instance_id].detach().cpu(),
            "instances_fv": rigid_nodes.instances_fv[:, instance_id].detach().cpu(),
        }
        
        # 3. 实例大小信息
        if hasattr(rigid_nodes, "instances_size"):
            instance_data["size"] = rigid_nodes.instances_size[instance_id].detach().cpu()
        
        # 4. 元数据
        if save_metadata:
            instance_data["metadata"] = {
                "instance_id": instance_id,
                "num_points": pts_mask.sum().item(),
                "num_frames": rigid_nodes.instances_quats.shape[0],
                "sh_degree": rigid_nodes.sh_degree,
                "save_timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
                "device": str(rigid_nodes._means.device),
            }

       # ========== 新增：单独保存轨迹数据 ==========
        # 提取轨迹数据
        frame_count = rigid_nodes.instances_quats.shape[0]
        frame_indices = list(range(frame_count))
        obj_to_world_matrices = []
        
        # 计算每一帧的obj_to_world变换矩阵
        for frame_idx in range(frame_count):
            # 获取该帧的位置和旋转
            translation = rigid_nodes.instances_trans[frame_idx, instance_id].detach().clone().cpu().numpy()
            rotation_quat = rigid_nodes.instances_quats[frame_idx, instance_id].detach().clone().cpu().numpy()
            
            # 四元数转旋转矩阵 (假设四元数格式为 [w, x, y, z])
            def quat_to_rotation_matrix(quat):
                """将四元数转换为3x3旋转矩阵"""
                if len(quat) == 4:
                    w, x, y, z = quat
                    # 归一化四元数
                    norm = np.sqrt(w*w + x*x + y*y + z*z)
                    if norm > 1e-8:
                        w, x, y, z = w/norm, x/norm, y/norm, z/norm
                    
                    # 计算旋转矩阵
                    return np.array([
                        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
                    ])
                else:
                    # 如果不是4元素，返回单位矩阵
                    return np.eye(3)
            
            # 构建4x4变换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = quat_to_rotation_matrix(rotation_quat)
            transform_matrix[:3, 3] = translation
            
            # 转换为列表格式，保持与目标格式一致
            matrix_list = transform_matrix.tolist()
            obj_to_world_matrices.append(matrix_list)
        
        # 构建轨迹数据字典，格式与目标JSON完全一致
        trajectory_data = {
            str(instance_id): {
                "id": instance_id,
                "class_name": "Vehicle",  # 或者根据实际情况设置 (Car, Truck, Bus等)
                "frame_annotations": {
                    "frame_idx": frame_indices,
                    "obj_to_world": obj_to_world_matrices
                }
            }
        }
        
        # 保存轨迹数据到单独的JSON文件
        trajectory_save_path = Path(save_path).parent / f"trajectory_{Path(save_path).stem}.json"
        with open(trajectory_save_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"轨迹数据已单独保存到: {trajectory_save_path}")
        logger.info(f"轨迹包含 {len(frame_indices)} 帧数据")

    # 保存到文件
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用pickle保存，保持tensor格式
    with open(save_path, 'wb') as f:
        pickle.dump(instance_data, f)
    
    logger.info(f"Rigid实例 {instance_id} 已保存到: {save_path}")
    logger.info(f"保存的数据大小: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 打印保存的数据统计
    for category, data in instance_data.items():
        if category == "metadata":
            continue
        logger.info(f"  {category} 包含 {len(data)} 个属性")
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"    {key}: {value.shape} {value.dtype}")
            else:
                logger.info(f"    {key}: {type(value)}")
    
    return instance_data


def load_instance_data(file_path: str) -> Dict[str, Any]:
    """
    从文件加载实例数据
    
    Args:
        file_path: 实例数据文件路径
        
    Returns:
        加载的实例数据字典
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"实例文件不存在: {file_path}")
    
    logger.info(f"正在加载实例数据: {file_path}")
    
    with open(file_path, 'rb') as f:
        instance_data = pickle.load(f)
    
    # 打印加载的数据信息
    if "metadata" in instance_data:
        metadata = instance_data["metadata"]
        logger.info(f"加载的实例信息:")
        logger.info(f"  原始实例ID: {metadata.get('instance_id', 'Unknown')}")
        logger.info(f"  点数: {metadata.get('num_points', 'Unknown')}")
        logger.info(f"  帧数: {metadata.get('num_frames', 'Unknown')}")
        logger.info(f"  保存时间: {metadata.get('save_timestamp', 'Unknown')}")
    
    return instance_data


def insert_smpl_instance(
    trainer,
    instance_data: Dict[str, Any],
    new_instance_id: Optional[int] = None,
    device: str = "cuda"
) -> int:
    """
    将保存的SMPL实例数据插入到场景中
    
    Args:
        trainer: 训练器实例
        instance_data: 实例数据字典
        new_instance_id: 新的实例ID，如果为None则自动分配
        device: 设备
        
    Returns:
        分配的新实例ID
    """
    logger.info("=== 开始插入SMPL实例 ===")
    
    if "SMPLNodes" not in trainer.models:
        raise ValueError("模型中没有找到SMPLNodes")
    
    smpl_nodes = trainer.models["SMPLNodes"]
    
    # 检查是否需要初始化SMPLNodes
    is_empty = (not hasattr(smpl_nodes, 'point_ids') or 
                smpl_nodes.point_ids is None or 
                smpl_nodes.point_ids.numel() == 0)
    
    if is_empty:
        logger.info("SMPLNodes为空，开始初始化...")
        initialize_empty_smpl_nodes(smpl_nodes, instance_data, device)
        return 0  # 第一个实例总是ID 0
    
    # 获取当前所有实例ID并分配新ID
    try:
        current_ids = smpl_nodes.point_ids[..., 0].unique().cpu().numpy()
    except Exception as e:
        logger.warning(f"无法获取现有实例ID: {e}")
        current_ids = []
        
    if new_instance_id is None:
        new_instance_id = int(max(current_ids) + 1) if len(current_ids) > 0 else 0
    elif new_instance_id in current_ids:
        logger.warning(f"实例ID {new_instance_id} 已存在，将覆盖现有实例")
    
    logger.info(f"分配新实例ID: {new_instance_id}")
    logger.info(f"当前存在的实例ID: {current_ids}")
    
    # 获取实例数据
    geometry = instance_data["geometry"]
    motion = instance_data["motion"]
    smpl_template = instance_data.get("smpl_template", {})
    voxel_deformer_data = instance_data.get("voxel_deformer", {})
    control_config = instance_data.get("control_config", {})
    constraints = instance_data.get("constraints", {})
    metadata = instance_data.get("metadata", {})
    
    num_points = geometry["_means"].shape[0]
    num_frames = motion["instances_quats"].shape[0]
    
    logger.info(f"插入的实例信息: 点数={num_points}, 帧数={num_frames}")
    
    with torch.no_grad():
        # 将数据移动到指定设备
        for key, value in geometry.items():
            if isinstance(value, torch.Tensor):
                geometry[key] = value.to(device)
        
        for key, value in motion.items():
            if isinstance(value, torch.Tensor):
                motion[key] = value.to(device)
        
        # 1. 扩展几何体数据
        smpl_nodes._means = Parameter(
            torch.cat([smpl_nodes._means, geometry["_means"]], dim=0)
        )
        smpl_nodes._scales = Parameter(
            torch.cat([smpl_nodes._scales, geometry["_scales"]], dim=0)
        )
        smpl_nodes._quats = Parameter(
            torch.cat([smpl_nodes._quats, geometry["_quats"]], dim=0)
        )
        smpl_nodes._features_dc = Parameter(
            torch.cat([smpl_nodes._features_dc, geometry["_features_dc"]], dim=0)
        )
        smpl_nodes._features_rest = Parameter(
            torch.cat([smpl_nodes._features_rest, geometry["_features_rest"]], dim=0)
        )
        smpl_nodes._opacities = Parameter(
            torch.cat([smpl_nodes._opacities, geometry["_opacities"]], dim=0)
        )
        
        # 2. 创建新的point_ids
        new_point_ids = torch.full(
            (num_points, 1), new_instance_id, 
            device=device, dtype=torch.long
        )
        smpl_nodes.point_ids = torch.cat([smpl_nodes.point_ids, new_point_ids], dim=0)
        
        # 3. 处理约束数据（如果存在）
        if "on_mesh_x" in constraints and hasattr(smpl_nodes, "on_mesh_x"):
            constraint_data = constraints["on_mesh_x"].to(device)
            smpl_nodes.on_mesh_x = torch.cat([smpl_nodes.on_mesh_x, constraint_data], dim=0)
            logger.info(f"恢复了约束数据: {constraint_data.shape}")
        
        # 4. 扩展运动数据 - 需要匹配当前场景的帧数
        current_num_frames = smpl_nodes.instances_quats.shape[0]
        current_num_instances = smpl_nodes.instances_quats.shape[1]
        
        logger.info(f"当前场景状态: {current_num_frames}帧, {current_num_instances}实例")
        
        # 调整加载的运动数据以匹配当前场景帧数
        if num_frames != current_num_frames:
            logger.info(f"调整运动数据帧数: {num_frames} -> {current_num_frames}")
            motion_adjusted = {}
            for key, value in motion.items():
                if isinstance(value, torch.Tensor):
                    motion_adjusted[key] = match_sequence_length(current_num_frames, value)
                else:
                    motion_adjusted[key] = value
            motion = motion_adjusted
        
        # 扩展实例相关的张量
        new_instances_quats = torch.zeros(
            current_num_frames, current_num_instances + 1, 
            *smpl_nodes.instances_quats.shape[2:], device=device
        )
        new_instances_quats[:, :current_num_instances] = smpl_nodes.instances_quats
        new_instances_quats[:, current_num_instances] = motion["instances_quats"]
        smpl_nodes.instances_quats = Parameter(new_instances_quats)
        
        new_instances_trans = torch.zeros(
            current_num_frames, current_num_instances + 1, 3, device=device
        )
        new_instances_trans[:, :current_num_instances] = smpl_nodes.instances_trans
        new_instances_trans[:, current_num_instances] = motion["instances_trans"]
        smpl_nodes.instances_trans = Parameter(new_instances_trans)
        
        new_smpl_qauts = torch.zeros(
            current_num_frames, current_num_instances + 1, 23, 4, device=device
        )
        new_smpl_qauts[:, :current_num_instances] = smpl_nodes.smpl_qauts
        new_smpl_qauts[:, current_num_instances] = motion["smpl_qauts"]
        smpl_nodes.smpl_qauts = Parameter(new_smpl_qauts)
        
        new_instances_fv = torch.zeros(
            current_num_frames, current_num_instances + 1, device=device, dtype=torch.bool
        )
        new_instances_fv[:, :current_num_instances] = smpl_nodes.instances_fv
        new_instances_fv[:, current_num_instances] = motion["instances_fv"]
        smpl_nodes.instances_fv = new_instances_fv
        
        # 扩展实例大小信息
        if "instances_size" in motion:
            new_instances_size = torch.zeros(
                current_num_instances + 1, 3, device=device
            )
            new_instances_size[:current_num_instances] = smpl_nodes.instances_size
            new_instances_size[current_num_instances] = motion["instances_size"]
            smpl_nodes.instances_size = new_instances_size
            logger.info(f"恢复了实例大小信息")
        
        # 5. 全面更新SMPL模板相关的所有张量
        if hasattr(smpl_nodes, 'template'):
            template = smpl_nodes.template
            new_num_instances = current_num_instances + 1
            
            logger.info("开始扩展SMPL模板张量...")
            
            # 扩展所有可能的模板属性，优先使用保存的数据
            template_attrs_mapping = {
                'init_beta': 'betas',      # 形状参数
                'J_canonical': 'J_canonical',    # 关节位置
                'W': 'W',                 # LBS权重
                'A0_inv': 'A0_inv',       # 逆变换矩阵
            }
            
            for attr_name, saved_key in template_attrs_mapping.items():
                if hasattr(template, attr_name):
                    old_attr = getattr(template, attr_name)
                    if isinstance(old_attr, torch.Tensor) and old_attr.dim() >= 1:
                        # 确保第一个维度是实例数量
                        if old_attr.shape[0] == current_num_instances:
                            attr_shape = old_attr.shape[1:]
                            new_attr = torch.zeros(new_num_instances, *attr_shape, device=device, dtype=old_attr.dtype)
                            new_attr[:current_num_instances] = old_attr
                            
                            # 优先使用保存的数据
                            if saved_key in smpl_template:
                                saved_data = smpl_template[saved_key].to(device)
                                if saved_data.shape == attr_shape:
                                    new_attr[current_num_instances] = saved_data
                                    logger.info(f"  从保存数据恢复 {attr_name}")
                                else:
                                    logger.warning(f"  保存的 {saved_key} 形状不匹配: {saved_data.shape} vs {attr_shape}")
                                    # 使用默认值
                                    if attr_name == 'init_beta':
                                        new_attr[current_num_instances] = torch.zeros_like(old_attr[0])
                                    elif attr_name == 'A0_inv':
                                        new_attr[current_num_instances] = torch.eye(4, device=device).unsqueeze(0).repeat(24, 1, 1)
                                    else:
                                        new_attr[current_num_instances] = old_attr[0].clone()
                            else:
                                # 没有保存数据时使用默认值
                                if current_num_instances > 0:
                                    if attr_name == 'init_beta':
                                        new_attr[current_num_instances] = torch.zeros_like(old_attr[0])
                                    elif attr_name == 'A0_inv':
                                        new_attr[current_num_instances] = torch.eye(4, device=device).unsqueeze(0).repeat(24, 1, 1)
                                    else:
                                        new_attr[current_num_instances] = old_attr[0].clone()
                                logger.info(f"  使用默认值初始化 {attr_name}")
                            
                            setattr(template, attr_name, new_attr)
                            logger.info(f"  扩展 {attr_name}: {old_attr.shape} -> {new_attr.shape}")
                        else:
                            logger.warning(f"  跳过 {attr_name}: 第一维度不匹配 ({old_attr.shape[0]} != {current_num_instances})")
            
            # 处理其他模板属性
            other_template_attrs = ['canonical_pose', 'parents']
            for attr_name in other_template_attrs:
                if attr_name in smpl_template:
                    saved_data = smpl_template[attr_name].to(device)
                    if hasattr(template, attr_name):
                        # 如果模板已有该属性，检查是否需要更新
                        existing_attr = getattr(template, attr_name)
                        if not torch.equal(existing_attr, saved_data):
                            logger.info(f"  更新模板属性 {attr_name}")
                            setattr(template, attr_name, saved_data)
                    else:
                        # 如果模板没有该属性，直接设置
                        logger.info(f"  添加模板属性 {attr_name}")
                        setattr(template, attr_name, saved_data)
            
            # 检查并扩展_template_layer相关的属性（如果存在）
            if hasattr(template, '_template_layer'):
                template_layer = template._template_layer
                
                # 检查template_layer的所有buffer和parameter
                for name, param in template_layer.named_parameters():
                    if isinstance(param, torch.Tensor) and param.dim() >= 2:
                        if param.shape[0] == current_num_instances:
                            param_shape = param.shape[1:]
                            new_param = torch.zeros(new_num_instances, *param_shape, device=device, dtype=param.dtype)
                            new_param[:current_num_instances] = param.data
                            if current_num_instances > 0:
                                new_param[current_num_instances] = param.data[0].clone()
                            param.data = new_param
                            logger.info(f"  扩展template_layer参数 {name}: -> {new_param.shape}")
                
                for name, buffer in template_layer.named_buffers():
                    if isinstance(buffer, torch.Tensor) and buffer.dim() >= 2:
                        if buffer.shape[0] == current_num_instances:
                            buffer_shape = buffer.shape[1:]
                            new_buffer = torch.zeros(new_num_instances, *buffer_shape, device=device, dtype=buffer.dtype)
                            new_buffer[:current_num_instances] = buffer
                            if current_num_instances > 0:
                                new_buffer[current_num_instances] = buffer[0].clone()
                            template_layer.register_buffer(name, new_buffer)
                            logger.info(f"  扩展template_layer缓冲区 {name}: -> {new_buffer.shape}")
            
            # 处理体素变形器
            if hasattr(template, 'voxel_deformer') and smpl_nodes.use_voxel_deformer:
                voxel_deformer = template.voxel_deformer
                logger.info("扩展体素变形器张量...")
                
                voxel_attrs_mapping = {
                    'lbs_voxel_base': 'lbs_voxel_base',
                    'offset': 'offset',
                    'scale': 'scale'
                }
                
                for attr_name, saved_key in voxel_attrs_mapping.items():
                    if hasattr(voxel_deformer, attr_name):
                        old_attr = getattr(voxel_deformer, attr_name)
                        if isinstance(old_attr, torch.Tensor) and old_attr.dim() >= 2:
                            if old_attr.shape[0] == current_num_instances:
                                attr_shape = old_attr.shape[1:]
                                new_attr = torch.zeros(new_num_instances, *attr_shape, device=device, dtype=old_attr.dtype)
                                new_attr[:current_num_instances] = old_attr
                                
                                # 优先使用保存的体素变形器数据
                                if saved_key in voxel_deformer_data:
                                    saved_voxel_data = voxel_deformer_data[saved_key].to(device)
                                    if saved_voxel_data.shape == attr_shape:
                                        new_attr[current_num_instances] = saved_voxel_data
                                        logger.info(f"  从保存数据恢复体素变形器 {attr_name}")
                                    else:
                                        new_attr[current_num_instances] = old_attr[0].clone()
                                        logger.warning(f"  体素变形器 {saved_key} 形状不匹配，使用默认值")
                                else:
                                    if current_num_instances > 0:
                                        new_attr[current_num_instances] = old_attr[0].clone()
                                    logger.info(f"  使用默认值初始化体素变形器 {attr_name}")
                                
                                setattr(voxel_deformer, attr_name, new_attr)
                                logger.info(f"  扩展体素变形器 {attr_name}: {old_attr.shape} -> {new_attr.shape}")
                
                # 处理可训练的体素校正参数
                if hasattr(voxel_deformer, 'voxel_w_correction'):
                    old_correction = voxel_deformer.voxel_w_correction
                    if old_correction.shape[0] == current_num_instances:
                        correction_shape = old_correction.shape[1:]
                        new_correction = torch.zeros(new_num_instances, *correction_shape, device=device, dtype=old_correction.dtype)
                        new_correction[:current_num_instances] = old_correction.data
                        
                        # 如果有保存的校正数据
                        if "voxel_w_correction" in voxel_deformer_data:
                            saved_correction = voxel_deformer_data["voxel_w_correction"].to(device)
                            if saved_correction.shape == correction_shape:
                                new_correction[current_num_instances] = saved_correction
                                logger.info(f"  从保存数据恢复体素校正参数")
                            else:
                                logger.warning(f"  体素校正参数形状不匹配，使用零初始化")
                        # 否则新实例的校正参数初始化为零（已经是零了）
                        
                        voxel_deformer.voxel_w_correction = Parameter(new_correction)
                        logger.info(f"  扩展体素校正参数: {old_correction.shape} -> {new_correction.shape}")
                
                # 恢复其他体素变形器的全局参数
                global_voxel_attrs = ['global_scale', 'ratio', 'resolution_dhw', 'bbox', 'grid_denorm']
                for attr_name in global_voxel_attrs:
                    if attr_name in voxel_deformer_data and hasattr(voxel_deformer, attr_name):
                        saved_data = voxel_deformer_data[attr_name]
                        if isinstance(saved_data, torch.Tensor):
                            saved_data = saved_data.to(device)
                        existing_attr = getattr(voxel_deformer, attr_name)
                        
                        # 安全地检查是否需要更新
                        should_update = False
                        if isinstance(existing_attr, torch.Tensor) and isinstance(saved_data, torch.Tensor):
                            # 确保两个张量在同一设备上进行比较
                            existing_attr_on_device = existing_attr.to(device)
                            try:
                                if not torch.equal(existing_attr_on_device, saved_data):
                                    should_update = True
                            except RuntimeError as e:
                                # 如果形状不匹配或其他错误，也认为需要更新
                                logger.warning(f"  比较 {attr_name} 时出错: {e}，将进行更新")
                                should_update = True
                        elif isinstance(existing_attr, (list, tuple)) and isinstance(saved_data, (list, tuple)):
                            if existing_attr != saved_data:
                                should_update = True
                        elif existing_attr != saved_data:
                            should_update = True
                        
                        if should_update:
                            logger.info(f"  更新体素变形器全局参数 {attr_name}")
                            if isinstance(saved_data, torch.Tensor):
                                setattr(voxel_deformer, attr_name, saved_data)
                            else:
                                setattr(voxel_deformer, attr_name, saved_data)
 
        
        # 6. 恢复KNN数据（如果存在）
        if "knn_data" in instance_data and hasattr(smpl_nodes, "nn_ind"):
            knn_data = instance_data["knn_data"]
            if "nn_ind" in knn_data:
                old_nn_ind = smpl_nodes.nn_ind
                nn_ind_shape = old_nn_ind.shape[1:]
                new_nn_ind = torch.zeros(new_num_instances, *nn_ind_shape, device=device, dtype=old_nn_ind.dtype)
                new_nn_ind[:current_num_instances] = old_nn_ind
                new_nn_ind[current_num_instances] = knn_data["nn_ind"].to(device)
                smpl_nodes.nn_ind = new_nn_ind
                logger.info(f"恢复了KNN数据")
        
        logger.info("所有模板张量扩展完成")
    
    logger.info(f"SMPL实例插入完成，新实例ID: {new_instance_id}")
    logger.info(f"更新后的模型状态:")
    logger.info(f"  总点数: {smpl_nodes._means.shape[0]}")
    logger.info(f"  总实例数: {smpl_nodes.instances_quats.shape[1]}")
    logger.info(f"  帧数: {smpl_nodes.instances_quats.shape[0]}")
    
    # 验证关键张量的形状
    if hasattr(smpl_nodes, 'template'):
        template = smpl_nodes.template
        expected_instances = smpl_nodes.instances_quats.shape[1]
        
        verification_attrs = ['init_beta', 'J_canonical', 'W', 'A0_inv']
        for attr_name in verification_attrs:
            if hasattr(template, attr_name):
                attr = getattr(template, attr_name)
                if isinstance(attr, torch.Tensor):
                    actual_instances = attr.shape[0] if attr.dim() > 0 else 0
                    status = "✓" if actual_instances == expected_instances else "✗"
                    logger.info(f"  {status} {attr_name}: {attr.shape} (期望实例数: {expected_instances})")
                    if actual_instances != expected_instances:
                        logger.error(f"实例数不匹配: {attr_name} 有 {actual_instances} 个实例，期望 {expected_instances} 个")
    
    return new_instance_id


def initialize_empty_smpl_nodes(smpl_nodes, instance_data, device):
    """
    完整修复的SMPLNodes初始化函数 - 处理CUDA张量转换问题
    
    关键修复：
    1. 处理get_on_mesh_init_geo_values的CUDA转换问题
    2. 提供备用方案使用保存的几何参数
    3. 严格按照原始训练代码逻辑
    """
    from torch.nn import Parameter
    from models.human_body import SMPLTemplate, get_on_mesh_init_geo_values
    from models.gaussians.basics import RGB2SH, num_sh_bases
    
    logger.info("正确初始化空的SMPLNodes...")
    
    # 获取实例数据
    geometry = instance_data["geometry"]
    motion = instance_data["motion"]
    smpl_template = instance_data.get("smpl_template", {})
    voxel_deformer_data = instance_data.get("voxel_deformer", {})
    constraints = instance_data.get("constraints", {})
    
    # 移动数据到设备
    for key, value in geometry.items():
        if isinstance(value, torch.Tensor):
            geometry[key] = value.to(device)
    
    for key, value in motion.items():
        if isinstance(value, torch.Tensor):
            motion[key] = value.to(device)
    
    num_points = geometry["_means"].shape[0]
    num_frames = motion["instances_quats"].shape[0]
    
    # 验证点数是否符合SMPL规范
    if num_points != smpl_nodes.smpl_points_num:
        logger.warning(f"点数不匹配: 期望={smpl_nodes.smpl_points_num}, 实际={num_points}")
    
    logger.info(f"初始化参数: 点数={num_points}, 帧数={num_frames}")
    
    with torch.no_grad():
        # 第一步：初始化SMPL模板 - 必须在生成几何参数之前完成
        if "betas" in smpl_template:
            init_beta = smpl_template["betas"].to(device).view(1, -1)  # [1, 10]
        else:
            init_beta = torch.zeros(1, 10, device=device)
            logger.warning("未找到保存的betas，使用零初始化")
        
        logger.info("创建SMPL模板...")
        smpl_nodes.template = SMPLTemplate(
            smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl",
            num_human=1,  # 只有一个实例
            init_beta=init_beta,
            cano_pose_type="da_pose",
            use_voxel_deformer=smpl_nodes.use_voxel_deformer if hasattr(smpl_nodes, 'use_voxel_deformer') else False,
            is_resume=False,  # 注意：这里是False，不是True
        ).to(device)
        
        # 启用体素变形器
        if hasattr(smpl_nodes, 'use_voxel_deformer') and smpl_nodes.use_voxel_deformer:
            smpl_nodes.template.voxel_deformer.enable_voxel_correction()
            logger.info("启用了体素变形器")
        
        # 第二步：使用模板生成正确的几何参数 - 处理CUDA转换问题
        opacity_init_value = torch.tensor(smpl_nodes.ctrl_cfg.opacity_init_value if hasattr(smpl_nodes, 'ctrl_cfg') else 0.1)
        logger.info(f"使用opacity_init_value: {opacity_init_value}")
        
        use_generated_geometry = False
        try:
            # 尝试生成几何参数
            x, q, s, o = get_on_mesh_init_geo_values(
                smpl_nodes.template,
                opacity_init_logit=torch.logit(opacity_init_value),
            )
            use_generated_geometry = True
            logger.info("成功生成几何参数")
        except Exception as e:
            logger.error(f"get_on_mesh_init_geo_values失败: {e}")
            logger.info("使用保存的几何参数作为备用方案...")
            x = geometry["_means"]
            q = geometry["_quats"]
            s = geometry["_scales"]
            o = geometry["_opacities"]
            use_generated_geometry = False
        
        # 处理球形高斯
        if hasattr(smpl_nodes, 'ball_gaussians') and smpl_nodes.ball_gaussians:
            s = s.mean(-1, keepdim=True)
        
        # 确保数据类型和设备
        x = x.to(dtype=torch.float32, device=device)
        s = s.to(dtype=torch.float32, device=device)
        q = q.to(dtype=torch.float32, device=device)
        o = o.to(dtype=torch.float32, device=device)
        
        logger.info(f"几何参数形状: x={x.shape}, s={s.shape}, q={q.shape}, o={o.shape}")
        
        # 第三步：初始化运动参数 - 注意形状
        instances_quats = motion["instances_quats"]
        if instances_quats.dim() == 2:
            instances_quats = instances_quats.unsqueeze(1)  # [num_frames, 1, 4]
        elif instances_quats.dim() == 3 and instances_quats.shape[2] == 1:
            instances_quats = instances_quats.squeeze(2)  # [num_frames, 1, 4]
        
        instances_trans = motion["instances_trans"]
        if instances_trans.dim() == 2:
            instances_trans = instances_trans.unsqueeze(1)  # [num_frames, 1, 3]
        
        smpl_qauts = motion["smpl_qauts"]
        if smpl_qauts.dim() == 3:
            smpl_qauts = smpl_qauts.unsqueeze(1)  # [num_frames, 1, 23, 4]
        
        instances_fv = motion["instances_fv"]
        if instances_fv.dim() == 1:
            instances_fv = instances_fv.unsqueeze(1)  # [num_frames, 1]
        
        # 第四步：处理translation的局部化 - 只在使用生成几何参数时执行
        if use_generated_geometry:
            logger.info("处理translation的局部化...")
            
            # 设置临时的instances_fv用于计算
            smpl_nodes.instances_fv = instances_fv
            
            # 使用模板计算变形
            for fi in range(num_frames):
                instance_mask = instances_fv[fi]
                if instance_mask.sum() == 0:
                    continue
                    
                # 组合全局四元数和关节四元数
                theta = torch.cat((instances_quats[fi].unsqueeze(1), smpl_qauts[fi]), dim=1)
                masked_theta = theta[instance_mask]
                masked_theta = masked_theta / masked_theta.norm(dim=-1, keepdim=True)
                
                # 使用模板计算变形
                W, A = smpl_nodes.template(
                    masked_theta=masked_theta, instances_mask=instance_mask
                )
                T = torch.einsum("bnj, bjrc -> bnrc", W, A)
                R = T[:, :, :3, :3]  # [N, 3, 3]
                t = T[:, :, :3, 3]  # [N, 3]
                
                # 计算变形后的平均位置
                reshaped_means = x.reshape(1, smpl_nodes.smpl_points_num, 3)
                deformed_means = (
                    torch.einsum("bnij,bnj->bni", R, reshaped_means[instance_mask]) + t
                )
                bbox_min = deformed_means.min(dim=1)[0]
                bbox_max = deformed_means.max(dim=1)[0]
                local_shift = (bbox_min + bbox_max) / 2
                
                # 调整translation
                instances_trans[fi, instance_mask] = (
                    instances_trans[fi, instance_mask] - local_shift
                )
        else:
            logger.info("跳过translation局部化（使用保存的几何参数）")
        
        # 第五步：设置所有参数
        smpl_nodes._means = Parameter(x, requires_grad=not (hasattr(smpl_nodes, 'ctrl_cfg') and smpl_nodes.ctrl_cfg.get('freeze_x', False)))
        smpl_nodes._scales = Parameter(s, requires_grad=not (hasattr(smpl_nodes, 'ctrl_cfg') and smpl_nodes.ctrl_cfg.get('freeze_s', False)))
        smpl_nodes._quats = Parameter(q, requires_grad=not (hasattr(smpl_nodes, 'ctrl_cfg') and smpl_nodes.ctrl_cfg.get('freeze_q', False)))
        smpl_nodes._opacities = Parameter(o, requires_grad=not (hasattr(smpl_nodes, 'ctrl_cfg') and smpl_nodes.ctrl_cfg.get('freeze_o', False)))
        
        # 运动参数
        smpl_nodes.instances_quats = Parameter(instances_quats.unsqueeze(2))  # [num_frames, 1, 1, 4]
        smpl_nodes.instances_trans = Parameter(instances_trans)  # [num_frames, 1, 3]
        smpl_nodes.smpl_qauts = Parameter(smpl_qauts)  # [num_frames, 1, 23, 4]
        smpl_nodes.instances_fv = instances_fv  # [num_frames, 1]
        
        # 设置实例大小
        if "instances_size" in instance_data:
            size_data = instance_data["instances_size"].to(device)
            smpl_nodes.instances_size = size_data.unsqueeze(0) if size_data.dim() == 1 else size_data
        elif "instances_size" in motion:
            size_data = motion["instances_size"].to(device)
            smpl_nodes.instances_size = size_data.unsqueeze(0) if size_data.dim() == 1 else size_data
        else:
            # 计算默认大小
            smpl_nodes.instances_size = torch.tensor([[1.0, 1.0, 1.0]], device=device)
        
        # 设置point_ids
        smpl_nodes.point_ids = torch.full((num_points, 1), 0, device=device, dtype=torch.long)
        
        # 第六步：正确初始化颜色 - 遵循原始逻辑
        logger.info("初始化颜色...")
        dim_sh = num_sh_bases(smpl_nodes.sh_degree)
        
        # 根据是否使用生成几何参数决定颜色初始化方式
        if use_generated_geometry:
            # 使用随机颜色（原始逻辑）
            init_colors = torch.rand((num_points, 3), device=device)
        else:
            # 使用保存的颜色特征
            try:
                # 从保存的features_dc恢复颜色
                features_dc = geometry["_features_dc"]
                if features_dc.shape[1] == 3:  # 如果是RGB格式
                    init_colors = features_dc
                else:
                    # 如果是SH格式，使用随机颜色
                    init_colors = torch.rand((num_points, 3), device=device)
            except:
                init_colors = torch.rand((num_points, 3), device=device)
        
        fused_color = RGB2SH(init_colors)
        
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(device)
        if smpl_nodes.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
            
        smpl_nodes._features_dc = Parameter(shs[:, 0, :], requires_grad=not (hasattr(smpl_nodes, 'ctrl_cfg') and smpl_nodes.ctrl_cfg.get('freeze_dc', False)))
        smpl_nodes._features_rest = Parameter(shs[:, 1:, :], requires_grad=not (hasattr(smpl_nodes, 'ctrl_cfg') and smpl_nodes.ctrl_cfg.get('freeze_rest', False)))
        
        # 第七步：初始化KNN - 重要！
        if hasattr(smpl_nodes, 'update_knn'):
            try:
                smpl_nodes.update_knn(x)
                logger.info("完成KNN初始化")
            except Exception as e:
                logger.warning(f"KNN初始化失败: {e}")
        
        # 第八步：设置约束
        if hasattr(smpl_nodes, 'ctrl_cfg') and smpl_nodes.ctrl_cfg.get('constrain_xyz_offset', False):
            if use_generated_geometry:
                smpl_nodes.on_mesh_x = x.clone()
                logger.info("设置了on_mesh_x约束（使用生成的几何参数）")
            elif "on_mesh_x" in constraints:
                smpl_nodes.on_mesh_x = constraints["on_mesh_x"].to(device)
                logger.info("设置了on_mesh_x约束（使用保存的约束数据）")
            else:
                smpl_nodes.on_mesh_x = x.clone()
                logger.info("设置了on_mesh_x约束（使用当前几何参数）")
        
        # 第九步：恢复保存的模板数据
        if smpl_template:
            template = smpl_nodes.template
            for attr_name, saved_key in [
                ('J_canonical', 'J_canonical'),
                ('W', 'W'),
                ('A0_inv', 'A0_inv')
            ]:
                if saved_key in smpl_template and hasattr(template, attr_name):
                    try:
                        saved_data = smpl_template[saved_key].to(device)
                        current_attr = getattr(template, attr_name)
                        if current_attr is not None and current_attr.shape[0] >= 1:
                            current_attr[0] = saved_data
                            logger.info(f"恢复模板参数: {attr_name}")
                    except Exception as e:
                        logger.warning(f"恢复模板参数 {attr_name} 失败: {e}")
        
        # 第十步：恢复体素变形器数据
        if voxel_deformer_data and hasattr(smpl_nodes.template, 'voxel_deformer'):
            voxel_deformer = smpl_nodes.template.voxel_deformer
            for attr_name in ['lbs_voxel_base', 'offset', 'scale']:
                if attr_name in voxel_deformer_data and hasattr(voxel_deformer, attr_name):
                    try:
                        saved_data = voxel_deformer_data[attr_name].to(device)
                        current_attr = getattr(voxel_deformer, attr_name)
                        if current_attr is not None and current_attr.shape[0] >= 1:
                            current_attr[0] = saved_data
                            logger.info(f"恢复体素变形器参数: {attr_name}")
                    except Exception as e:
                        logger.warning(f"恢复体素变形器参数 {attr_name} 失败: {e}")
        
        # 验证最终形状
        expected_shapes = {
            "_means": (num_points, 3),
            "instances_quats": (num_frames, 1, 1, 4),
            "instances_trans": (num_frames, 1, 3),
            "smpl_qauts": (num_frames, 1, 23, 4),
            "instances_fv": (num_frames, 1),
            "instances_size": (1, 3),
            "point_ids": (num_points, 1)
        }
        
        logger.info("验证最终张量形状:")
        all_correct = True
        for attr_name, expected_shape in expected_shapes.items():
            if hasattr(smpl_nodes, attr_name):
                actual_tensor = getattr(smpl_nodes, attr_name)
                actual_shape = actual_tensor.shape
                status = "✓" if actual_shape == expected_shape else "✗"
                logger.info(f"  {status} {attr_name}: {actual_shape} (期望: {expected_shape})")
                if actual_shape != expected_shape:
                    all_correct = False
            else:
                logger.warning(f"  ? {attr_name}: 属性不存在")
                all_correct = False
        
        if not all_correct:
            logger.error("存在形状不匹配的张量，可能影响渲染")
        
        # 验证模板
        if hasattr(smpl_nodes, 'template'):
            template = smpl_nodes.template
            logger.info("验证模板属性:")
            for attr_name in ['init_beta', 'J_canonical', 'W']:
                if hasattr(template, attr_name):
                    attr = getattr(template, attr_name)
                    if isinstance(attr, torch.Tensor):
                        logger.info(f"  ✓ {attr_name}: {attr.shape}")
                    else:
                        logger.info(f"  ? {attr_name}: {type(attr)}")
                else:
                    logger.warning(f"  ✗ {attr_name}: 不存在")
        
        # 验证设备一致性
        logger.info("验证设备一致性:")
        for attr_name in ['_means', 'instances_quats', 'instances_trans', 'smpl_qauts']:
            if hasattr(smpl_nodes, attr_name):
                attr = getattr(smpl_nodes, attr_name)
                if isinstance(attr, torch.Tensor):
                    logger.info(f"  {attr_name}: {attr.device}")
                elif hasattr(attr, 'device'):
                    logger.info(f"  {attr_name}: {attr.device}")
    
    logger.info("SMPLNodes初始化完成:")
    logger.info(f"  点数: {num_points}")
    logger.info(f"  帧数: {num_frames}")
    logger.info(f"  实例数: 1")
    logger.info(f"  实例ID: 0")
    logger.info(f"  模板已初始化: {hasattr(smpl_nodes, 'template')}")
    logger.info(f"  约束数据: {hasattr(smpl_nodes, 'on_mesh_x')}")
    logger.info(f"  KNN已初始化: {hasattr(smpl_nodes, 'nn_ind')}")
    logger.info(f"  使用生成几何参数: {use_generated_geometry}")
    
    return 0


def insert_rigid_instance(
    trainer,
    instance_data: Dict[str, Any],
    new_instance_id: Optional[int] = None,
    device: str = "cuda"
) -> int:
    """
    将保存的Rigid实例数据插入到场景中
    
    Args:
        trainer: 训练器实例
        instance_data: 实例数据字典
        new_instance_id: 新的实例ID，如果为None则自动分配
        device: 设备
        
    Returns:
        分配的新实例ID
    """
    logger.info("=== 开始插入Rigid实例 ===")
    
    if "RigidNodes" not in trainer.models:
        raise ValueError("模型中没有找到RigidNodes")
    
    rigid_nodes = trainer.models["RigidNodes"]
    
    # 检查是否需要初始化RigidNodes
    is_empty = (not hasattr(rigid_nodes, 'point_ids') or 
                rigid_nodes.point_ids is None or 
                rigid_nodes.point_ids.numel() == 0)
    
    if is_empty:
        logger.info("RigidNodes为空，开始初始化...")
        initialize_empty_rigid_nodes(rigid_nodes, instance_data, device)
        return 0  # 第一个实例总是ID 0
    
    # 如果不为空，获取现有实例ID
    try:
        current_ids = rigid_nodes.point_ids[..., 0].unique().cpu().numpy()
    except Exception as e:
        logger.warning(f"无法获取现有实例ID: {e}")
        current_ids = []
        
    if new_instance_id is None:
        new_instance_id = int(max(current_ids) + 1) if len(current_ids) > 0 else 0
    elif new_instance_id in current_ids:
        logger.warning(f"实例ID {new_instance_id} 已存在，将覆盖现有实例")
    
    logger.info(f"分配新实例ID: {new_instance_id}")
    logger.info(f"当前存在的实例ID: {current_ids}")
    
    # 获取实例数据
    geometry = instance_data["geometry"]
    motion = instance_data["motion"]
    metadata = instance_data.get("metadata", {})
    
    num_points = geometry["_means"].shape[0]
    num_frames = motion["instances_quats"].shape[0]
    
    logger.info(f"插入的实例信息: 点数={num_points}, 帧数={num_frames}")
    
    with torch.no_grad():
        # 将数据移动到指定设备
        for key, value in geometry.items():
            if isinstance(value, torch.Tensor):
                geometry[key] = value.to(device)
        
        for key, value in motion.items():
            if isinstance(value, torch.Tensor):
                motion[key] = value.to(device)
        
        # 1. 扩展几何体数据
        rigid_nodes._means = Parameter(
            torch.cat([rigid_nodes._means, geometry["_means"]], dim=0)
        )
        rigid_nodes._scales = Parameter(
            torch.cat([rigid_nodes._scales, geometry["_scales"]], dim=0)
        )
        rigid_nodes._quats = Parameter(
            torch.cat([rigid_nodes._quats, geometry["_quats"]], dim=0)
        )
        rigid_nodes._features_dc = Parameter(
            torch.cat([rigid_nodes._features_dc, geometry["_features_dc"]], dim=0)
        )
        rigid_nodes._features_rest = Parameter(
            torch.cat([rigid_nodes._features_rest, geometry["_features_rest"]], dim=0)
        )
        rigid_nodes._opacities = Parameter(
            torch.cat([rigid_nodes._opacities, geometry["_opacities"]], dim=0)
        )
        
        # 2. 创建新的point_ids并连接
        new_point_ids = torch.full(
            (num_points, 1), new_instance_id, 
            device=device, dtype=torch.long
        )
        rigid_nodes.point_ids = torch.cat([rigid_nodes.point_ids, new_point_ids], dim=0)
        logger.info(f"扩展point_ids，新增实例ID: {new_instance_id}")
        
        # 3. 扩展运动数据 - 需要匹配当前场景的帧数
        current_num_frames = rigid_nodes.instances_quats.shape[0]
        current_num_instances = rigid_nodes.instances_quats.shape[1]
        
        logger.info(f"当前场景状态: {current_num_frames}帧, {current_num_instances}实例")
        
        # 调整加载的运动数据以匹配当前场景帧数
        if num_frames != current_num_frames:
            logger.info(f"调整运动数据帧数: {num_frames} -> {current_num_frames}")
            motion_adjusted = {}
            for key, value in motion.items():
                if isinstance(value, torch.Tensor):
                    motion_adjusted[key] = match_sequence_length(current_num_frames, value)
                else:
                    motion_adjusted[key] = value
            motion = motion_adjusted
        
        # 扩展实例相关的张量
        new_instances_quats = torch.zeros(
            current_num_frames, current_num_instances + 1, 4, device=device
        )
        new_instances_quats[:, :current_num_instances] = rigid_nodes.instances_quats
        new_instances_quats[:, current_num_instances] = motion["instances_quats"]
        rigid_nodes.instances_quats = Parameter(new_instances_quats)
        
        new_instances_trans = torch.zeros(
            current_num_frames, current_num_instances + 1, 3, device=device
        )
        new_instances_trans[:, :current_num_instances] = rigid_nodes.instances_trans
        new_instances_trans[:, current_num_instances] = motion["instances_trans"]
        rigid_nodes.instances_trans = Parameter(new_instances_trans)
        
        new_instances_fv = torch.zeros(
            current_num_frames, current_num_instances + 1, device=device, dtype=torch.bool
        )
        new_instances_fv[:, :current_num_instances] = rigid_nodes.instances_fv
        new_instances_fv[:, current_num_instances] = motion["instances_fv"]
        rigid_nodes.instances_fv = new_instances_fv
        
        # 4. 扩展实例大小信息（如果存在）
        if "size" in instance_data and hasattr(rigid_nodes, "instances_size"):
            size_data = instance_data["size"].to(device)
            new_instances_size = torch.zeros(
                current_num_instances + 1, 3, device=device
            )
            new_instances_size[:current_num_instances] = rigid_nodes.instances_size
            new_instances_size[current_num_instances] = size_data
            rigid_nodes.instances_size = new_instances_size
            logger.info(f"扩展实例大小信息: {size_data}")
        elif "instances_size" in motion:
            # 如果instance_data中没有size但motion中有instances_size
            size_data = motion["instances_size"].to(device)
            if hasattr(rigid_nodes, "instances_size"):
                new_instances_size = torch.zeros(
                    current_num_instances + 1, 3, device=device
                )
                new_instances_size[:current_num_instances] = rigid_nodes.instances_size
                new_instances_size[current_num_instances] = size_data
                rigid_nodes.instances_size = new_instances_size
            else:
                # 如果rigid_nodes没有instances_size属性，创建一个
                all_instances_size = torch.zeros(
                    current_num_instances + 1, 3, device=device
                )
                all_instances_size[current_num_instances] = size_data
                # 为其他实例设置默认大小
                all_instances_size[:current_num_instances] = torch.tensor([1.0, 1.0, 1.0], device=device)
                rigid_nodes.instances_size = all_instances_size
            logger.info(f"从motion中恢复实例大小信息: {size_data}")
    
    logger.info(f"Rigid实例插入完成，新实例ID: {new_instance_id}")
    logger.info(f"更新后的模型状态:")
    logger.info(f"  总点数: {rigid_nodes._means.shape[0]}")
    logger.info(f"  总实例数: {rigid_nodes.instances_quats.shape[1]}")
    logger.info(f"  帧数: {rigid_nodes.instances_quats.shape[0]}")
    
    # 验证关键张量的形状
    expected_instances = rigid_nodes.instances_quats.shape[1]
    logger.info(f"验证Rigid实例数量一致性:")
    logger.info(f"  instances_quats: {rigid_nodes.instances_quats.shape}")
    logger.info(f"  instances_trans: {rigid_nodes.instances_trans.shape}")
    logger.info(f"  instances_fv: {rigid_nodes.instances_fv.shape}")
    if hasattr(rigid_nodes, "instances_size"):
        logger.info(f"  instances_size: {rigid_nodes.instances_size.shape}")
    
    return new_instance_id

def initialize_empty_rigid_nodes(rigid_nodes, instance_data, device):
    """
    正确的RigidNodes初始化函数 - 严格按照原始训练代码逻辑
    
    关键修复：
    1. 使用K近邻计算scale初始化
    2. 使用随机四元数初始化旋转
    3. 正确的颜色初始化流程
    4. 正确的运动参数初始化
    5. 处理quat_act激活函数
    """
    from torch.nn import Parameter
    from models.gaussians.basics import RGB2SH, num_sh_bases, random_quat_tensor, k_nearest_sklearn
    from pytorch3d.transforms import matrix_to_quaternion
    
    logger.info("正确初始化空的RigidNodes...")
    
    # 获取实例数据
    geometry = instance_data["geometry"]
    motion = instance_data["motion"]
    
    # 移动数据到设备
    for key, value in geometry.items():
        if isinstance(value, torch.Tensor):
            geometry[key] = value.to(device)
    
    for key, value in motion.items():
        if isinstance(value, torch.Tensor):
            motion[key] = value.to(device)
    
    num_points = geometry["_means"].shape[0]
    num_frames = motion["instances_quats"].shape[0]
    
    logger.info(f"初始化参数: 点数={num_points}, 帧数={num_frames}")
    
    with torch.no_grad():
        # 第一步：初始化几何参数 - 按照原始逻辑
        init_means = geometry["_means"]
        
        # 使用K近邻计算scale（这是原始代码的关键步骤）
        logger.info("计算K近邻距离用于scale初始化...")
        try:
            distances, _ = k_nearest_sklearn(init_means.data, 3)
            distances = torch.from_numpy(distances).to(device)
            avg_dist = distances.mean(dim=-1, keepdim=True)
            avg_dist = avg_dist.clamp(0.002, 100)
            init_scales = torch.log(avg_dist.repeat(1, 3))
            logger.info(f"使用K近邻计算的scale，平均距离范围: {avg_dist.min():.6f} - {avg_dist.max():.6f}")
        except Exception as e:
            logger.warning(f"K近邻计算失败: {e}，使用保存的scale")
            init_scales = geometry["_scales"]
        
        # 使用随机四元数初始化旋转（这是原始代码的做法）
        logger.info("生成随机四元数用于旋转初始化...")
        try:
            init_quats = random_quat_tensor(num_points).to(device)
            logger.info("使用随机四元数初始化旋转")
        except Exception as e:
            logger.warning(f"随机四元数生成失败: {e}，使用保存的四元数")
            init_quats = geometry["_quats"]
        
        # 设置几何参数
        rigid_nodes._means = Parameter(init_means)
        rigid_nodes._scales = Parameter(init_scales)
        rigid_nodes._quats = Parameter(init_quats)
        
        # 第二步：初始化颜色 - 按照原始逻辑
        logger.info("初始化颜色参数...")
        
        # 尝试从保存的颜色数据恢复，或使用随机颜色
        if "init_colors" in instance_data:
            init_colors = instance_data["init_colors"].to(device)
            logger.info("使用保存的颜色数据")
        else:
            # 从features_dc尝试恢复颜色
            try:
                features_dc = geometry["_features_dc"]
                if features_dc.shape[1] == 3:
                    # 如果features_dc就是RGB格式
                    init_colors = features_dc
                else:
                    # 如果是SH格式，使用随机颜色
                    init_colors = torch.rand((num_points, 3), device=device)
                logger.info("从features_dc恢复颜色")
            except:
                init_colors = torch.rand((num_points, 3), device=device)
                logger.info("使用随机颜色")
        
        # 确保颜色在正确范围内
        init_colors = torch.clamp(init_colors, 0.0, 1.0)
        
        # 转换为球谐系数
        fused_color = RGB2SH(init_colors)
        dim_sh = num_sh_bases(rigid_nodes.sh_degree)
        
        shs = torch.zeros((num_points, dim_sh, 3)).float().to(device)
        if rigid_nodes.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        
        rigid_nodes._features_dc = Parameter(shs[:, 0, :])
        rigid_nodes._features_rest = Parameter(shs[:, 1:, :])
        
        # 初始化不透明度
        rigid_nodes._opacities = Parameter(
            torch.logit(0.1 * torch.ones(num_points, 1, device=device))
        )
        
        # 第三步：初始化运动参数 - 关键部分
        logger.info("初始化运动参数...")
        
        # 设置instances_fv
        instances_fv = motion["instances_fv"]
        if instances_fv.dim() == 1:
            instances_fv = instances_fv.unsqueeze(1)  # [num_frames, 1]
        rigid_nodes.instances_fv = instances_fv
        
        # 设置point_ids
        rigid_nodes.point_ids = torch.full((num_points, 1), 0, device=device, dtype=torch.long)
        
        # 处理instances_pose - 这是关键步骤
        # 从motion中恢复pose信息
        instances_quats = motion["instances_quats"]
        instances_trans = motion["instances_trans"]
        
        # 确保形状正确
        if instances_quats.dim() == 2:
            instances_quats = instances_quats.unsqueeze(1)  # [num_frames, 1, 4]
        if instances_trans.dim() == 2:
            instances_trans = instances_trans.unsqueeze(1)  # [num_frames, 1, 3]
        
        # 构造pose矩阵用于get_instances_quats处理
        logger.info("构造pose矩阵并处理四元数...")
        
        # 创建4x4变换矩阵
        instances_pose = torch.zeros(num_frames, 1, 4, 4, device=device)
        
        # 设置旋转部分
        for fi in range(num_frames):
            if instances_fv[fi, 0]:  # 如果该帧实例可见
                quat = instances_quats[fi, 0]  # [4]
                trans = instances_trans[fi, 0]  # [3]
                
                # 四元数转旋转矩阵
                from pytorch3d.transforms import quaternion_to_matrix
                rot_matrix = quaternion_to_matrix(quat.unsqueeze(0)).squeeze(0)  # [3, 3]
                
                # 构造4x4矩阵
                instances_pose[fi, 0, :3, :3] = rot_matrix
                instances_pose[fi, 0, :3, 3] = trans
                instances_pose[fi, 0, 3, 3] = 1.0
        
        # 使用get_instances_quats处理（这是原始代码的做法）
        try:
            processed_quats = rigid_nodes.get_instances_quats(instances_pose)
            logger.info(f"使用get_instances_quats处理后的四元数形状: {processed_quats.shape}")
        except Exception as e:
            logger.warning(f"get_instances_quats处理失败: {e}，使用原始四元数")
            processed_quats = instances_quats
        
        # 设置运动参数
        rigid_nodes.instances_quats = Parameter(processed_quats)  # [num_frames, 1, 4]
        rigid_nodes.instances_trans = Parameter(instances_trans)  # [num_frames, 1, 3]
        
        # 第四步：设置实例大小
        if "size" in instance_data:
            size_data = instance_data["size"].to(device)
            rigid_nodes.instances_size = size_data.unsqueeze(0) if size_data.dim() == 1 else size_data
        elif "instances_size" in motion:
            size_data = motion["instances_size"].to(device)
            rigid_nodes.instances_size = size_data.unsqueeze(0) if size_data.dim() == 1 else size_data
        else:
            # 从点云边界计算合理的大小
            means_bounds = init_means.abs().max(dim=0)[0]
            default_size = means_bounds * 2.0
            rigid_nodes.instances_size = default_size.unsqueeze(0)
            logger.info(f"计算默认实例大小: {default_size}")
        
        # 验证最终形状
        expected_shapes = {
            "_means": (num_points, 3),
            "_scales": (num_points, 3),
            "_quats": (num_points, 4),
            "_features_dc": (num_points, 3),
            "_features_rest": (num_points, dim_sh-1, 3) if dim_sh > 1 else (num_points, 0, 3),
            "_opacities": (num_points, 1),
            "instances_quats": (num_frames, 1, 4),
            "instances_trans": (num_frames, 1, 3),
            "instances_fv": (num_frames, 1),
            "instances_size": (1, 3),
            "point_ids": (num_points, 1)
        }
        
        logger.info("验证最终张量形状:")
        all_correct = True
        for attr_name, expected_shape in expected_shapes.items():
            if hasattr(rigid_nodes, attr_name):
                actual_tensor = getattr(rigid_nodes, attr_name)
                actual_shape = actual_tensor.shape
                status = "✓" if actual_shape == expected_shape else "✗"
                logger.info(f"  {status} {attr_name}: {actual_shape} (期望: {expected_shape})")
                if actual_shape != expected_shape:
                    all_correct = False
            else:
                logger.warning(f"  ? {attr_name}: 属性不存在")
                all_correct = False
        
        if not all_correct:
            logger.error("存在形状不匹配的张量，可能影响渲染")
        
        # 验证设备一致性
        logger.info("验证设备一致性:")
        for attr_name in ['_means', '_scales', '_quats', 'instances_quats', 'instances_trans']:
            if hasattr(rigid_nodes, attr_name):
                attr = getattr(rigid_nodes, attr_name)
                if isinstance(attr, torch.Tensor):
                    logger.info(f"  {attr_name}: {attr.device}")
                elif hasattr(attr, 'device'):
                    logger.info(f"  {attr_name}: {attr.device}")
        
        # 验证requires_grad设置
        logger.info("验证梯度设置:")
        for attr_name in ['_means', '_scales', '_quats', '_features_dc', '_features_rest', '_opacities']:
            if hasattr(rigid_nodes, attr_name):
                attr = getattr(rigid_nodes, attr_name)
                if isinstance(attr, Parameter):
                    logger.info(f"  {attr_name}: requires_grad={attr.requires_grad}")
    
    logger.info("RigidNodes初始化完成:")
    logger.info(f"  点数: {num_points}")
    logger.info(f"  帧数: {num_frames}")
    logger.info(f"  实例数: 1")
    logger.info(f"  实例ID: 0")
    logger.info(f"  使用K近邻scale: {init_scales is not geometry['_scales']}")
    logger.info(f"  使用随机四元数: {init_quats is not geometry['_quats']}")
    
    return 0


# 备用的K近邻实现，如果原始函数不可用
def k_nearest_sklearn_backup(points, k=3):
    """K近邻的备用实现"""
    try:
        from sklearn.neighbors import NearestNeighbors
        points_np = points.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points_np)
        distances, indices = nbrs.kneighbors(points_np)
        return distances[:, 1:], indices[:, 1:]  # 排除自己
    except ImportError:
        logger.warning("sklearn不可用，使用简单距离计算")
        # 简单的距离计算备用方案
        n_points = points.shape[0]
        distances = []
        for i in range(n_points):
            dist = torch.norm(points - points[i], dim=1)
            dist[i] = float('inf')  # 排除自己
            k_dist = torch.topk(dist, k, largest=False)[0]
            distances.append(k_dist.cpu().numpy())
        distances = np.array(distances)
        indices = np.zeros((n_points, k), dtype=int)
        return distances, indices


# 备用的随机四元数生成
def random_quat_tensor_backup(num_points, device):
    """随机四元数的备用实现"""
    try:
        from utils.geometry import uniform_sample_sphere
        quats = uniform_sample_sphere(num_points, device)
        return quats
    except ImportError:
        logger.warning("uniform_sample_sphere不可用，使用简单随机四元数")
        # 简单的随机四元数生成
        quats = torch.randn(num_points, 4, device=device)
        quats = quats / torch.norm(quats, dim=1, keepdim=True)
        return quats


def batch_save_instances(
    trainer,
    instance_type: str,  # "smpl" or "rigid"
    instance_ids: List[int],
    save_dir: str,
    prefix: str = ""
) -> List[str]:
    """
    批量保存多个实例
    
    Args:
        trainer: 训练器实例
        instance_type: 实例类型 ("smpl" 或 "rigid")
        instance_ids: 要保存的实例ID列表
        save_dir: 保存目录
        prefix: 文件名前缀
        
    Returns:
        保存的文件路径列表
    """
    logger.info(f"=== 开始批量保存{len(instance_ids)}个{instance_type}实例 ===")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    save_func = save_smpl_instance if instance_type.lower() == "smpl" else save_rigid_instance
    
    for instance_id in instance_ids:
        try:
            filename = f"{prefix}{instance_type}_instance_{instance_id}.pkl"
            file_path = save_dir / filename
            
            save_func(trainer, instance_id, str(file_path))
            saved_files.append(str(file_path))
            
        except Exception as e:
            logger.error(f"保存实例 {instance_id} 时出错: {str(e)}")
    
    logger.info(f"批量保存完成，成功保存 {len(saved_files)}/{len(instance_ids)} 个实例")
    return saved_files


# 在 scene_editing.py 文件末尾添加以下函数

def handle_insert_and_transform_operation(
    trainer, 
    instance_files: List[str],
    new_instance_ids: List[int] = None,
    translation: List[float] = None,
    rotation_axis: List[float] = None,
    rotation_angle: float = None,
    frame_range: str = None,
    device: str = "cuda"
):
    """
    处理插入并变换的组合操作
    
    Args:
        trainer: 训练器实例
        instance_files: 要插入的实例文件路径列表
        new_instance_ids: 新实例ID列表，如果不指定则自动分配
        translation: 平移偏移量 [x, y, z]
        rotation_axis: 旋转轴 [x, y, z]
        rotation_angle: 旋转角度(度)
        frame_range: 帧范围，格式为 "start-end" 或单个数字
        device: 设备类型
    
    Returns:
        tuple: (插入的实例ID列表, 操作是否成功)
    """
    logger.info("=" * 60)
    logger.info("开始插入并变换实例")
    logger.info("=" * 60)
    
    # 验证输入参数
    if not instance_files:
        logger.error("没有指定实例文件")
        return [], False
    
    if not translation and not (rotation_axis and rotation_angle is not None):
        logger.error("需要指定至少一种变换类型（平移或旋转）")
        return [], False
    
    # 第一步：插入实例
    logger.info("第一步：插入实例")
    
    # 验证实例文件
    valid_files = []
    for file_path in instance_files:
        if not os.path.exists(file_path):
            logger.warning(f"实例文件不存在，跳过: {file_path}")
            continue
        valid_files.append(file_path)
    
    if not valid_files:
        logger.error("没有有效的实例文件可以插入")
        return [], False
    
    logger.info(f"将插入以下实例文件:")
    for file_path in valid_files:
        try:
            file_size = Path(file_path).stat().st_size / 1024 / 1024  # MB
            logger.info(f"  {file_path} ({file_size:.2f} MB)")
        except:
            logger.info(f"  {file_path}")
    
    # 执行插入操作
    inserted_instances = []  # [(instance_id, instance_type), ...]
    for i, file_path in enumerate(valid_files):
        try:
            logger.info(f"\n--- 插入文件 {i+1}/{len(valid_files)}: {file_path} ---")
            
            # 加载实例数据
            instance_data = load_instance_data(file_path)
            
            # 确定实例类型
            if "smpl_template" in instance_data or "voxel_deformer" in instance_data:
                instance_type = "smpl"
                insert_func = insert_smpl_instance
            else:
                instance_type = "rigid"
                insert_func = insert_rigid_instance
            
            logger.info(f"检测到实例类型: {instance_type.upper()}")
            
            # 验证模型是否支持该类型
            model_key = get_model_key(instance_type)
            if model_key not in trainer.models:
                logger.error(f"目标场景不支持{instance_type.upper()}实例，跳过")
                continue
            
            # 确定新实例ID
            new_id = new_instance_ids[i] if new_instance_ids and i < len(new_instance_ids) else None
            
            # 执行插入
            actual_id = insert_func(
                trainer=trainer,
                instance_data=instance_data,
                new_instance_id=new_id,
                device=device
            )
            
            if actual_id is not None:
                inserted_instances.append((actual_id, instance_type))
                logger.info(f"✓ 成功插入实例，分配ID: {actual_id}, 类型: {instance_type}")
            else:
                logger.error(f"✗ 插入失败，未返回有效ID")
            
        except Exception as e:
            logger.error(f"插入文件 {file_path} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    if not inserted_instances:
        logger.error("没有成功插入任何实例")
        return [], False
    
    logger.info(f"第一步完成：成功插入 {len(inserted_instances)} 个实例")
    inserted_ids = [instance_id for instance_id, _ in inserted_instances]
    logger.info(f"插入的实例ID: {inserted_ids}")
    
    # 第二步：对插入的实例进行变换
    logger.info("\n第二步：对插入的实例进行变换")
    
    success_count = 0
    for instance_id, instance_type in inserted_instances:
        logger.info(f"\n--- 变换实例 {instance_id} ({instance_type}) ---")
        
        try:
            model_key = get_model_key(instance_type)
            model = trainer.models[model_key]
            
            # 确定帧范围
            total_frames = model.num_frames if hasattr(model, 'num_frames') else model.instances_trans.shape[0]
            
            if frame_range:
                try:
                    if "-" in frame_range:
                        start_frame, end_frame = map(int, frame_range.split("-"))
                        frame_indices = list(range(max(0, start_frame), min(total_frames, end_frame + 1)))
                    else:
                        frame_idx = int(frame_range)
                        frame_indices = [frame_idx] if 0 <= frame_idx < total_frames else []
                except ValueError:
                    logger.warning(f"无效的帧范围格式: {frame_range}，使用所有帧")
                    frame_indices = list(range(total_frames))
            else:
                frame_indices = list(range(total_frames))
            
            logger.info(f"将对 {len(frame_indices)} 帧进行变换")
            
            # 执行平移（如果指定）
            if translation:
                logger.info(f"对实例 {instance_id} 进行平移: {translation}")
                translation_offset = torch.tensor(translation, device=trainer.device, dtype=torch.float32)
                
                with torch.no_grad():
                    for frame_idx in frame_indices:
                        model.add_transform_offset(
                            instance_id=instance_id,
                            frame_idx=frame_idx,
                            translation_offset=translation_offset
                        )
                
                logger.info(f"✓ 完成平移，影响 {len(frame_indices)} 帧")
            
            # 执行旋转（如果指定）
            if rotation_axis and rotation_angle is not None:
                logger.info(f"对实例 {instance_id} 进行旋转: 轴={rotation_axis}, 角度={rotation_angle}度")
                
                # 处理角度转换
                angle_radians = math.radians(rotation_angle)
                
                # 归一化旋转轴
                rotation_axis_tensor = torch.tensor(rotation_axis, device=trainer.device, dtype=torch.float32)
                rotation_axis_tensor = rotation_axis_tensor / torch.norm(rotation_axis_tensor)
                
                # 创建轴角表示并转换为四元数
                axis_angle = rotation_axis_tensor * angle_radians
                from pytorch3d.transforms import axis_angle_to_quaternion
                rotation_quaternion = axis_angle_to_quaternion(axis_angle)
                
                with torch.no_grad():
                    for frame_idx in frame_indices:
                        model.add_transform_offset(
                            instance_id=instance_id,
                            frame_idx=frame_idx,
                            rotation_offset=rotation_quaternion
                        )
                
                logger.info(f"✓ 完成旋转，影响 {len(frame_indices)} 帧")
            
            success_count += 1
            logger.info(f"✓ 实例 {instance_id} 变换完成")
            
        except Exception as e:
            logger.error(f"变换实例 {instance_id} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"\n第二步完成：成功变换 {success_count}/{len(inserted_instances)} 个实例")
    
    # 打印最终状态
    logger.info("\n插入并变换操作完成，最终实例ID:")
    logger.info(f"  成功插入并变换的实例: {inserted_ids}")
    
    return inserted_ids, success_count == len(inserted_instances)