from typing import List, Dict, Optional
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

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import render_images, save_videos
from pathlib import Path
from utils.simplify_loc2rot import joints2smpl
from utils.rotation_conversions import axis_angle_to_quaternion, quaternion_to_matrix
from utils.rotation_conversions import quaternion_to_euler_angles
from third_party.smplx.transfer_model.transfer_model import run_fitting

logger = logging.getLogger()

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


def RGB2SH(rgb):
    """将RGB颜色转换为球谐系数"""
    # 添加调试信息
    if hasattr(rgb, 'shape'):
        logger.debug(f"RGB2SH输入形状: {rgb.shape}")
    
    # 检查输入维度并确保输出是正确的形状
    if len(rgb.shape) == 2 and rgb.shape[1] == 3:
        # 标准输入 [N, 3]
        result = (rgb - 0.5) / 0.28209479177387814
        return result
    elif len(rgb.shape) == 1:
        # 如果输入是一维的 [N]，扩展为 [N, 3]
        rgb_expanded = rgb.unsqueeze(1).repeat(1, 3)
        result = (rgb_expanded - 0.5) / 0.28209479177387814
        return result
    else:
        # 其他情况，尝试直接转换
        logger.warning(f"RGB2SH: 未预期的输入形状 {rgb.shape}，尝试直接转换")
        return (rgb - 0.5) / 0.28209479177387814

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

def load_smplx_to_smpl(npz_path: str) -> Dict:
    """直接从NPZ文件加载并转换SMPLX姿态到SMPL格式"""
    logger.info(f"开始加载SMPLX数据: {npz_path}")
    try:
        smplx_data = np.load(npz_path)
        logger.info(f"加载成功，poses shape: {smplx_data['poses'].shape}")
        
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
        
        return {
            'betas': torch.from_numpy(betas).float(),
            'global_orient': smpl_quats[:, 0],  # [nframe, 4]
            'body_pose': smpl_quats[:, 1:],     # [nframe, 23, 4]
            'transl': torch.from_numpy(trans).float()
        }
    except Exception as e:
        logger.error(f"加载SMPLX数据失败: {str(e)}")
        raise

def load_ply_and_create_instance(ply_path: str, smpl_params: Dict, target_vertices: int = None) -> Dict:
    """加载PLY文件并创建SMPL实例数据，可选择采样到指定顶点数"""
    logger.info(f"开始加载PLY文件: {ply_path}")
    try:
        # 加载PLY点云
        mesh = trimesh.load(ply_path)
        points = torch.from_numpy(mesh.vertices).float()
        original_vertex_count = len(points)
        logger.info(f"加载PLY成功，原始顶点数: {original_vertex_count}")
        
        # 处理顶点颜色 - 完全重写颜色处理逻辑
        colors = None
        has_colors = False
        
        # 检查mesh.visual是否存在颜色信息
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            if mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
                logger.info(f"检测到顶点颜色，形状: {mesh.visual.vertex_colors.shape}")
                # 通常PLY文件中颜色是RGBA格式，取前3个通道作为RGB
                if mesh.visual.vertex_colors.shape[1] >= 3:
                    colors = torch.from_numpy(mesh.visual.vertex_colors[:, :3]).float() / 255.0
                    has_colors = True
                    logger.info(f"颜色范围: {colors.min().item()} - {colors.max().item()}")
        
        # 如果没有颜色信息，使用随机彩色而不是白色
        if not has_colors:
            logger.warning("PLY文件没有颜色信息，使用随机彩色")
            # 使用HSV色彩空间生成鲜艳的颜色
            hue = torch.rand(original_vertex_count, 1)  # 随机色调
            saturation = torch.ones(original_vertex_count, 1) * 0.8  # 高饱和度
            value = torch.ones(original_vertex_count, 1) * 0.9  # 高亮度
            
            # 转换HSV到RGB
            h = hue * 6.0
            i = torch.floor(h)
            f = h - i
            p = value * (1.0 - saturation)
            q = value * (1.0 - saturation * f)
            t = value * (1.0 - saturation * (1.0 - f))
            
            i = i % 6
            
            r = torch.zeros_like(hue)
            g = torch.zeros_like(hue)
            b = torch.zeros_like(hue)
            
            mask = (i == 0)
            r[mask] = value[mask]
            g[mask] = t[mask]
            b[mask] = p[mask]
            
            mask = (i == 1)
            r[mask] = q[mask]
            g[mask] = value[mask]
            b[mask] = p[mask]
            
            mask = (i == 2)
            r[mask] = p[mask]
            g[mask] = value[mask]
            b[mask] = t[mask]
            
            mask = (i == 3)
            r[mask] = p[mask]
            g[mask] = q[mask]
            b[mask] = value[mask]
            
            mask = (i == 4)
            r[mask] = t[mask]
            g[mask] = p[mask]
            b[mask] = value[mask]
            
            mask = (i == 5)
            r[mask] = value[mask]
            g[mask] = p[mask]
            b[mask] = q[mask]
            
            colors = torch.cat([r, g, b], dim=1)
            logger.info(f"生成随机颜色，范围: {colors.min().item()} - {colors.max().item()}")
        
        # 如果指定了目标顶点数，进行采样
        if target_vertices is not None and target_vertices != original_vertex_count:
            logger.info(f"将点云采样至{target_vertices}个顶点")
            if original_vertex_count > target_vertices:
                # 降采样
                indices = torch.linspace(0, original_vertex_count-1, target_vertices).long()
                points = points[indices]
                colors = colors[indices]
            else:
                # 上采样
                repeat_factor = int(np.ceil(target_vertices / original_vertex_count))
                repeated_points = points.repeat(repeat_factor, 1)
                repeated_colors = colors.repeat(repeat_factor, 1)
                points = repeated_points[:target_vertices]
                colors = repeated_colors[:target_vertices]
        
        # 创建SMPL实例数据
        instance_data = {
            "class_name": "smpl",
            "pts": points,
            "colors": colors,
            "betas": smpl_params['betas'].unsqueeze(0),  # [1,10]
            "global_orient": smpl_params['global_orient'],  # [nframe,4]
            "body_pose": smpl_params['body_pose'],         # [nframe,23,4]
            "transl": smpl_params['transl'],               # [nframe,3]
            "size": torch.tensor([1.0, 1.0, 1.0]),
            "frame_info": torch.ones(1), 
            "num_pts": len(points)
        }
        
        logger.info(f"成功创建SMPL实例数据，顶点数: {len(points)}，颜色范围: {colors.min().item()} - {colors.max().item()}")
        return instance_data
        
    except Exception as e:
        logger.error(f"加载PLY文件失败: {str(e)}")
        raise
    

def replace_smpl_instance(trainer, instance_id, new_instance, keep_translation=True, keep_global_rot=True):
    """完全替换SMPL实例"""
    smpl_nodes = trainer.models["SMPLNodes"]
    
    # 1. 创建完全删除旧实例的函数
    def remove_instance_points(instance_id):
        # 获取除目标实例外的所有点
        keep_mask = smpl_nodes.point_ids[..., 0] != instance_id
        
        # 保存这些点
        kept_means = smpl_nodes._means[keep_mask]
        kept_scales = smpl_nodes._scales[keep_mask]
        kept_quats = smpl_nodes._quats[keep_mask] 
        kept_features_dc = smpl_nodes._features_dc[keep_mask]
        kept_features_rest = smpl_nodes._features_rest[keep_mask] if hasattr(smpl_nodes, '_features_rest') else None
        kept_opacities = smpl_nodes._opacities[keep_mask]
        kept_point_ids = smpl_nodes.point_ids[keep_mask]
        
        # 保存其他实例对应的参数
        instance_params = {}
        for i in range(smpl_nodes.num_instances):
            if i != instance_id:
                instance_params[i] = {
                    'instances_quats': smpl_nodes.instances_quats[:, i].clone(),
                    'instances_trans': smpl_nodes.instances_trans[:, i].clone(),
                    'smpl_qauts': smpl_nodes.smpl_qauts[:, i].clone(),
                    'instances_fv': smpl_nodes.instances_fv[:, i].clone()
                }
        
        return kept_means, kept_scales, kept_quats, kept_features_dc, kept_features_rest, kept_opacities, kept_point_ids, instance_params
    
    # 2. 创建添加新实例点的函数
    def add_new_instance_points(instance_id, input_pts, input_colors, input_scales, input_quats, input_opacities):
        # 准备新实例的点ID
        new_point_ids = torch.full((len(input_pts), 1), instance_id, 
                                  device=input_pts.device, dtype=torch.long)
        
        # 准备新实例的球谐颜色系数
        colors_sh = (input_colors - 0.5) / 0.28209479177387814
        
        # 准备高阶球谐系数(如果需要)
        features_rest = None
        if hasattr(smpl_nodes, '_features_rest') and smpl_nodes._features_rest is not None:
            dim_sh = smpl_nodes._features_rest.shape[1] if smpl_nodes._features_rest.dim() > 1 else 15
            features_rest = torch.zeros((len(input_pts), dim_sh, 3), 
                                       device=input_pts.device, dtype=torch.float32)
            
            # 调整形状以匹配原始_features_rest
            if smpl_nodes._features_rest.dim() == 2:
                features_rest = features_rest.view(len(input_pts), -1)
        
        return new_point_ids, colors_sh, features_rest
    
    # 3. 执行完全替换
    try:
        with torch.no_grad():
            # 准备新实例数据
            num_frames = smpl_nodes.instances_quats.shape[0]
            device = smpl_nodes._means.device
            
            # 匹配姿态和平移参数
            global_orient = match_sequence_length(num_frames, new_instance["global_orient"])
            body_pose = match_sequence_length(num_frames, new_instance["body_pose"])
            trans = match_sequence_length(num_frames, new_instance["transl"])
            
            # 确保点云数据在正确设备上
            input_pts = new_instance["pts"].to(device)
            input_colors = new_instance["colors"].to(device)
            
            # 删除旧实例点并保存其他实例点
            (kept_means, kept_scales, kept_quats, kept_features_dc, 
             kept_features_rest, kept_opacities, kept_point_ids, 
             instance_params) = remove_instance_points(instance_id)
            
            # 准备新点云的参数
            scale_value = torch.ones(len(input_pts), 1 if smpl_nodes.ball_gaussians else 3, device=device) * 0.03
            input_scales = torch.log(scale_value)
            
            # 准备旋转
            identity_quat = torch.zeros(len(input_pts), 4, device=device)
            identity_quat[:, 3] = 1.0  # 单位四元数
            input_quats = identity_quat
            
            # 准备不透明度
            input_opacities = torch.ones(len(input_pts), 1, device=device) * 5.0  # 较大的logit值

            # 创建新实例的点ID和球谐系数
            new_point_ids, colors_sh, features_rest = add_new_instance_points(
                instance_id, input_pts, input_colors, input_scales, input_quats, input_opacities)
            
            # 合并保留的点和新点
            smpl_nodes._means = Parameter(torch.cat([kept_means, input_pts], dim=0))
            smpl_nodes._scales = Parameter(torch.cat([kept_scales, input_scales], dim=0))
            smpl_nodes._quats = Parameter(torch.cat([kept_quats, input_quats], dim=0))
            smpl_nodes._features_dc = Parameter(torch.cat([kept_features_dc, colors_sh], dim=0))
            smpl_nodes._opacities = Parameter(torch.cat([kept_opacities, input_opacities], dim=0))
            smpl_nodes.point_ids = torch.cat([kept_point_ids, new_point_ids], dim=0)
            
            # 合并高阶球谐系数
            if kept_features_rest is not None and features_rest is not None:
                if kept_features_rest.dim() == features_rest.dim():
                    smpl_nodes._features_rest = Parameter(torch.cat([kept_features_rest, features_rest], dim=0))
            
            # 更新姿态和平移
            if not keep_global_rot:
                for i in range(num_frames):
                    if i < len(global_orient):
                        smpl_nodes.instances_quats[i, instance_id] = global_orient[i]
            
            for i in range(num_frames):
                if i < len(body_pose):
                    smpl_nodes.smpl_qauts[i, instance_id] = body_pose[i]
            
            if not keep_translation:
                for i in range(num_frames):
                    if i < len(trans):
                        smpl_nodes.instances_trans[i, instance_id] = trans[i]
            
            # 更新SMPL形状参数
            if "betas" in new_instance and hasattr(smpl_nodes.template, "init_beta"):
                smpl_nodes.template.init_beta[instance_id] = new_instance["betas"][0]
                
                # 更新SMPL模板
                canonical_pose = smpl_nodes.template.canonical_pose
                body_pose = canonical_pose[None, 1:].repeat(1, 1, 1, 1)
                global_orient = canonical_pose[None, 0].repeat(1, 1, 1)
                
                instance_beta = smpl_nodes.template.init_beta[instance_id:instance_id+1]
                smpl_output = smpl_nodes.template._template_layer(
                    betas=instance_beta,
                    body_pose=body_pose,
                    global_orient=global_orient,
                    return_full_pose=True
                )
                
                smpl_nodes.template.J_canonical[instance_id] = smpl_output.J[0]
                A0 = smpl_output.A
                A0_inv = torch.inverse(A0)
                smpl_nodes.template.A0_inv[instance_id] = A0_inv
            
            # 确保实例可见性为1
            smpl_nodes.instances_fv[:, instance_id] = 1
            
            # 清除缓存
            if hasattr(trainer, "reset_renderer_cache"):
                trainer.reset_renderer_cache()
            
            if hasattr(trainer, "renderer") and hasattr(trainer.renderer, "reset_cache"):
                trainer.renderer.reset_cache()
                
            # 触发前向传播以更新内部状态
            orig_frame = smpl_nodes.cur_frame if hasattr(smpl_nodes, 'cur_frame') else 0
            smpl_nodes.cur_frame = 0
            dummy_cam = type('obj', (), {
                'camtoworlds': type('obj', (), {'data': torch.eye(4, 4).unsqueeze(0).to(device)})
            })
            _ = smpl_nodes.get_gaussians(dummy_cam)
            smpl_nodes.cur_frame = orig_frame
            
            logger.info(f"SMPL实例{instance_id}完全替换完成，新点云数量: {len(input_pts)}")
            
    except Exception as e:
        logger.error(f"替换SMPL实例失败: {str(e)}")
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
    
    # 加载并转换新实例
    if args.new_npz_path and args.new_ply_path:
        logger.info(f"从 {args.new_npz_path} 和 {args.new_ply_path} 加载新实例")
        
        # 先获取原始SMPL实例的点数
        smpl_nodes = trainer.models["SMPLNodes"]
        instance_id = args.instance_id
        pts_mask = smpl_nodes.point_ids[..., 0] == instance_id
        target_vertex_count = pts_mask.sum().item()
        logger.info(f"原始SMPL实例使用了 {target_vertex_count} 个顶点")
        
        # 1. 转换SMPLX到SMPL
        smplx_model_path = "smpl_models/smplx/SMPLX_NEUTRAL.npz"
        smpl_model_path = "smpl_models/smpl/SMPL_NEUTRAL.pkl"
        smpl_params = load_smplx_to_smpl(args.new_npz_path)
        
        # 2. 加载PLY并创建实例，指定目标顶点数
        new_instance = load_ply_and_create_instance(
            args.new_ply_path, 
            smpl_params, 
            target_vertices=target_vertex_count
        )
        
        # 3. 替换实例
        with torch.no_grad():
            replace_smpl_instance(
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
        "--new_npz_path", type=str, default="", help="Path to .npz file with SMPLX params"
    )
    parser.add_argument(
        "--new_ply_path", type=str, default="", help="Path to .ply file with point cloud"
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