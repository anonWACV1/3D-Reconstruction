from typing import List, Dict, Optional
from omegaconf import OmegaConf
import os
import time
import logging
import argparse
import torch
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
        
        # 如果指定了目标顶点数，进行采样
        if target_vertices is not None and target_vertices != original_vertex_count:
            logger.info(f"将PLY点云采样至{target_vertices}个顶点")
            
            if original_vertex_count > target_vertices:
                # 降采样 - 使用均匀采样
                indices = torch.linspace(0, original_vertex_count-1, target_vertices).long()
                points = points[indices]
                logger.info(f"点云降采样完成: {original_vertex_count} -> {len(points)}")
            else:
                # 上采样 - 复制点
                repeat_factor = int(np.ceil(target_vertices / original_vertex_count))
                repeated_points = points.repeat(repeat_factor, 1)
                points = repeated_points[:target_vertices]
                logger.info(f"点云上采样完成: {original_vertex_count} -> {len(points)}")
        
        # 处理顶点颜色
        has_colors = False
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors
            if colors is not None and len(colors) > 0:
                if colors.ndim == 1:
                    colors = np.stack([colors]*3, axis=1)
                colors = torch.from_numpy(colors[:, :3]).float() / 255.0
                has_colors = True
                logger.info(f"使用PLY中的顶点颜色数据，颜色数量: {len(colors)}")
            
                # 如果点云已采样，也需要对颜色进行相应采样
                if len(colors) != len(points):
                    if len(colors) > len(points):
                        # 降采样颜色
                        indices = torch.linspace(0, len(colors)-1, len(points)).long()
                        colors = colors[indices]
                    else:
                        # 上采样颜色
                        repeat_factor = int(np.ceil(len(points) / len(colors)))
                        repeated_colors = colors.repeat(repeat_factor, 1)
                        colors = repeated_colors[:len(points)]
        
        # 如果没有颜色或颜色数据为空，使用默认颜色
        if not has_colors:
            logger.warning("PLY无有效顶点颜色数据，使用默认白色")
            colors = torch.ones(len(points), 3, dtype=torch.float32)
        
        logger.info(f"最终颜色数据形状: {colors.shape}")
        
        # 调试打印SMPL参数维度
        logger.debug(f"global_orient shape: {smpl_params['global_orient'].shape}")
        logger.debug(f"body_pose shape: {smpl_params['body_pose'].shape}")
        
        # 确保输入维度正确
        global_orient = smpl_params['global_orient']
        body_pose = smpl_params['body_pose']
        
        if global_orient.dim() == 3:  # 如果是旋转矩阵 [1,3,3]
            global_orient = global_orient.squeeze(0)  # [3,3]
        if body_pose.dim() == 4:  # 如果是旋转矩阵序列 [1,23,3,3]
            body_pose = body_pose.squeeze(0)  # [23,3,3]
        
        # 转换为SMPL节点格式 (确保输出形状正确)
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
        logger.info(f"成功创建SMPL实例数据，最终顶点数: {len(points)}")
        return instance_data
        
    except Exception as e:
        logger.error(f"加载PLY文件失败: {str(e)}")
        raise

def replace_smpl_instance(
    trainer, 
    instance_id: int, 
    new_instance: Dict,
    keep_translation: bool = True,
    keep_global_rot: bool = True
):
    """替换SMPL实例"""
    smpl_nodes = trainer.models["SMPLNodes"]
    num_frames = smpl_nodes.instances_quats.shape[0]
    logger.info(f"目标序列帧数: {num_frames}")
    
    # 准备新实例数据并匹配帧数
    logger.info("处理全局旋转...")
    global_orient = new_instance["global_orient"]  # [nframe,4]
    logger.debug(f"原始global_orient形状: {global_orient.shape}")
    global_orient = match_sequence_length(
        num_frames, 
        global_orient.unsqueeze(1)  # [nframe,1,4]
    )
    logger.debug(f"匹配后global_orient形状: {global_orient.shape}")
    
    logger.info("处理身体姿态...")
    body_pose = new_instance["body_pose"]  # [nframe,23,4]
    logger.debug(f"原始body_pose形状: {body_pose.shape}")
    body_pose = match_sequence_length(
        num_frames,
        body_pose.unsqueeze(1)  # [nframe,1,23,4]
    )
    logger.debug(f"匹配后body_pose形状: {body_pose.shape}")
    
    logger.info("处理平移...")
    trans = new_instance["transl"]  # [nframe,3]
    logger.debug(f"原始trans形状: {trans.shape}")
    trans = match_sequence_length(
        num_frames,
        trans.unsqueeze(1)  # [nframe,1,3]
    )
    logger.debug(f"匹配后trans形状: {trans.shape}")
    
    # 更新模型参数
    with torch.no_grad():
        logger.info("更新姿态参数...")
        if not keep_global_rot:
            smpl_nodes.instances_quats[:, instance_id] = global_orient
            logger.debug(f"instances_quats更新后形状: {smpl_nodes.instances_quats[:, instance_id].shape}")
        
        smpl_nodes.smpl_qauts[:, instance_id] = body_pose.squeeze(1)
        logger.debug(f"smpl_qauts更新后形状: {smpl_nodes.smpl_qauts[:, instance_id].shape}")
        
        if not keep_translation:
            smpl_nodes.instances_trans[:, instance_id] = trans.squeeze(1)
            logger.debug(f"instances_trans更新后形状: {smpl_nodes.instances_trans[:, instance_id].shape}")
            
        if hasattr(smpl_nodes, 'betas'):
            smpl_nodes.betas[instance_id] = new_instance["betas"][0]
            logger.debug(f"betas更新后形状: {smpl_nodes.betas[instance_id].shape}")
        
        # 更新点云数据 - 检查点云大小是否匹配
        pts_mask = smpl_nodes.point_ids[..., 0] == instance_id
        logger.info(f"原始点云蒙版大小: {pts_mask.sum().item()}")
        logger.info(f"新点云大小: {new_instance['pts'].shape[0]}")
        
        # 获取当前SMPL实例使用的点数
        target_num_pts = pts_mask.sum().item()
        input_pts = new_instance["pts"]
        input_colors = new_instance["colors"]
        
        # 检查点云大小是否匹配
        if input_pts.shape[0] != target_num_pts:
            logger.warning(f"点云大小不匹配：输入{input_pts.shape[0]}点，但目标需要{target_num_pts}点")
            
            # 方法1：使用简单的采样来匹配点数
            if input_pts.shape[0] > target_num_pts:
                logger.info(f"对输入点云进行下采样从{input_pts.shape[0]}到{target_num_pts}点")
                # 简单采样
                indices = torch.linspace(0, input_pts.shape[0]-1, target_num_pts).long()
                input_pts = input_pts[indices]
                input_colors = input_colors[indices]
            else:
                logger.info(f"对输入点云进行上采样从{input_pts.shape[0]}到{target_num_pts}点")
                # 简单复制最后的点以达到目标数量
                repeat_times = int(np.ceil(target_num_pts / input_pts.shape[0]))
                repeated_pts = input_pts.repeat(repeat_times, 1)
                repeated_colors = input_colors.repeat(repeat_times, 1)
                input_pts = repeated_pts[:target_num_pts]
                input_colors = repeated_colors[:target_num_pts]
        
        logger.info(f"更新点云数据，共{target_num_pts}个点")
        smpl_nodes._means[pts_mask] = input_pts.to("cuda")
        colors_cuda = input_colors.to("cuda")
        colors_sh = RGB2SH(colors_cuda)
        # 确保颜色形状为 [N, 3]
        if len(colors_sh.shape) == 1:
            colors_sh = colors_sh.unsqueeze(1).repeat(1, 3)
        elif len(colors_sh.shape) == 2 and colors_sh.shape[1] == 1:
            colors_sh = colors_sh.repeat(1, 3)
        smpl_nodes._features_dc[pts_mask] = colors_sh
        logger.info("点云数据更新完成")

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
            render_keys=["rgbs", "depths"],
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