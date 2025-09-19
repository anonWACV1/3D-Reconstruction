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
from torch.nn import Parameter
from sklearn.neighbors import NearestNeighbors
import math

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
    """将RGB颜色转换为球谐系数(DC分量)"""
    # 确保输入是有效的RGB值
    if rgb.numel() == 0:
        raise ValueError("输入RGB张量为空")
    
    # 确保颜色值在[0,1]范围内
    if (rgb < 0).any() or (rgb > 1).any():
        rgb = torch.clamp(rgb, 0, 1)
    
    # 转换公式
    return (rgb - 0.5) / 0.28209479177387814


def k_nearest_sklearn(points, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    return distances, nbrs

def random_quat_tensor(N):
    u = torch.rand(N, 3)
    return torch.stack([
        torch.sqrt(1-u[:,0])*torch.sin(2*math.pi*u[:,1]),
        torch.sqrt(1-u[:,0])*torch.cos(2*math.pi*u[:,1]),
        torch.sqrt(u[:,0])*torch.sin(2*math.pi*u[:,2]),
        torch.sqrt(u[:,0])*torch.cos(2*math.pi*u[:,2])
    ], dim=1)

def num_sh_bases(degree):
    return (degree + 1) ** 2


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


def load_ply_and_create_instance(ply_path: str, smpl_params: Dict) -> Dict:
    """加载PLY文件并创建SMPL实例数据"""
    logger.info(f"开始加载PLY文件: {ply_path}")

    try:
        # 加载PLY点云
        mesh = trimesh.load(ply_path)
        points = torch.from_numpy(mesh.vertices).float()
        logger.info(f"加载PLY成功，顶点数: {len(points)}")
        
        # 处理顶点颜色 - 修复版本
        colors = None
        if hasattr(mesh.visual, 'vertex_colors'):
            # 尝试多种方式获取颜色数据
            try:
                # 方法1: 直接获取vertex_colors
                colors = mesh.visual.vertex_colors
                if colors is None or len(colors) == 0:
                    # 方法2: 尝试从face_colors转换
                    if hasattr(mesh.visual, 'face_colors') and len(mesh.visual.face_colors) > 0:
                        colors = np.repeat(mesh.visual.face_colors, 3, axis=0)
                
                # 确保颜色数据有效
                if colors is not None and len(colors) > 0:
                    if colors.ndim == 1:
                        colors = np.stack([colors]*3, axis=1)
                    colors = torch.from_numpy(colors[:, :3]).float() / 255.0
                    logger.info(f"使用PLY中的顶点颜色数据，形状: {colors.shape}")
                else:
                    raise ValueError("PLY文件中的颜色数据为空")
            except Exception as e:
                logger.warning(f"获取颜色数据失败: {str(e)}")
                colors = None
        
        # 如果没有获取到颜色数据，使用默认白色
        if colors is None:
            colors = torch.ones(len(points), 3, dtype=torch.float32)
            logger.warning("使用默认白色作为顶点颜色")
        
        # 确保颜色数据与顶点数量匹配
        if len(colors) != len(points):
            logger.warning(f"颜色数量({len(colors)})与顶点数量({len(points)})不匹配，将重复或截断")
            if len(colors) < len(points):
                # 重复颜色数据
                repeat_times = (len(points) // len(colors)) + 1
                colors = colors.repeat(repeat_times, 1)[:len(points)]
            else:
                # 截断颜色数据
                colors = colors[:len(points)]
        
        # 调试打印SMPL参数维度
        logger.info(f"global_orient shape: {smpl_params['global_orient'].shape}")
        logger.info(f"body_pose shape: {smpl_params['body_pose'].shape}")
        
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
        logger.info("成功创建SMPL实例数据")
        return instance_data
        
    except Exception as e:
        logger.error(f"加载PLY文件失败: {str(e)}")
        raise

def create_smpl_instance(
    trainer,
    new_instance: Dict,
    keep_translation: bool = True,
    keep_global_rot: bool = True
) -> int:
    """创建新的SMPL实例并返回实例ID"""
    smpl_nodes = trainer.models["SMPLNodes"]
    device = smpl_nodes.device
    
    # 首先将所有新实例的张量移动到正确的设备上
    for key in new_instance:
        if torch.is_tensor(new_instance[key]):
            new_instance[key] = new_instance[key].to(device)
            
    # 添加调试日志
    logger.info(f"smpl_nodes.sh_degree: {smpl_nodes.sh_degree}")
    num_points = new_instance["pts"].shape[0]
    logger.info(f"新实例的顶点数: {num_points}")
    
    # 1. 获取新实例ID (当前最大ID+1)
    existing_ids = torch.unique(smpl_nodes.point_ids[..., 0])
    new_instance_id = existing_ids.max().item() + 1 if len(existing_ids) > 0 else 0
    logger.info(f"创建新SMPL实例，ID: {new_instance_id}")
    
    # 2. 准备姿态参数
    num_frames = smpl_nodes.instances_quats.shape[0]
    logger.info(f"目标帧数: {num_frames}")
    
    # 处理global_orient维度
    global_orient = new_instance["global_orient"]
    logger.info(f"原始global_orient形状: {global_orient.shape}")
    
    # 确保global_orient是2D张量 [nframe, 4]
    if global_orient.dim() == 4:
        logger.info("检测到4D global_orient，正在降维...")
        global_orient = global_orient.squeeze(1).squeeze(1)  # [nframe,4]
    elif global_orient.dim() == 3:
        logger.info("检测到3D global_orient，正在降维...")
        global_orient = global_orient.squeeze(1)  # [nframe,4]
    
    logger.info(f"降维后global_orient形状: {global_orient.shape}")
    
    # 匹配序列长度
    global_orient = match_sequence_length(num_frames, global_orient)  # [num_frames,4]
    logger.info(f"匹配长度后global_orient形状: {global_orient.shape}")
    
    # 处理body_pose维度
    body_pose = new_instance["body_pose"]
    logger.info(f"原始body_pose形状: {body_pose.shape}")
    
    # 确保body_pose是3D张量 [nframe, 23, 4]
    if body_pose.dim() == 5:
        logger.info("检测到5D body_pose，正在降维...")
        body_pose = body_pose.squeeze(1).squeeze(1)  # [nframe,23,4]
    elif body_pose.dim() == 4 and body_pose.shape[1] == 1:
        logger.info("检测到4D body_pose，正在降维...")
        body_pose = body_pose.squeeze(1)  # [nframe,23,4]
    
    logger.info(f"降维后body_pose形状: {body_pose.shape}")
    
    # 匹配序列长度
    body_pose = match_sequence_length(num_frames, body_pose)  # [num_frames,23,4]
    logger.info(f"匹配长度后body_pose形状: {body_pose.shape}")
    
    # 处理trans维度
    trans = new_instance["transl"]
    logger.info(f"原始trans形状: {trans.shape}")
    trans = match_sequence_length(num_frames, trans)  # [num_frames,3]
    logger.info(f"匹配长度后trans形状: {trans.shape}")

    # 3. 创建新的高斯点云属性
    num_points = new_instance["pts"].shape[0]
    new_point_ids = torch.full((num_points, 1), new_instance_id, 
                            dtype=torch.long, device=device)
    
    # 初始化均值(位置)
    new_means = new_instance["pts"].to(device)
    
    # 初始化尺度(基于最近邻距离)
    distances, _ = k_nearest_sklearn(new_means.cpu().numpy(), 3)
    avg_dist = torch.from_numpy(distances).mean(dim=-1, keepdim=True).to(device)
    avg_dist = avg_dist.clamp(0.002, 100)
    new_scales = torch.log(avg_dist.repeat(1, 3))
    
    # 初始化旋转(随机四元数)
    new_quats = random_quat_tensor(num_points).to(device)
    
    # 修改SH系数初始化部分
    colors = new_instance["colors"].to(device)
    logger.debug(f"原始颜色数据形状: {colors.shape}")  # 应为[num_points, 3]

    fused_color = RGB2SH(colors)  # [num_points, 3]
    logger.debug(f"转换后SH系数形状: {fused_color.shape}")  # 应为[num_points, 3]
    
    dim_sh = num_sh_bases(smpl_nodes.sh_degree)
    logger.info(f"计算得到的dim_sh: {dim_sh}")
    
    if smpl_nodes.sh_degree > 0:
        shs = torch.zeros((num_points, dim_sh, 3), device=device)
        # 检查维度是否匹配
        if shs[:, 0, :3].shape != fused_color.shape:
            raise ValueError(f"SH系数维度不匹配: {shs[:,0,:3].shape} vs {fused_color.shape}")
        shs[:, 0, :3] = fused_color
    else:
        shs = torch.logit(colors.unsqueeze(1), eps=1e-10)
    
    # 记录设备信息
    logger.info(f"现有instances_quats设备: {smpl_nodes.instances_quats.device}")
    logger.info(f"新global_orient设备: {global_orient.device}")
    logger.info(f"新body_pose设备: {body_pose.device}")
    logger.info(f"新trans设备: {trans.device}")
    
    # 修改点云属性扩展部分
    with torch.no_grad():
        # 确保_features_dc和_features_rest维度正确
        if not hasattr(smpl_nodes, '_features_dc'):
            smpl_nodes._features_dc = Parameter(torch.zeros(0, 3, device=device))
        
        # 处理_features_rest的维度问题
        if not hasattr(smpl_nodes, '_features_rest'):
            rest_dims = (dim_sh-1)*3
            smpl_nodes._features_rest = Parameter(torch.zeros(0, rest_dims, device=device))
        elif smpl_nodes._features_rest.dim() != 2:  # 确保是2D张量
            smpl_nodes._features_rest = Parameter(smpl_nodes._features_rest.reshape(-1, (dim_sh-1)*3))
            
        # 扩展点云属性
        smpl_nodes._features_dc = Parameter(torch.cat([
            smpl_nodes._features_dc, 
            shs[:, 0, :]  # [num_points, 3]
        ], dim=0))
        
        # 处理_features_rest的拼接
        rest_features = shs[:, 1:, :].reshape(num_points, -1)  # [num_points, (dim_sh-1)*3]
        smpl_nodes._features_rest = Parameter(torch.cat([
            smpl_nodes._features_rest,
            rest_features
        ], dim=0))
        
        # 扩展姿态参数（注意维度处理）
        global_orient_expanded = global_orient.unsqueeze(1).to(device)  # [num_frames,1,4]
        body_pose_expanded = body_pose.unsqueeze(1).to(device)  # [num_frames,1,23,4]
        trans_expanded = trans.unsqueeze(1).to(device)  # [num_frames,1,3]
        
        logger.info(f"展开后global_orient形状: {global_orient_expanded.shape}")
        logger.info(f"展开后body_pose形状: {body_pose_expanded.shape}")
        logger.info(f"展开后trans形状: {trans_expanded.shape}")
        
        smpl_nodes.instances_quats = Parameter(
            torch.cat([
                smpl_nodes.instances_quats, 
                global_orient_expanded
            ], dim=1)
        )
        
        smpl_nodes.smpl_qauts = Parameter(
            torch.cat([
                smpl_nodes.smpl_qauts,
                body_pose_expanded
            ], dim=1)
        )
        
        smpl_nodes.instances_trans = Parameter(
            torch.cat([
                smpl_nodes.instances_trans,
                trans_expanded
            ], dim=1)
        )
        
        # 扩展形状参数
        if hasattr(smpl_nodes, 'betas'):
            smpl_nodes.betas = torch.cat([
                smpl_nodes.betas,
                new_instance["betas"].unsqueeze(0).to(device)
            ], dim=0)


    
    logger.info(f"成功创建新SMPL实例，ID: {new_instance_id}")
    return new_instance_id
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
        
        # 1. 转换SMPLX到SMPL
        smplx_model_path = "smpl_models/smplx/SMPLX_NEUTRAL.npz"
        smpl_model_path = "smpl_models/smpl/SMPL_NEUTRAL.pkl"
        smpl_params = load_smplx_to_smpl(args.new_npz_path)
        
        # 2. 加载PLY并创建实例
        new_instance = load_ply_and_create_instance(args.new_ply_path, smpl_params)
        
        # 3. 创建新实例
        with torch.no_grad():
            new_instance_id = create_smpl_instance(
                trainer, 
                new_instance,
                keep_translation=args.keep_translation,
                keep_global_rot=args.keep_global_rot
            )
        logger.info(f"新创建的实例ID: {new_instance_id}")
    
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