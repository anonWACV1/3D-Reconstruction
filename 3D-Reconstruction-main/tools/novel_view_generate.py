import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from typing import List, Dict, Optional
from tqdm import tqdm
from PIL import Image
import imageio
from scipy.spatial.transform import Rotation as R  # 导入旋转库

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import render_images, save_videos, render_novel_views


def render_novel_views_as_images(
    trainer,
    render_data: list,
    save_dir: str,
    single_frame: bool = True
) -> None:
    """
    执行渲染并将结果保存为单独的图像文件。
    
    Args:
        trainer: 包含渲染方法的训练器对象
        render_data (list): 列表，每个元素包含渲染单帧所需的数据
        save_dir (str): 保存输出图像的目录
        single_frame (bool): 如果为True，只保存一个图像（第一帧）；如果为False，保存所有帧
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, frame_data in enumerate(render_data):
            # 如果只需要一个图像且不是第一帧，跳过
            if single_frame and i > 0:
                break
                
            # 将数据移到GPU
            for key, value in frame_data["cam_infos"].items():
                if isinstance(value, torch.Tensor):
                    frame_data["cam_infos"][key] = value.cuda(non_blocking=True)
            for key, value in frame_data["image_infos"].items():
                if isinstance(value, torch.Tensor):
                    frame_data["image_infos"][key] = value.cuda(non_blocking=True)
            
            # 执行渲染
            outputs = trainer(
                image_infos=frame_data["image_infos"],
                camera_infos=frame_data["cam_infos"],
                novel_view=True
            )
            
            # 提取RGB图像并裁剪
            rgb = outputs["rgb"].cpu().numpy().clip(0.0, 1.0)
            
            # 转换为uint8并保存为图像
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            
            # 构建图像文件名
            if single_frame:
                img_path = os.path.join(save_dir, "image.png")
            else:
                img_path = os.path.join(save_dir, f"frame_{i:04d}.png")
            
            # 保存图像
            Image.fromarray(rgb_uint8).save(img_path)
            print(f"图像已保存到 {img_path}")
            
        if single_frame:
            print(f"已保存单张图像到 {save_dir}")
        else:
            print(f"已保存 {len(render_data)} 帧图像到 {save_dir}")


def generate_fixed_offset_trajectories(
    dataset,
    frame_idx,
    target_frames,
    translation_offset,
    rotation_offset
):
    """
    生成固定偏移的轨迹
    
    Args:
        dataset: 数据集实例
        frame_idx: 当前帧索引
        target_frames: 目标帧数
        translation_offset: 平移偏移量 [x, y, z]
        rotation_offset: 旋转偏移量 [pitch, yaw, roll]
        
    Returns:
        torch.Tensor: 生成的轨迹
    """
    # 获取当前帧的相机位姿
    per_cam_poses = {}
    for cam_id in dataset.pixel_source.camera_list:
        per_cam_poses[cam_id] = dataset.pixel_source.camera_data[cam_id].cam_to_worlds
    
    # 使用当前帧的相机位姿
    frame_poses = {}
    for cam_id, poses in per_cam_poses.items():
        # 确保索引在范围内
        safe_idx = min(frame_idx, len(poses) - 1)
        frame_poses[cam_id] = poses[safe_idx:safe_idx+1]
    
    # 获取设备信息
    device = next(iter(per_cam_poses.values())).device
    
    try:
        # 尝试从utils.trajectory_utils导入fixed_offset_trajectory函数
        from utils.camera import fixed_offset_trajectory
    except ImportError:
        try:
            from utils.camera import fixed_offset_trajectory
        except ImportError:
            # 如果导入失败，则在当前模块中定义该函数
            def fixed_offset_trajectory(
                dataset_type: str,
                per_cam_poses: Dict[int, torch.Tensor],
                original_frames: int,
                target_frames: int,
                translation_offset: list = [0.0, 1.0, 0.0],
                rotation_offset: list = [0.0, 0.0, 0.0],
            ) -> torch.Tensor:
                """
                生成相对于前视相机的固定偏移轨迹

                Args:
                    dataset_type: 数据集类型
                    per_cam_poses: 每个相机的位姿
                    original_frames: 原始帧数
                    target_frames: 目标帧数
                    translation_offset: [x, y, z] 平移偏移量（米）
                    rotation_offset: [pitch, yaw, roll] 旋转偏移量（度）
                """
                assert 0 in per_cam_poses.keys(), "需要前视中心相机（ID 0）"

                # 获取设备信息
                device = per_cam_poses[0].device

                # 转换偏移量为张量
                trans_offset = torch.tensor(translation_offset, device=device, dtype=torch.float32)
                rot_offset = torch.tensor(rotation_offset, device=device, dtype=torch.float32)

                # 生成关键帧（使用原始相机位姿）
                key_poses = per_cam_poses[0][:: max(1, original_frames // 4)]

                def convert_to_tensor(data, device):
                    return torch.tensor(data, device=device, dtype=torch.float32)

                # 应用偏移量
                modified_poses = []
                for pose in key_poses:
                    # 创建新位姿矩阵
                    new_pose = torch.eye(4, device=device)

                    rot_matrix = R.from_euler(
                        "xyz", rot_offset.cpu().numpy(), degrees=True
                    ).as_matrix()
                    rot_matrix = rot_matrix.astype(np.float32)

                    # 保持矩阵乘法数据类型一致
                    new_rot = pose[:3, :3] @ convert_to_tensor(rot_matrix, device)

                    # 确保平移偏移量数据类型正确
                    trans_offset_tensor = convert_to_tensor(translation_offset, device)
                    offset_trans = pose[:3, :3] @ trans_offset_tensor
                    new_trans = pose[:3, 3] + offset_trans

                    new_pose[:3, :3] = new_rot
                    new_pose[:3, 3] = new_trans

                    modified_poses.append(new_pose)

                # 插值生成所需帧数的轨迹
                return interpolate_poses(torch.stack(modified_poses), target_frames)

            # 定义插值函数
            def interpolate_poses(poses, n_samples):
                """
                对位姿进行插值，生成平滑轨迹
                
                Args:
                    poses: 关键帧位姿，形状为 [N, 4, 4]
                    n_samples: 插值后的总帧数
                    
                Returns:
                    torch.Tensor: 插值后的轨迹，形状为 [n_samples, 4, 4]
                """
                device = poses.device
                n_poses = poses.shape[0]
                
                # 创建一个线性插值的参数t
                t_original = torch.linspace(0, 1, n_poses, device=device)
                t_interp = torch.linspace(0, 1, n_samples, device=device)
                
                # 分别处理旋转和平移
                positions = poses[:, :3, 3]
                rotations = poses[:, :3, :3]
                
                # 线性插值平移
                interp_positions = torch.zeros((n_samples, 3), device=device)
                for i in range(3):
                    # 手动实现线性插值
                    for j in range(n_samples):
                        t = t_interp[j]
                        # 找到t所在的区间
                        idx = (t_original <= t).sum() - 1
                        idx = max(0, min(idx, n_poses - 2))
                        
                        # 计算局部插值参数
                        alpha = (t - t_original[idx]) / (t_original[idx + 1] - t_original[idx])
                        interp_positions[j, i] = positions[idx, i] + alpha * (positions[idx + 1, i] - positions[idx, i])          
                # 对旋转矩阵进行SLERP插值
                interp_rotations = torch.zeros((n_samples, 3, 3), device=device)
                for i in range(n_samples):
                    t = t_interp[i]
                    # 找到t所在的区间
                    idx = (t_original <= t).sum() - 1
                    idx = max(0, min(idx, n_poses - 2))
                    
                    # 计算局部插值参数
                    alpha = (t - t_original[idx]) / (t_original[idx + 1] - t_original[idx])
                    
                    # 使用简单的线性插值替代SLERP
                    rot1 = rotations[idx]
                    rot2 = rotations[idx + 1]
                    interp_rot = rot1 + alpha * (rot2 - rot1)
                    
                    # 正交化以确保是有效的旋转矩阵
                    u, _, v = torch.svd(interp_rot)
                    interp_rotations[i] = u @ v.t()
                
                # 组合成完整的位姿矩阵
                interp_poses = torch.eye(4, device=device).unsqueeze(0).repeat(n_samples, 1, 1)
                interp_poses[:, :3, :3] = interp_rotations
                interp_poses[:, :3, 3] = interp_positions
                
                return interp_poses
    
    # 调用fixed_offset_trajectory生成轨迹
    traj = fixed_offset_trajectory(
        dataset_type=dataset.type,
        per_cam_poses=frame_poses,
        original_frames=1,  # 只使用一帧
        target_frames=target_frames,
        translation_offset=translation_offset,
        rotation_offset=rotation_offset
    )
    
    return traj


def generate_novel_views(
    dataset: DrivingDataset,
    trainer: BasicTrainer,
    output_dir: str,
    frame_indices: List[int],
    offsets: List[Dict] = None,
    target_frames: int = 30,
    fps: int = 30,
    save_video: bool = False,
    save_image: bool = True
):
    """
    批量生成多视角新视图

    Args:
        dataset: 数据集实例
        trainer: 训练器实例
        output_dir: 输出目录
        frame_indices: 需要处理的帧索引列表
        offsets: 每个视角的偏移配置列表
        target_frames: 每个轨迹的帧数
        fps: 视频帧率
        save_video: 是否保存视频
        save_image: 是否保存图像
    """
    # 如果没有提供偏移量，使用默认配置
    if offsets is None:
        offsets = [
            {
                "name": "NEW_RGB_1",
                "translation": [1.5, 0, 1.6],
                "rotation": [0, 0, 0]
            },
            {
                "name": "NEW_RGB_2",
                "translation": [1.0, -3.2, 1.6],
                "rotation": [0, 0, 0]
            },
            {
                "name": "NEW_RGB_3",
                "translation": [1.0, -3.2, 1.6],
                "rotation": [0, -30, 0]
            },
            {
                "name": "NEW_RGB_4",
                "translation": [1.0, 3.2, 1.6],
                "rotation": [0, 0, 0]
            },
            {
                "name": "NEW_RGB_5",
                "translation": [1.0, 3.2, 1.6],
                "rotation": [0, 30, 0]
            }
        ]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有需要处理的帧
    for idx in tqdm(frame_indices, desc="处理帧"):
        try:
            # 为每个偏移量生成新视图
            for offset in offsets:
                try:
                    # 生成轨迹数据
                    print(f"为帧 {idx} 生成 {offset['name']} 的轨迹...")
                    
                    # 直接生成固定偏移轨迹
                    traj = generate_fixed_offset_trajectories(
                        dataset=dataset,
                        frame_idx=idx,
                        target_frames=target_frames,
                        translation_offset=offset["translation"],
                        rotation_offset=offset["rotation"]
                    )
                    
                    if traj is None:
                        print(f"警告：为帧 {idx} 和偏移 {offset['name']} 生成轨迹失败")
                        continue
                    
                    # 准备渲染数据
                    if hasattr(dataset, "prepare_novel_view_render_data"):
                        # 使用数据集的方法
                        render_data = dataset.prepare_novel_view_render_data(traj)
                    else:
                        # 使用像素源的方法
                        render_data = dataset.pixel_source.prepare_novel_view_render_data(
                            dataset.type, traj
                        )
                    
                    # 创建存储目录
                    view_dir = os.path.join(output_dir, f"frame_{idx:04d}_{offset['name']}")
                    os.makedirs(view_dir, exist_ok=True)
                    
                    # 如果需要保存视频
                    if save_video:
                        # 渲染并保存视频
                        video_path = os.path.join(view_dir, "video.mp4")
                        print(f"渲染视频到 {video_path}...")
                        
                        # 使用render_novel_views函数渲染并保存视频
                        render_novel_views(
                            trainer=trainer,
                            render_data=render_data,
                            save_path=video_path,
                            fps=fps
                        )
                        print(f"成功为帧 {idx} 渲染 {offset['name']} 视角的视频")
                    
                    # 如果需要保存图像
                    if save_image:
                        print(f"渲染图像到 {view_dir}...")
                        
                        # 使用自定义函数渲染并保存图像
                        render_novel_views_as_images(
                            trainer=trainer,
                            render_data=render_data,
                            save_dir=view_dir,
                            single_frame=True
                        )
                        print(f"成功为帧 {idx} 渲染 {offset['name']} 视角的图像")
                    
                except Exception as e:
                    print(f"为帧 {idx} 生成 {offset['name']} 视角时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                
        except Exception as e:
            print(f"处理帧 {idx} 时发生错误: {str(e)}")
            continue


def load_config_and_model(checkpoint_path):
    """加载配置和模型的健壮方法"""
    try:
        # 尝试从检查点目录加载配置
        log_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(log_dir, "config.yaml")
        
        if not os.path.exists(config_path):
            # 尝试其他可能的配置文件名
            alternative_paths = [
                os.path.join(log_dir, "config.yml"),
                os.path.join(log_dir, "cfg.yaml"),
                os.path.join(log_dir, "model_config.yaml")
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    config_path = alt_path
                    break
        
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            print(f"从 {config_path} 加载配置成功")
        else:
            # 如果找不到配置文件，尝试从检查点加载
            print("在检查点目录中找不到配置文件，尝试从检查点加载")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "config" in checkpoint:
                cfg = checkpoint["config"]
            else:
                raise FileNotFoundError(f"无法找到配置文件，也无法从检查点加载配置")
    except Exception as e:
        print(f"加载配置时出错: {str(e)}")
        raise
        
    return cfg


def main():
    parser = argparse.ArgumentParser("生成新视图")
    parser.add_argument(
        "--resume_from",
        type=str,
        required=True,
        help="要恢复的检查点路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="渲染图像的输出目录 (默认为检查点目录下的 novel_view_image)",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="起始帧索引")
    parser.add_argument("--end_frame", type=int, default=100, help="结束帧索引")
    parser.add_argument("--stride", type=int, default=10, help="帧采样步长")
    parser.add_argument("--fps", type=int, default=30, help="输出视频的帧率")
    parser.add_argument("--target_frames", type=int, default=30, help="每个视角的目标帧数")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备 (cuda/cpu)")
    parser.add_argument("--save_video", action="store_true", help="保存视频")
    parser.add_argument("--save_image", action="store_true", default=True, help="保存图像")
    args = parser.parse_args()

    # 设置默认输出目录（如果未指定）
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.resume_from)
        args.output_dir = os.path.join(checkpoint_dir, "novel_view_image")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置和模型
    print(f"从 {args.resume_from} 加载模型...")
    cfg = load_config_and_model(args.resume_from)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"使用设备: {device}")

    try:
        # 初始化数据集
        print("初始化数据集...")
        if "data" in cfg:
            dataset = DrivingDataset(data_cfg=cfg.data)
        elif "dataset" in cfg:
            dataset = DrivingDataset(data_cfg=cfg.dataset)
        else:
            raise ValueError("配置中找不到数据集配置")
        
        # 按照提供的方法初始化训练器
        print("初始化训练器...")
        trainer_type = cfg.trainer.type
        
        # 使用import_str导入训练器类
        trainer_cls = import_str(trainer_type)
        
        # 准备训练器初始化参数
        trainer_kwargs = dict(cfg.trainer)
        trainer_kwargs["num_timesteps"] = dataset.num_img_timesteps
        trainer_kwargs["model_config"] = cfg.model
        trainer_kwargs["num_train_images"] = len(dataset.train_image_set) if hasattr(dataset, "train_image_set") else 0
        trainer_kwargs["num_full_images"] = len(dataset.full_image_set) if hasattr(dataset, "full_image_set") else 0
        trainer_kwargs["test_set_indices"] = dataset.test_timesteps if hasattr(dataset, "test_timesteps") else None
        
        # 获取场景边界框
        scene_aabb = None
        if hasattr(dataset, "get_aabb"):
            try:
                scene_aabb = dataset.get_aabb().reshape(2, 3)
                trainer_kwargs["scene_aabb"] = scene_aabb
            except Exception as e:
                print(f"警告：获取场景AABB失败: {str(e)}")
        
        # 设置设备
        trainer_kwargs["device"] = device
        
        # 初始化训练器实例
        print("训练器参数:", list(trainer_kwargs.keys()))
        trainer = trainer_cls(**trainer_kwargs)
        
        # 从检查点恢复
        print(f"从检查点恢复模型: {args.resume_from}")
        trainer.resume_from_checkpoint(args.resume_from, load_only_model=True)
        # 设置为评估模式
        trainer.eval()

        # 生成帧索引列表
        frame_indices = list(range(args.start_frame, args.end_frame, args.stride))
        print(f"将处理 {len(frame_indices)} 个帧，从 {args.start_frame} 到 {args.end_frame}，步长为 {args.stride}")

        # 执行批量生成
        generate_novel_views(
            dataset=dataset,
            trainer=trainer,
            output_dir=args.output_dir,
            frame_indices=frame_indices,
            target_frames=args.target_frames,
            fps=args.fps,
            save_video=args.save_video,
            save_image=args.save_image
        )
        
        print(f"所有新视图已成功渲染并保存到 {args.output_dir}")
        
    except Exception as e:
        print(f"运行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()