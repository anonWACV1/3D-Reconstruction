#!/usr/bin/env python3
"""
独立的相机pose可视化程序
读取指定目录下的json和npz文件，可视化相机轨迹和坐标

使用方法:
python pose_visualizer.py --input_dir /path/to/pose/files --output_dir /path/to/output
python pose_visualizer.py --npz_file /path/to/poses.npz --output_dir /path/to/output
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import glob
from typing import Dict, List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PoseVisualizer:
    """相机pose可视化器"""
    
    def __init__(self, output_dir: str = "./pose_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib中文字体（如果需要）
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.dpi'] = 300
        
    def load_npz_file(self, npz_path: str) -> Dict:
        """加载NPZ文件"""
        logger.info(f"Loading NPZ file: {npz_path}")
        
        data = np.load(npz_path, allow_pickle=True)
        poses_dict = {}
        for key in data.keys():
            poses_dict[key] = data[key]
            
        logger.info(f"Loaded {len(poses_dict['frame_indices'])} frames")
        return poses_dict
    
    def load_json_file(self, json_path: str) -> Dict:
        """加载JSON文件"""
        logger.info(f"Loading JSON file: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        logger.info(f"Loaded JSON with {data.get('total_frames', 'unknown')} frames")
        return data
    
    def find_pose_files(self, input_dir: str) -> tuple:
        """在目录中查找pose文件"""
        input_path = Path(input_dir)
        
        # 查找NPZ文件
        npz_files = list(input_path.glob("*.npz"))
        json_files = list(input_path.glob("*summary.json"))
        
        logger.info(f"Found {len(npz_files)} NPZ files and {len(json_files)} JSON files")
        
        return npz_files, json_files
    
    def extract_trajectory_data(self, poses_dict: Dict) -> Dict:
        """提取轨迹数据"""
        positions = np.array(poses_dict['camera_positions'])
        rotations = poses_dict['camera_rotations']
        cam_names = poses_dict['cam_names']
        frame_indices = poses_dict['frame_indices']
        
        # 计算欧拉角
        euler_angles = []
        for rot in rotations:
            euler = R.from_matrix(rot).as_euler('xyz', degrees=True)
            euler_angles.append(euler)
        euler_angles = np.array(euler_angles)
        
        # 计算速度
        distances = []
        if len(positions) > 1:
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        
        # 计算旋转速度
        rotation_speeds = []
        if len(euler_angles) > 1:
            rotation_speeds = np.linalg.norm(np.diff(euler_angles, axis=0), axis=1)
        
        return {
            'positions': positions,
            'rotations': rotations,
            'euler_angles': euler_angles,
            'cam_names': cam_names,
            'frame_indices': frame_indices,
            'distances': distances,
            'rotation_speeds': rotation_speeds
        }
    
    def create_camera_color_map(self, cam_names: List[str]) -> Dict:
        """为不同相机创建颜色映射"""
        unique_cameras = list(set(cam_names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cameras)))
        return {cam: colors[i] for i, cam in enumerate(unique_cameras)}
    
    def plot_3d_trajectory(self, traj_data: Dict, save_path: str):
        """绘制3D轨迹图"""
        fig = plt.figure(figsize=(16, 12))
        
        positions = traj_data['positions']
        rotations = traj_data['rotations']
        cam_names = traj_data['cam_names']
        frame_indices = traj_data['frame_indices']
        
        camera_color_map = self.create_camera_color_map(cam_names)
        unique_cameras = list(camera_color_map.keys())
        
        # 主3D图
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 绘制不同相机的轨迹
        for cam_name in unique_cameras:
            mask = np.array(cam_names) == cam_name
            cam_positions = positions[mask]
            if len(cam_positions) > 0:
                ax1.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 
                        'o-', color=camera_color_map[cam_name], markersize=3, linewidth=1.5, 
                        label=cam_name, alpha=0.8)
        
        # 添加相机朝向箭头
        arrow_step = max(1, len(positions)//20)
        for i in range(0, len(positions), arrow_step):
            pos = positions[i]
            rot = rotations[i]
            forward = rot @ np.array([0, 0, -1]) * 2.0
            ax1.quiver(pos[0], pos[1], pos[2], 
                      forward[0], forward[1], forward[2], 
                      color='red', alpha=0.6, arrow_length_ratio=0.1)
        
        # 标记起点和终点
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                   c='green', s=100, marker='o', label='Start', alpha=0.9)
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                   c='red', s=100, marker='s', label='End', alpha=0.9)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Camera Trajectory with Orientation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # XY平面图
        ax2 = fig.add_subplot(222)
        for cam_name in unique_cameras:
            mask = np.array(cam_names) == cam_name
            cam_positions = positions[mask]
            if len(cam_positions) > 0:
                ax2.plot(cam_positions[:, 0], cam_positions[:, 1], 
                        'o-', color=camera_color_map[cam_name], markersize=4, 
                        label=cam_name, alpha=0.8, linewidth=2)
        
        ax2.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
        ax2.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Plane View (Bird\'s Eye)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Z高度变化
        ax3 = fig.add_subplot(223)
        for cam_name in unique_cameras:
            mask = np.array(cam_names) == cam_name
            cam_positions = positions[mask]
            cam_frames = np.array(frame_indices)[mask]
            if len(cam_positions) > 0:
                ax3.plot(cam_frames, cam_positions[:, 2], 
                        'o-', color=camera_color_map[cam_name], markersize=3, 
                        label=cam_name, alpha=0.8, linewidth=2)
        
        ax3.set_xlabel('Frame Index')
        ax3.set_ylabel('Z Height (m)')
        ax3.set_title('Height Variation over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 移动速度分析
        ax4 = fig.add_subplot(224)
        if len(traj_data['distances']) > 0:
            distances = traj_data['distances']
            ax4.plot(frame_indices[1:], distances, 'b-o', markersize=2, alpha=0.7, linewidth=1.5)
            ax4.set_xlabel('Frame Index')
            ax4.set_ylabel('Inter-frame Distance (m)')
            ax4.set_title('Movement Speed Analysis')
            ax4.grid(True, alpha=0.3)
            
            mean_speed = distances.mean()
            max_speed = distances.max()
            ax4.axhline(y=mean_speed, color='r', linestyle='--', alpha=0.7, 
                       label=f'Mean: {mean_speed:.3f}m')
            ax4.axhline(y=max_speed, color='orange', linestyle='--', alpha=0.7, 
                       label=f'Max: {max_speed:.3f}m')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"3D trajectory plot saved to {save_path}")
        plt.close()
    
    def plot_detailed_orientation(self, traj_data: Dict, save_path: str):
        """绘制详细的相机朝向图"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        positions = traj_data['positions']
        rotations = traj_data['rotations']
        cam_names = traj_data['cam_names']
        
        camera_color_map = self.create_camera_color_map(cam_names)
        unique_cameras = list(camera_color_map.keys())
        
        # 绘制轨迹
        for cam_name in unique_cameras:
            mask = np.array(cam_names) == cam_name
            cam_positions = positions[mask]
            if len(cam_positions) > 0:
                ax.plot(cam_positions[:, 0], cam_positions[:, 1], 
                       'o-', color=camera_color_map[cam_name], markersize=5, 
                       label=f'{cam_name} trajectory', alpha=0.8, linewidth=2)
        
        # 绘制相机朝向
        arrow_step = max(1, len(positions)//25)
        for i in range(0, len(positions), arrow_step):
            pos = positions[i]
            rot = rotations[i]
            cam_name = cam_names[i]
            
            # 前方向（红色）
            forward = rot @ np.array([0, 0, -1]) * 3.0
            ax.arrow(pos[0], pos[1], forward[0], forward[1], 
                    head_width=0.8, head_length=0.5, fc='red', ec='red', alpha=0.7)
            
            # 右方向（蓝色）
            right = rot @ np.array([1, 0, 0]) * 1.5
            ax.arrow(pos[0], pos[1], right[0], right[1], 
                    head_width=0.5, head_length=0.3, fc='blue', ec='blue', alpha=0.5)
        
        # 标记关键点
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, 
               label='Start', zorder=10, markeredgecolor='black', markeredgewidth=2)
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=12, 
               label='End', zorder=10, markeredgecolor='black', markeredgewidth=2)
        
        # 添加一些关键帧标注
        key_frames = np.linspace(0, len(positions)-1, 5, dtype=int)
        for idx in key_frames:
            ax.annotate(f'F{traj_data["frame_indices"][idx]}', 
                       (positions[idx, 0], positions[idx, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Camera Trajectory with Detailed Orientation\n'
                    'Red arrows: Forward direction, Blue arrows: Right direction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Detailed orientation plot saved to {save_path}")
        plt.close()
    
    def plot_rotation_analysis(self, traj_data: Dict, save_path: str):
        """绘制旋转分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        euler_angles = traj_data['euler_angles']
        frame_indices = traj_data['frame_indices']
        cam_names = traj_data['cam_names']
        rotation_speeds = traj_data['rotation_speeds']
        
        camera_color_map = self.create_camera_color_map(cam_names)
        unique_cameras = list(camera_color_map.keys())
        
        # 绘制Roll, Pitch, Yaw
        rotation_labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
        for i, (ax, label) in enumerate(zip(axes.flat[:3], rotation_labels)):
            for cam_name in unique_cameras:
                mask = np.array(cam_names) == cam_name
                cam_angles = euler_angles[mask, i]
                cam_frames = np.array(frame_indices)[mask]
                if len(cam_angles) > 0:
                    ax.plot(cam_frames, cam_angles, 'o-', 
                           color=camera_color_map[cam_name], markersize=3, 
                           label=cam_name, alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Frame Index')
            ax.set_ylabel(f'{label} (degrees)')
            ax.set_title(f'Camera {label} over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 旋转速度分析
        if len(rotation_speeds) > 0:
            axes[1, 1].plot(frame_indices[1:], rotation_speeds, 
                           'purple', marker='o', markersize=2, alpha=0.7, linewidth=2)
            axes[1, 1].set_xlabel('Frame Index')
            axes[1, 1].set_ylabel('Rotation Speed (deg/frame)')
            axes[1, 1].set_title('Camera Rotation Speed')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加统计线
            mean_rot_speed = rotation_speeds.mean()
            max_rot_speed = rotation_speeds.max()
            axes[1, 1].axhline(y=mean_rot_speed, color='r', linestyle='--', alpha=0.7,
                              label=f'Mean: {mean_rot_speed:.2f}°/frame')
            axes[1, 1].axhline(y=max_rot_speed, color='orange', linestyle='--', alpha=0.7,
                              label=f'Max: {max_rot_speed:.2f}°/frame')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Rotation analysis plot saved to {save_path}")
        plt.close()
    
    def plot_coordinate_statistics(self, traj_data: Dict, save_path: str):
        """绘制坐标统计图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        positions = traj_data['positions']
        frame_indices = traj_data['frame_indices']
        cam_names = traj_data['cam_names']
        
        camera_color_map = self.create_camera_color_map(cam_names)
        unique_cameras = list(camera_color_map.keys())
        
        # X, Y, Z坐标随时间变化
        coord_labels = ['X Coordinate', 'Y Coordinate', 'Z Coordinate']
        for i, (ax, label) in enumerate(zip(axes[0], coord_labels)):
            for cam_name in unique_cameras:
                mask = np.array(cam_names) == cam_name
                cam_positions = positions[mask]
                cam_frames = np.array(frame_indices)[mask]
                if len(cam_positions) > 0:
                    ax.plot(cam_frames, cam_positions[:, i], 'o-',
                           color=camera_color_map[cam_name], markersize=3,
                           label=cam_name, alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Frame Index')
            ax.set_ylabel(f'{label} (m)')
            ax.set_title(f'{label} over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 坐标分布直方图
        coord_names = ['X', 'Y', 'Z']
        for i, (ax, coord_name) in enumerate(zip(axes[1], coord_names)):
            ax.hist(positions[:, i], bins=30, alpha=0.7, color=plt.cm.viridis(i/3))
            ax.set_xlabel(f'{coord_name} Coordinate (m)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{coord_name} Coordinate Distribution')
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = positions[:, i].mean()
            std_val = positions[:, i].std()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8,
                      label=f'Mean: {mean_val:.2f}m')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.6,
                      label=f'+1σ: {mean_val+std_val:.2f}m')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.6,
                      label=f'-1σ: {mean_val-std_val:.2f}m')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Coordinate statistics plot saved to {save_path}")
        plt.close()
    
    def save_detailed_report(self, poses_dict: Dict, traj_data: Dict, save_path: str):
        """保存详细的文本报告"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CAMERA POSE TRAJECTORY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本信息
            f.write("BASIC INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total frames: {len(traj_data['frame_indices'])}\n")
            f.write(f"Frame range: {min(traj_data['frame_indices'])} - {max(traj_data['frame_indices'])}\n")
            
            unique_cameras = list(set(traj_data['cam_names']))
            f.write(f"Camera types: {unique_cameras}\n")
            
            # 处理cam_ids
            cam_ids_clean = []
            for cam_id in poses_dict['cam_ids']:
                if isinstance(cam_id, np.ndarray):
                    cam_ids_clean.append(int(cam_id.item()))
                else:
                    cam_ids_clean.append(int(cam_id))
            f.write(f"Camera IDs: {list(set(cam_ids_clean))}\n\n")
            
            # 轨迹统计
            positions = traj_data['positions']
            f.write("TRAJECTORY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Position ranges:\n")
            f.write(f"  X: [{positions[:, 0].min():.6f}, {positions[:, 0].max():.6f}] m\n")
            f.write(f"  Y: [{positions[:, 1].min():.6f}, {positions[:, 1].max():.6f}] m\n")
            f.write(f"  Z: [{positions[:, 2].min():.6f}, {positions[:, 2].max():.6f}] m\n")
            
            f.write(f"Position means:\n")
            f.write(f"  X: {positions[:, 0].mean():.6f} ± {positions[:, 0].std():.6f} m\n")
            f.write(f"  Y: {positions[:, 1].mean():.6f} ± {positions[:, 1].std():.6f} m\n")
            f.write(f"  Z: {positions[:, 2].mean():.6f} ± {positions[:, 2].std():.6f} m\n")
            
            if len(traj_data['distances']) > 0:
                distances = traj_data['distances']
                total_distance = distances.sum()
                f.write(f"Total trajectory length: {total_distance:.6f} m\n")
                f.write(f"Average inter-frame distance: {distances.mean():.6f} m\n")
                f.write(f"Maximum inter-frame distance: {distances.max():.6f} m\n")
                f.write(f"Minimum inter-frame distance: {distances.min():.6f} m\n\n")
            
            # 旋转统计
            euler_angles = traj_data['euler_angles']
            f.write("ROTATION STATISTICS:\n")
            f.write("-" * 40 + "\n")
            rotation_names = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
            for i, name in enumerate(rotation_names):
                angles = euler_angles[:, i]
                f.write(f"{name}:\n")
                f.write(f"  Range: [{angles.min():.3f}, {angles.max():.3f}] degrees\n")
                f.write(f"  Mean: {angles.mean():.3f} ± {angles.std():.3f} degrees\n")
            
            if len(traj_data['rotation_speeds']) > 0:
                rot_speeds = traj_data['rotation_speeds']
                f.write(f"Rotation speed statistics:\n")
                f.write(f"  Mean: {rot_speeds.mean():.3f} deg/frame\n")
                f.write(f"  Max: {rot_speeds.max():.3f} deg/frame\n\n")
            
            # 相机内参
            if 'camera_intrinsics' in poses_dict:
                intrinsics = poses_dict['camera_intrinsics'][0]
                f.write("CAMERA INTRINSICS:\n")
                f.write("-" * 40 + "\n")
                fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                f.write(f"Focal length: fx={fx:.6f}, fy={fy:.6f}\n")
                f.write(f"Principal point: cx={cx:.6f}, cy={cy:.6f}\n")
                
                height = poses_dict['heights'][0]
                width = poses_dict['widths'][0]
                if isinstance(height, np.ndarray):
                    height = height.item()
                if isinstance(width, np.ndarray):
                    width = width.item()
                f.write(f"Image size: {int(width)}x{int(height)}\n\n")
            
            # 每个相机的详细统计
            f.write("PER-CAMERA STATISTICS:\n")
            f.write("-" * 40 + "\n")
            for cam_name in unique_cameras:
                mask = np.array(traj_data['cam_names']) == cam_name
                cam_positions = positions[mask]
                cam_frames = np.array(traj_data['frame_indices'])[mask]
                
                f.write(f"{cam_name}:\n")
                f.write(f"  Frames: {len(cam_frames)} ({cam_frames.min()}-{cam_frames.max()})\n")
                if len(cam_positions) > 1:
                    cam_distances = np.linalg.norm(np.diff(cam_positions, axis=0), axis=1)
                    f.write(f"  Trajectory length: {cam_distances.sum():.6f} m\n")
                    f.write(f"  Avg movement: {cam_distances.mean():.6f} m/frame\n")
                f.write(f"  Position range:\n")
                f.write(f"    X: [{cam_positions[:, 0].min():.3f}, {cam_positions[:, 0].max():.3f}]\n")
                f.write(f"    Y: [{cam_positions[:, 1].min():.3f}, {cam_positions[:, 1].max():.3f}]\n")
                f.write(f"    Z: [{cam_positions[:, 2].min():.3f}, {cam_positions[:, 2].max():.3f}]\n\n")
        
        logger.info(f"Detailed report saved to {save_path}")
    
    def visualize_poses(self, poses_dict: Dict, prefix: str = ""):
        """完整的pose可视化流程"""
        logger.info(f"Starting pose visualization with prefix: {prefix}")
        
        # 提取轨迹数据
        traj_data = self.extract_trajectory_data(poses_dict)
        
        # 创建保存路径
        prefix = f"{prefix}_" if prefix else ""
        
        # 生成各种可视化图
        self.plot_3d_trajectory(
            traj_data, 
            self.output_dir / f"{prefix}3d_trajectory.png"
        )
        
        self.plot_detailed_orientation(
            traj_data, 
            self.output_dir / f"{prefix}detailed_orientation.png"
        )
        
        self.plot_rotation_analysis(
            traj_data, 
            self.output_dir / f"{prefix}rotation_analysis.png"
        )
        
        self.plot_coordinate_statistics(
            traj_data, 
            self.output_dir / f"{prefix}coordinate_statistics.png"
        )
        
        # 保存详细报告
        self.save_detailed_report(
            poses_dict, 
            traj_data, 
            self.output_dir / f"{prefix}detailed_report.txt"
        )
        
        logger.info(f"Pose visualization completed. Files saved to {self.output_dir}")
        
        return traj_data
    
    def process_directory(self, input_dir: str):
        """处理整个目录中的pose文件"""
        npz_files, json_files = self.find_pose_files(input_dir)
        
        if not npz_files:
            logger.warning(f"No NPZ files found in {input_dir}")
            return
        
        # 处理每个NPZ文件
        for npz_file in npz_files:
            logger.info(f"Processing {npz_file.name}")
            
            try:
                poses_dict = self.load_npz_file(str(npz_file))
                
                # 使用文件名作为前缀
                prefix = npz_file.stem
                self.visualize_poses(poses_dict, prefix)
                
            except Exception as e:
                logger.error(f"Error processing {npz_file}: {e}")
    
    def process_single_file(self, npz_path: str):
        """处理单个NPZ文件"""
        try:
            poses_dict = self.load_npz_file(npz_path)
            
            # 使用文件名作为前缀
            prefix = Path(npz_path).stem
            self.visualize_poses(poses_dict, prefix)
            
        except Exception as e:
            logger.error(f"Error processing {npz_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Camera pose visualization tool")
    
    # 输入选项
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", type=str, 
                      help="Directory containing NPZ and JSON files")
    group.add_argument("--npz_file", type=str, 
                      help="Single NPZ file to process")
    
    # 输出选项
    parser.add_argument("--output_dir", type=str, default="./pose_analysis",
                       help="Output directory for visualizations and reports")
    
    # 可视化选项
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved images")
    parser.add_argument("--figsize", type=str, default="12,8",
                       help="Figure size as 'width,height' in inches")
    parser.add_argument("--style", type=str, default="default",
                       choices=["default", "seaborn", "ggplot", "bmh"],
                       help="Matplotlib style")
    
    # 过滤选项
    parser.add_argument("--camera_filter", type=str, nargs="+",
                       help="Only process specific cameras (e.g., CAM_FRONT CAM_BACK)")
    parser.add_argument("--frame_range", type=str,
                       help="Frame range to process as 'start:end' or 'start:end:step'")
    
    # 输出格式选项
    parser.add_argument("--export_formats", type=str, nargs="+", 
                       default=["png"], choices=["png", "pdf", "svg"],
                       help="Export formats for plots")
    parser.add_argument("--save_data", action="store_true",
                       help="Save processed trajectory data as CSV")
    
    args = parser.parse_args()
    
    # 设置matplotlib样式
    if args.style != "default":
        plt.style.use(args.style)
    
    # 解析figsize
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
        plt.rcParams['figure.figsize'] = figsize
    except:
        logger.warning("Invalid figsize format, using default")
    
    # 设置DPI
    plt.rcParams['figure.dpi'] = args.dpi
    
    # 创建可视化器
    visualizer = PoseVisualizer(args.output_dir)
    
    # 处理文件
    if args.input_dir:
        logger.info(f"Processing directory: {args.input_dir}")
        visualizer.process_directory(args.input_dir)
    else:
        logger.info(f"Processing file: {args.npz_file}")
        visualizer.process_single_file(args.npz_file)
    
    logger.info("Processing completed!")


class EnhancedPoseVisualizer(PoseVisualizer):
    """增强版pose可视化器，支持更多功能"""
    
    def __init__(self, output_dir: str = "./pose_analysis", export_formats: list = ["png"]):
        super().__init__(output_dir)
        self.export_formats = export_formats
    
    def filter_data(self, poses_dict: Dict, camera_filter: list = None, frame_range: str = None):
        """过滤数据"""
        if camera_filter is None and frame_range is None:
            return poses_dict
        
        # 创建索引掩码
        mask = np.ones(len(poses_dict['frame_indices']), dtype=bool)
        
        # 相机过滤
        if camera_filter:
            cam_mask = np.isin(poses_dict['cam_names'], camera_filter)
            mask &= cam_mask
            logger.info(f"Filtered to cameras: {camera_filter}")
        
        # 帧范围过滤
        if frame_range:
            try:
                if ':' in frame_range:
                    parts = frame_range.split(':')
                    if len(parts) == 2:
                        start, end = map(int, parts)
                        step = 1
                    else:
                        start, end, step = map(int, parts)
                else:
                    start = int(frame_range)
                    end = start + 1
                    step = 1
                
                frame_indices = np.array(poses_dict['frame_indices'])
                frame_mask = (frame_indices >= start) & (frame_indices < end)
                if step > 1:
                    # 每step帧取一个
                    selected_indices = np.arange(len(frame_indices))[frame_mask][::step]
                    frame_mask = np.zeros_like(frame_mask)
                    frame_mask[selected_indices] = True
                
                mask &= frame_mask
                logger.info(f"Filtered to frame range: {start}:{end}:{step}")
                
            except Exception as e:
                logger.warning(f"Invalid frame range format: {frame_range}, error: {e}")
        
        # 应用过滤
        if mask.sum() == 0:
            logger.warning("No data left after filtering!")
            return poses_dict
        
        filtered_dict = {}
        for key, value in poses_dict.items():
            if isinstance(value, (list, np.ndarray)) and len(value) == len(mask):
                if isinstance(value, list):
                    filtered_dict[key] = [value[i] for i in range(len(value)) if mask[i]]
                else:
                    filtered_dict[key] = value[mask]
            else:
                filtered_dict[key] = value
        
        logger.info(f"Filtered data: {mask.sum()}/{len(mask)} frames retained")
        return filtered_dict
    
    def save_trajectory_data(self, traj_data: Dict, save_path: str):
        """保存轨迹数据为CSV"""
        import pandas as pd
        
        data_rows = []
        for i, (frame_idx, cam_name, pos, euler) in enumerate(zip(
            traj_data['frame_indices'],
            traj_data['cam_names'], 
            traj_data['positions'],
            traj_data['euler_angles']
        )):
            row = {
                'frame_idx': frame_idx,
                'cam_name': cam_name,
                'x': pos[0],
                'y': pos[1], 
                'z': pos[2],
                'roll': euler[0],
                'pitch': euler[1],
                'yaw': euler[2]
            }
            
            # 添加移动距离（如果不是第一帧）
            if i > 0:
                prev_pos = traj_data['positions'][i-1]
                row['distance_from_prev'] = np.linalg.norm(pos - prev_pos)
            else:
                row['distance_from_prev'] = 0.0
            
            # 添加旋转速度（如果不是第一帧）
            if i > 0:
                prev_euler = traj_data['euler_angles'][i-1]
                row['rotation_speed'] = np.linalg.norm(euler - prev_euler)
            else:
                row['rotation_speed'] = 0.0
                
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        df.to_csv(save_path, index=False)
        logger.info(f"Trajectory data saved to {save_path}")
    
    def save_plot(self, fig, base_path: str):
        """保存图片为多种格式"""
        for fmt in self.export_formats:
            save_path = f"{base_path}.{fmt}"
            fig.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        logger.info(f"Plot saved in formats: {self.export_formats}")
    
    def plot_interactive_trajectory(self, traj_data: Dict, save_path: str):
        """创建交互式轨迹图（需要plotly）"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            positions = traj_data['positions']
            cam_names = traj_data['cam_names']
            frame_indices = traj_data['frame_indices']
            
            # 创建3D轨迹图
            fig = go.Figure()
            
            # 为每个相机添加轨迹
            unique_cameras = list(set(cam_names))
            colors = px.colors.qualitative.Set1[:len(unique_cameras)]
            
            for i, cam_name in enumerate(unique_cameras):
                mask = np.array(cam_names) == cam_name
                cam_positions = positions[mask]
                cam_frames = np.array(frame_indices)[mask]
                
                if len(cam_positions) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=cam_positions[:, 0],
                        y=cam_positions[:, 1], 
                        z=cam_positions[:, 2],
                        mode='lines+markers',
                        marker=dict(size=4, color=colors[i % len(colors)]),
                        line=dict(width=3, color=colors[i % len(colors)]),
                        name=cam_name,
                        text=[f'Frame {f}' for f in cam_frames],
                        hovertemplate=
                        '<b>%{text}</b><br>' +
                        'X: %{x:.3f}m<br>' +
                        'Y: %{y:.3f}m<br>' +
                        'Z: %{z:.3f}m<br>' +
                        '<extra></extra>'
                    ))
            
            # 标记起点和终点
            fig.add_trace(go.Scatter3d(
                x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
                mode='markers',
                marker=dict(size=10, color='green', symbol='circle'),
                name='Start'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
                mode='markers',
                marker=dict(size=10, color='red', symbol='square'),
                name='End'
            ))
            
            fig.update_layout(
                title='Interactive 3D Camera Trajectory',
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)',
                    aspectmode='data'
                ),
                width=1000,
                height=700
            )
            
            # 保存为HTML
            html_path = save_path.replace('.png', '.html')
            fig.write_html(html_path)
            logger.info(f"Interactive plot saved to {html_path}")
            
        except ImportError:
            logger.warning("Plotly not available, skipping interactive plot")
    
    def create_summary_dashboard(self, traj_data: Dict, save_path: str):
        """创建汇总仪表盘"""
        fig = plt.figure(figsize=(20, 12))
        
        positions = traj_data['positions']
        euler_angles = traj_data['euler_angles']
        frame_indices = traj_data['frame_indices']
        cam_names = traj_data['cam_names']
        
        # 创建6个子图的网格
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 3D轨迹
        ax1 = fig.add_subplot(gs[0, :2], projection='3d')
        unique_cameras = list(set(cam_names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cameras)))
        camera_color_map = {cam: colors[i] for i, cam in enumerate(unique_cameras)}
        
        for cam_name in unique_cameras:
            mask = np.array(cam_names) == cam_name
            cam_positions = positions[mask]
            if len(cam_positions) > 0:
                ax1.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 
                        'o-', color=camera_color_map[cam_name], markersize=2, 
                        linewidth=1, label=cam_name, alpha=0.8)
        
        ax1.set_title('3D Trajectory')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y') 
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # 2. XY平面图
        ax2 = fig.add_subplot(gs[0, 2:])
        for cam_name in unique_cameras:
            mask = np.array(cam_names) == cam_name
            cam_positions = positions[mask]
            if len(cam_positions) > 0:
                ax2.plot(cam_positions[:, 0], cam_positions[:, 1], 
                        'o-', color=camera_color_map[cam_name], markersize=3, 
                        label=cam_name, alpha=0.8)
        
        ax2.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
        ax2.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
        ax2.set_title('XY Plane View')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.legend()
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        
        # 3-5. 坐标变化
        coord_names = ['X', 'Y', 'Z']
        for i, coord_name in enumerate(coord_names):
            ax = fig.add_subplot(gs[1, i])
            for cam_name in unique_cameras:
                mask = np.array(cam_names) == cam_name
                cam_positions = positions[mask]
                cam_frames = np.array(frame_indices)[mask]
                if len(cam_positions) > 0:
                    ax.plot(cam_frames, cam_positions[:, i], 'o-',
                           color=camera_color_map[cam_name], markersize=2, 
                           label=cam_name, alpha=0.8)
            ax.set_title(f'{coord_name} Coordinate')
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'{coord_name} (m)')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        # 6. 速度分析
        ax6 = fig.add_subplot(gs[1, 3])
        if len(traj_data['distances']) > 0:
            distances = traj_data['distances']
            ax6.plot(frame_indices[1:], distances, 'b-', alpha=0.7, linewidth=1)
            ax6.set_title('Movement Speed')
            ax6.set_xlabel('Frame')
            ax6.set_ylabel('Distance (m)')
            ax6.grid(True, alpha=0.3)
        
        # 7-9. 旋转角度
        rotation_names = ['Roll', 'Pitch', 'Yaw']
        for i, rot_name in enumerate(rotation_names):
            ax = fig.add_subplot(gs[2, i])
            for cam_name in unique_cameras:
                mask = np.array(cam_names) == cam_name
                cam_angles = euler_angles[mask, i]
                cam_frames = np.array(frame_indices)[mask]
                if len(cam_angles) > 0:
                    ax.plot(cam_frames, cam_angles, 'o-',
                           color=camera_color_map[cam_name], markersize=2, 
                           label=cam_name, alpha=0.8)
            ax.set_title(f'{rot_name} Angle')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Angle (deg)')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        # 10. 统计信息文本
        ax10 = fig.add_subplot(gs[2, 3])
        ax10.axis('off')
        
        # 计算统计信息
        total_frames = len(frame_indices)
        total_distance = traj_data['distances'].sum() if len(traj_data['distances']) > 0 else 0
        
        stats_text = f"""
TRAJECTORY STATISTICS

Total Frames: {total_frames}
Cameras: {len(unique_cameras)}
Total Distance: {total_distance:.2f} m

Position Ranges:
X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m
Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m  
Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m

Rotation Ranges:
Roll: [{euler_angles[:, 0].min():.1f}, {euler_angles[:, 0].max():.1f}]°
Pitch: [{euler_angles[:, 1].min():.1f}, {euler_angles[:, 1].max():.1f}]°
Yaw: [{euler_angles[:, 2].min():.1f}, {euler_angles[:, 2].max():.1f}]°
        """.strip()
        
        ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, 
                 verticalalignment='top', fontfamily='monospace', fontsize=9)
        
        fig.suptitle('Camera Trajectory Analysis Dashboard', fontsize=16, fontweight='bold')
        
        self.save_plot(fig, save_path)
    
    def visualize_poses(self, poses_dict: Dict, prefix: str = "", 
                       camera_filter: list = None, frame_range: str = None,
                       save_data: bool = False):
        """增强版pose可视化"""
        logger.info(f"Starting enhanced pose visualization with prefix: {prefix}")
        
        # 过滤数据
        filtered_poses = self.filter_data(poses_dict, camera_filter, frame_range)
        
        # 提取轨迹数据
        traj_data = self.extract_trajectory_data(filtered_poses)
        
        # 创建保存路径
        prefix = f"{prefix}_" if prefix else ""
        
        # 生成汇总仪表盘
        dashboard_path = str(self.output_dir / f"{prefix}dashboard")
        self.create_summary_dashboard(traj_data, dashboard_path)
        
        # 生成各种详细可视化图
        self.plot_3d_trajectory(
            traj_data, 
            str(self.output_dir / f"{prefix}3d_trajectory")
        )
        
        self.plot_detailed_orientation(
            traj_data, 
            str(self.output_dir / f"{prefix}detailed_orientation")
        )
        
        self.plot_rotation_analysis(
            traj_data, 
            str(self.output_dir / f"{prefix}rotation_analysis")
        )
        
        self.plot_coordinate_statistics(
            traj_data, 
            str(self.output_dir / f"{prefix}coordinate_statistics")
        )
        
        # 生成交互式图（如果可用）
        interactive_path = str(self.output_dir / f"{prefix}interactive_trajectory.png")
        self.plot_interactive_trajectory(traj_data, interactive_path)
        
        # 保存详细报告
        self.save_detailed_report(
            filtered_poses, 
            traj_data, 
            self.output_dir / f"{prefix}detailed_report.txt"
        )
        
        # 保存数据（如果需要）
        if save_data:
            try:
                self.save_trajectory_data(
                    traj_data,
                    self.output_dir / f"{prefix}trajectory_data.csv"
                )
            except ImportError:
                logger.warning("Pandas not available, skipping CSV export")
        
        logger.info(f"Enhanced pose visualization completed. Files saved to {self.output_dir}")
        
        return traj_data


if __name__ == "__main__":
    main()