#!/usr/bin/env python3
"""
Instance可视化工具
读取保存的instance数据，并可视化其位置信息、序列长度等

使用示例:
export PYTHONPATH=$(pwd)
python tools/visualize_instance.py \
    --instance_file ./saved_instances/smpl_instance_0.pkl \
    --output_dir ./instance_visualization

# 批量可视化多个实例
python tools/visualize_instance.py \
    --instance_files ./saved_instances/*.pkl \
    --output_dir ./instance_visualization

# 可视化特定类型的实例
python tools/visualize_instance.py \
    --instance_file ./saved_instances/smpl_instance_0.pkl \
    --instance_type smpl \
    --output_dir ./instance_visualization
"""

import os
import sys
import pickle
import logging
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch

# 确保能够导入项目模块
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入场景编辑模块
try:
    from scene_editing.scene_editing import load_instance_data
except ImportError:
    def load_instance_data(file_path: str) -> Dict[str, Any]:
        """加载实例数据的替代实现"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InstanceVisualizer:
    """Instance数据可视化器"""
    
    def __init__(self, output_dir: str = "./instance_visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        
    def load_instance_data(self, file_path: str) -> Dict[str, Any]:
        """加载实例数据"""
        logger.info(f"加载实例数据: {file_path}")
        
        try:
            instance_data = load_instance_data(file_path)
            logger.info(f"成功加载实例数据，文件大小: {Path(file_path).stat().st_size / 1024 / 1024:.2f} MB")
            return instance_data
        except Exception as e:
            logger.error(f"加载实例数据失败: {str(e)}")
            raise
    
    def extract_motion_info(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取运动信息"""
        motion_info = {}
        
        # 获取运动数据
        motion = instance_data.get("motion", {})
        
        # 提取位置信息
        if "instances_trans" in motion:
            trans = motion["instances_trans"]
            if isinstance(trans, torch.Tensor):
                trans = trans.cpu().numpy()
            
            # 处理不同维度的位置数据
            if trans.ndim == 1:
                # 单个位置点
                trans = trans.reshape(1, -1)
            elif trans.ndim == 2 and trans.shape[1] == 3:
                # 已经是正确的形状 (num_frames, 3)
                pass
            elif trans.ndim == 3:
                # 可能是 (num_frames, num_instances, 3)，取第一个实例
                if trans.shape[1] == 1:
                    trans = trans.squeeze(1)
                else:
                    logger.warning(f"多实例位置数据，取第一个实例。形状: {trans.shape}")
                    trans = trans[:, 0, :]
            
            motion_info["translation"] = trans
            motion_info["num_frames"] = len(trans)
            
        # 提取旋转信息
        if "instances_quats" in motion:
            quats = motion["instances_quats"]
            if isinstance(quats, torch.Tensor):
                quats = quats.cpu().numpy()
            
            # 处理不同维度的四元数数据
            if quats.ndim == 1:
                # 单个四元数
                quats = quats.reshape(1, -1)
            elif quats.ndim == 2:
                # 检查是否是正确的形状
                if quats.shape[1] == 4:
                    # 正确的形状 (num_frames, 4)
                    pass
                elif quats.shape[1] == 1:
                    # 可能是 (num_frames, 1) 需要进一步处理
                    logger.warning(f"四元数数据形状异常: {quats.shape}")
                    quats = None
                else:
                    logger.warning(f"未知的四元数形状: {quats.shape}")
                    quats = None
            elif quats.ndim == 3:
                # 可能是 (num_frames, num_instances, 4) 或 (num_frames, 1, 4)
                if quats.shape[1] == 1:
                    quats = quats.squeeze(1)  # 去掉实例维度
                else:
                    logger.warning(f"多实例四元数数据，取第一个实例。形状: {quats.shape}")
                    quats = quats[:, 0, :]
            
            if quats is not None:
                motion_info["quaternions"] = quats
            
        # 提取SMPL特有的旋转信息
        if "smpl_qauts" in motion:
            smpl_quats = motion["smpl_qauts"]
            if isinstance(smpl_quats, torch.Tensor):
                smpl_quats = smpl_quats.cpu().numpy()
            
            # 处理SMPL四元数数据 (通常是 (num_frames, 23, 4) 或 (num_frames, num_instances, 23, 4))
            if smpl_quats.ndim == 3:
                # (num_frames, 23, 4) - 正确的形状
                motion_info["smpl_quaternions"] = smpl_quats
            elif smpl_quats.ndim == 4:
                # (num_frames, num_instances, 23, 4) - 取第一个实例
                logger.warning(f"多实例SMPL四元数数据，取第一个实例。形状: {smpl_quats.shape}")
                motion_info["smpl_quaternions"] = smpl_quats[:, 0, :, :]
            else:
                logger.warning(f"SMPL四元数形状异常: {smpl_quats.shape}")
            
        # 提取帧有效性信息
        if "instances_fv" in motion:
            fv = motion["instances_fv"]
            if isinstance(fv, torch.Tensor):
                fv = fv.cpu().numpy()
            
            # 处理不同维度的帧有效性数据
            if fv.ndim == 1:
                # 正确的形状 (num_frames,)
                motion_info["frame_validity"] = fv
            elif fv.ndim == 2:
                # 可能是 (num_frames, num_instances)
                if fv.shape[1] == 1:
                    motion_info["frame_validity"] = fv.squeeze(1)
                else:
                    logger.warning(f"多实例帧有效性数据，取第一个实例。形状: {fv.shape}")
                    motion_info["frame_validity"] = fv[:, 0]
            
        return motion_info
    
    def extract_geometry_info(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取几何信息"""
        geometry_info = {}
        
        # 获取几何数据
        geometry = instance_data.get("geometry", {})
        
        # 提取点坐标
        if "_means" in geometry:
            means = geometry["_means"]
            if isinstance(means, torch.Tensor):
                means = means.cpu().numpy()
            geometry_info["points"] = means
            geometry_info["num_points"] = len(means)
            
        # 提取点大小
        if "_scales" in geometry:
            scales = geometry["_scales"]
            if isinstance(scales, torch.Tensor):
                scales = scales.cpu().numpy()
            geometry_info["scales"] = scales
            
        # 提取点不透明度
        if "_opacities" in geometry:
            opacities = geometry["_opacities"]
            if isinstance(opacities, torch.Tensor):
                opacities = opacities.cpu().numpy()
            geometry_info["opacities"] = opacities
            
        return geometry_info
    
    def plot_trajectory_3d(self, motion_info: Dict[str, Any], save_path: str):
        """绘制3D轨迹图"""
        if "translation" not in motion_info:
            logger.warning("没有找到位置信息，跳过3D轨迹绘制")
            return
            
        translation = motion_info["translation"]
        
        # 确保translation是正确的形状
        if translation.ndim != 2 or translation.shape[1] != 3:
            logger.warning(f"位置数据形状不正确: {translation.shape}，跳过3D轨迹绘制")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹
        ax.plot(translation[:, 0], translation[:, 1], translation[:, 2], 
                'b-', linewidth=2, alpha=0.7, label='轨迹')
        
        # 标记起始点和结束点
        ax.scatter(translation[0, 0], translation[0, 1], translation[0, 2], 
                  c='green', s=100, marker='o', label='起始点')
        ax.scatter(translation[-1, 0], translation[-1, 1], translation[-1, 2], 
                  c='red', s=100, marker='s', label='结束点')
        
        # 设置坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Instance 3D轨迹')
        ax.legend()
        
        # 设置等比例坐标轴
        try:
            max_range = np.array([translation[:, 0].max()-translation[:, 0].min(),
                                 translation[:, 1].max()-translation[:, 1].min(),
                                 translation[:, 2].max()-translation[:, 2].min()]).max() / 2.0
            mid_x = (translation[:, 0].max()+translation[:, 0].min()) * 0.5
            mid_y = (translation[:, 1].max()+translation[:, 1].min()) * 0.5
            mid_z = (translation[:, 2].max()+translation[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        except:
            # 如果设置等比例坐标轴失败，使用默认设置
            logger.warning("设置等比例坐标轴失败，使用默认设置")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"3D轨迹图保存到: {save_path}")
    
    def plot_trajectory_2d(self, motion_info: Dict[str, Any], save_path: str):
        """绘制2D轨迹图"""
        if "translation" not in motion_info:
            logger.warning("没有找到位置信息，跳过2D轨迹绘制")
            return
            
        translation = motion_info["translation"]
        
        # 确保translation是正确的形状
        if translation.ndim != 2 or translation.shape[1] != 3:
            logger.warning(f"位置数据形状不正确: {translation.shape}，跳过2D轨迹绘制")
            return
            
        num_frames = len(translation)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Instance 2D轨迹分析', fontsize=16)
        
        # XY平面图
        ax1 = axes[0, 0]
        ax1.plot(translation[:, 0], translation[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax1.scatter(translation[0, 0], translation[0, 1], c='green', s=100, marker='o', label='起始点')
        ax1.scatter(translation[-1, 0], translation[-1, 1], c='red', s=100, marker='s', label='结束点')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('XY平面轨迹')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 尝试设置等比例坐标轴
        try:
            ax1.axis('equal')
        except:
            logger.warning("设置等比例坐标轴失败")
        
        # 时间序列图
        ax2 = axes[0, 1]
        frames = np.arange(num_frames)
        ax2.plot(frames, translation[:, 0], 'r-', label='X', linewidth=2)
        ax2.plot(frames, translation[:, 1], 'g-', label='Y', linewidth=2)
        ax2.plot(frames, translation[:, 2], 'b-', label='Z', linewidth=2)
        ax2.set_xlabel('帧数')
        ax2.set_ylabel('位置 (m)')
        ax2.set_title('位置时间序列')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 速度分析
        ax3 = axes[1, 0]
        if num_frames > 1:
            try:
                velocity = np.diff(translation, axis=0)
                speed = np.linalg.norm(velocity, axis=1)
                ax3.plot(frames[1:], speed, 'purple', linewidth=2)
                ax3.set_xlabel('帧数')
                ax3.set_ylabel('速度 (m/frame)')
                ax3.set_title('速度分析')
                ax3.grid(True, alpha=0.3)
            except:
                ax3.text(0.5, 0.5, '速度计算失败', transform=ax3.transAxes, 
                        ha='center', va='center', fontsize=12)
                ax3.set_title('速度分析')
        else:
            ax3.text(0.5, 0.5, '数据不足，无法计算速度', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('速度分析')
        
        # 帧有效性（如果有）
        ax4 = axes[1, 1]
        if "frame_validity" in motion_info:
            fv = motion_info["frame_validity"]
            if len(fv) == num_frames:
                ax4.plot(frames, fv.astype(int), 'orange', linewidth=2, marker='o', markersize=3)
                ax4.set_xlabel('帧数')
                ax4.set_ylabel('帧有效性')
                ax4.set_title('帧有效性')
                ax4.set_ylim(-0.1, 1.1)
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '帧有效性数据长度不匹配', transform=ax4.transAxes, 
                        ha='center', va='center', fontsize=12)
                ax4.set_title('帧有效性')
        else:
            ax4.text(0.5, 0.5, '无帧有效性数据', transform=ax4.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('帧有效性')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"2D轨迹图保存到: {save_path}")
    
    def plot_rotation_analysis(self, motion_info: Dict[str, Any], save_path: str):
        """绘制旋转分析图"""
        if "quaternions" not in motion_info:
            logger.warning("没有找到旋转信息，跳过旋转分析")
            return
            
        quaternions = motion_info["quaternions"]
        num_frames = len(quaternions)
        
        # 检查四元数维度并处理
        if quaternions.ndim == 1:
            logger.warning("四元数数据是1D，可能是单个四元数，跳过旋转分析")
            return
        elif quaternions.ndim == 2 and quaternions.shape[1] == 1:
            logger.warning("四元数数据维度不正确，跳过旋转分析")
            return
        elif quaternions.ndim == 3:
            # 如果是3D，取第一个实例或者reshape
            if quaternions.shape[1] == 1:
                quaternions = quaternions.squeeze(1)  # 去掉实例维度
            else:
                logger.warning("多实例四元数数据，取第一个实例")
                quaternions = quaternions[:, 0, :]
        
        # 确保四元数是正确的形状 (num_frames, 4)
        if quaternions.shape[1] != 4:
            logger.warning(f"四元数形状不正确: {quaternions.shape}，跳过旋转分析")
            return
        
        # 转换四元数为欧拉角
        def quat_to_euler(q):
            """四元数转欧拉角 (roll, pitch, yaw)"""
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            pitch = np.arcsin(np.clip(sinp, -1, 1))
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            return np.column_stack([roll, pitch, yaw])
        
        euler_angles = quat_to_euler(quaternions)
        euler_degrees = np.degrees(euler_angles)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Instance 旋转分析', fontsize=16)
        
        # 四元数时间序列
        ax1 = axes[0, 0]
        frames = np.arange(num_frames)
        ax1.plot(frames, quaternions[:, 0], 'r-', label='w', linewidth=2)
        ax1.plot(frames, quaternions[:, 1], 'g-', label='x', linewidth=2)
        ax1.plot(frames, quaternions[:, 2], 'b-', label='y', linewidth=2)
        ax1.plot(frames, quaternions[:, 3], 'm-', label='z', linewidth=2)
        ax1.set_xlabel('帧数')
        ax1.set_ylabel('四元数分量')
        ax1.set_title('四元数时间序列')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 欧拉角时间序列
        ax2 = axes[0, 1]
        ax2.plot(frames, euler_degrees[:, 0], 'r-', label='Roll', linewidth=2)
        ax2.plot(frames, euler_degrees[:, 1], 'g-', label='Pitch', linewidth=2)
        ax2.plot(frames, euler_degrees[:, 2], 'b-', label='Yaw', linewidth=2)
        ax2.set_xlabel('帧数')
        ax2.set_ylabel('欧拉角 (度)')
        ax2.set_title('欧拉角时间序列')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 角速度分析
        ax3 = axes[1, 0]
        if num_frames > 1:
            angular_velocity = np.diff(euler_angles, axis=0)
            angular_speed = np.linalg.norm(angular_velocity, axis=1)
            ax3.plot(frames[1:], np.degrees(angular_speed), 'purple', linewidth=2)
            ax3.set_xlabel('帧数')
            ax3.set_ylabel('角速度 (度/frame)')
            ax3.set_title('角速度分析')
            ax3.grid(True, alpha=0.3)
        
        # 旋转统计
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
旋转统计信息:

四元数范围:
  w: [{quaternions[:, 0].min():.3f}, {quaternions[:, 0].max():.3f}]
  x: [{quaternions[:, 1].min():.3f}, {quaternions[:, 1].max():.3f}]
  y: [{quaternions[:, 2].min():.3f}, {quaternions[:, 2].max():.3f}]
  z: [{quaternions[:, 3].min():.3f}, {quaternions[:, 3].max():.3f}]

欧拉角范围 (度):
  Roll: [{euler_degrees[:, 0].min():.1f}, {euler_degrees[:, 0].max():.1f}]
  Pitch: [{euler_degrees[:, 1].min():.1f}, {euler_degrees[:, 1].max():.1f}]
  Yaw: [{euler_degrees[:, 2].min():.1f}, {euler_degrees[:, 2].max():.1f}]
        """.strip()
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"旋转分析图保存到: {save_path}")
    
    def plot_geometry_analysis(self, geometry_info: Dict[str, Any], save_path: str):
        """绘制几何分析图"""
        if "points" not in geometry_info:
            logger.warning("没有找到几何信息，跳过几何分析")
            return
            
        points = geometry_info["points"]
        num_points = len(points)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Instance 几何分析', fontsize=16)
        
        # 点云分布（3D）
        ax1 = axes[0, 0]
        ax1.remove()
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        # 随机采样点以提高可视化性能
        if num_points > 5000:
            indices = np.random.choice(num_points, 5000, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points
            
        ax1.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
                   c='blue', s=1, alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'点云分布 (显示 {len(sample_points)}/{num_points} 点)')
        
        # 点云分布（2D投影）
        ax2 = axes[0, 1]
        ax2.scatter(sample_points[:, 0], sample_points[:, 1], c='blue', s=1, alpha=0.6)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('XY平面投影')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # 点规模分布
        ax3 = axes[1, 0]
        if "scales" in geometry_info:
            scales = geometry_info["scales"]
            if scales.ndim > 1:
                scales = np.linalg.norm(scales, axis=1)
            ax3.hist(scales, bins=50, alpha=0.7, color='green')
            ax3.set_xlabel('点规模')
            ax3.set_ylabel('频率')
            ax3.set_title('点规模分布')
        else:
            ax3.text(0.5, 0.5, '无规模信息', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('点规模分布')
        
        # 不透明度分布
        ax4 = axes[1, 1]
        if "opacities" in geometry_info:
            opacities = geometry_info["opacities"]
            if opacities.ndim > 1:
                opacities = opacities.flatten()
            ax4.hist(opacities, bins=50, alpha=0.7, color='orange')
            ax4.set_xlabel('不透明度')
            ax4.set_ylabel('频率')
            ax4.set_title('不透明度分布')
        else:
            ax4.text(0.5, 0.5, '无不透明度信息', transform=ax4.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('不透明度分布')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"几何分析图保存到: {save_path}")
    
    def generate_summary_report(self, instance_data: Dict[str, Any], 
                              motion_info: Dict[str, Any], 
                              geometry_info: Dict[str, Any],
                              save_path: str):
        """生成详细报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("Instance 数据分析报告")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 元数据信息
        if "metadata" in instance_data:
            metadata = instance_data["metadata"]
            report_lines.append("元数据信息:")
            report_lines.append("-" * 30)
            for key, value in metadata.items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # 几何信息
        if geometry_info:
            report_lines.append("几何信息:")
            report_lines.append("-" * 30)
            if "num_points" in geometry_info:
                report_lines.append(f"  点数量: {geometry_info['num_points']}")
            
            if "points" in geometry_info:
                points = geometry_info["points"]
                report_lines.append(f"  点坐标范围:")
                report_lines.append(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
                report_lines.append(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
                report_lines.append(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
            
            if "scales" in geometry_info:
                scales = geometry_info["scales"]
                if scales.ndim > 1:
                    scales = np.linalg.norm(scales, axis=1)
                report_lines.append(f"  点规模范围: [{scales.min():.3f}, {scales.max():.3f}]")
            
            if "opacities" in geometry_info:
                opacities = geometry_info["opacities"]
                if opacities.ndim > 1:
                    opacities = opacities.flatten()
                report_lines.append(f"  不透明度范围: [{opacities.min():.3f}, {opacities.max():.3f}]")
            
            report_lines.append("")
        
        # 运动信息
        if motion_info:
            report_lines.append("运动信息:")
            report_lines.append("-" * 30)
            if "num_frames" in motion_info:
                report_lines.append(f"  帧数: {motion_info['num_frames']}")
            
            if "translation" in motion_info:
                translation = motion_info["translation"]
                report_lines.append(f"  位置信息:")
                report_lines.append(f"    起始位置: [{translation[0, 0]:.3f}, {translation[0, 1]:.3f}, {translation[0, 2]:.3f}]")
                report_lines.append(f"    结束位置: [{translation[-1, 0]:.3f}, {translation[-1, 1]:.3f}, {translation[-1, 2]:.3f}]")
                
                if len(translation) > 1:
                    total_distance = np.sum(np.linalg.norm(np.diff(translation, axis=0), axis=1))
                    report_lines.append(f"    总移动距离: {total_distance:.3f} m")
                    report_lines.append(f"    平均移动速度: {total_distance / (len(translation) - 1):.3f} m/frame")
            
            if "quaternions" in motion_info:
                quaternions = motion_info["quaternions"]
                report_lines.append(f"  旋转信息:")
                report_lines.append(f"    四元数形状: {quaternions.shape}")
                report_lines.append(f"    起始四元数: [{quaternions[0, 0]:.3f}, {quaternions[0, 1]:.3f}, {quaternions[0, 2]:.3f}, {quaternions[0, 3]:.3f}]")
                report_lines.append(f"    结束四元数: [{quaternions[-1, 0]:.3f}, {quaternions[-1, 1]:.3f}, {quaternions[-1, 2]:.3f}, {quaternions[-1, 3]:.3f}]")
            
            if "frame_validity" in motion_info:
                fv = motion_info["frame_validity"]
                valid_frames = np.sum(fv)
                report_lines.append(f"  帧有效性: {valid_frames}/{len(fv)} 帧有效")
            
            report_lines.append("")
        
        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"详细报告保存到: {save_path}")
    
    def visualize_instance(self, instance_file: str, instance_type: str = None):
        """可视化单个实例"""
        logger.info(f"=" * 60)
        logger.info(f"开始可视化实例: {instance_file}")
        logger.info(f"=" * 60)
        
        # 加载实例数据
        instance_data = self.load_instance_data(instance_file)
        
        # 提取信息
        motion_info = self.extract_motion_info(instance_data)
        geometry_info = self.extract_geometry_info(instance_data)
        
        # 获取文件名用作前缀
        file_name = Path(instance_file).stem
        
        # 生成可视化图
        if motion_info:
            # 3D轨迹
            trajectory_3d_path = self.output_dir / f"{file_name}_trajectory_3d.png"
            self.plot_trajectory_3d(motion_info, str(trajectory_3d_path))
            
            # 2D轨迹分析
            trajectory_2d_path = self.output_dir / f"{file_name}_trajectory_2d.png"
            self.plot_trajectory_2d(motion_info, str(trajectory_2d_path))
            
            # 旋转分析
            rotation_path = self.output_dir / f"{file_name}_rotation_analysis.png"
            self.plot_rotation_analysis(motion_info, str(rotation_path))
        
        if geometry_info:
            # 几何分析
            geometry_path = self.output_dir / f"{file_name}_geometry_analysis.png"
            self.plot_geometry_analysis(geometry_info, str(geometry_path))
        
        # 生成详细报告
        report_path = self.output_dir / f"{file_name}_report.txt"
        self.generate_summary_report(instance_data, motion_info, geometry_info, str(report_path))
        
        logger.info(f"实例可视化完成！结果保存到: {self.output_dir}")
    
    def visualize_multiple_instances(self, instance_files: List[str]):
        """可视化多个实例"""
        logger.info(f"=" * 60)
        logger.info(f"开始批量可视化 {len(instance_files)} 个实例")
        logger.info(f"=" * 60)
        
        # 收集所有实例的信息用于比较
        all_motion_info = []
        all_geometry_info = []
        file_names = []
        
        for instance_file in instance_files:
            try:
                logger.info(f"处理实例: {instance_file}")
                
                # 加载实例数据
                instance_data = self.load_instance_data(instance_file)
                
                # 提取信息
                motion_info = self.extract_motion_info(instance_data)
                geometry_info = self.extract_geometry_info(instance_data)
                
                # 获取文件名
                file_name = Path(instance_file).stem
                file_names.append(file_name)
                
                # 可视化单个实例
                self.visualize_instance(instance_file)
                
                # 收集信息用于比较
                all_motion_info.append(motion_info)
                all_geometry_info.append(geometry_info)
                
            except Exception as e:
                logger.error(f"处理实例 {instance_file} 时出错: {str(e)}")
                continue
        
        # 生成比较图
        if len(all_motion_info) > 1:
            self.plot_instances_comparison(all_motion_info, all_geometry_info, file_names)
        
        logger.info(f"批量可视化完成！结果保存到: {self.output_dir}")
    
    def plot_instances_comparison(self, all_motion_info: List[Dict], 
                                 all_geometry_info: List[Dict], 
                                 file_names: List[str]):
        """绘制多个实例的比较图"""
        logger.info("生成实例比较图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('多个Instance比较分析', fontsize=16)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(file_names)))
        
        # 比较轨迹长度
        ax1 = axes[0, 0]
        trajectory_lengths = []
        point_counts = []
        
        for i, (motion_info, geometry_info, name) in enumerate(zip(all_motion_info, all_geometry_info, file_names)):
            # 计算轨迹长度
            if "translation" in motion_info:
                translation = motion_info["translation"]
                if len(translation) > 1:
                    distances = np.linalg.norm(np.diff(translation, axis=0), axis=1)
                    total_distance = np.sum(distances)
                    trajectory_lengths.append(total_distance)
                else:
                    trajectory_lengths.append(0)
            else:
                trajectory_lengths.append(0)
            
            # 统计点数
            if "num_points" in geometry_info:
                point_counts.append(geometry_info["num_points"])
            else:
                point_counts.append(0)
        
        x_pos = np.arange(len(file_names))
        ax1.bar(x_pos, trajectory_lengths, color=colors)
        ax1.set_xlabel('实例')
        ax1.set_ylabel('轨迹长度 (m)')
        ax1.set_title('轨迹长度比较')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(file_names, rotation=45, ha='right')
        
        # 比较点数量
        ax2 = axes[0, 1]
        ax2.bar(x_pos, point_counts, color=colors)
        ax2.set_xlabel('实例')
        ax2.set_ylabel('点数量')
        ax2.set_title('点数量比较')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(file_names, rotation=45, ha='right')
        
        # 比较序列长度
        ax3 = axes[1, 0]
        frame_counts = []
        for motion_info in all_motion_info:
            if "num_frames" in motion_info:
                frame_counts.append(motion_info["num_frames"])
            else:
                frame_counts.append(0)
        
        ax3.bar(x_pos, frame_counts, color=colors)
        ax3.set_xlabel('实例')
        ax3.set_ylabel('帧数')
        ax3.set_title('序列长度比较')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(file_names, rotation=45, ha='right')
        
        # 轨迹重叠显示
        ax4 = axes[1, 1]
        for i, (motion_info, name) in enumerate(zip(all_motion_info, file_names)):
            if "translation" in motion_info:
                translation = motion_info["translation"]
                ax4.plot(translation[:, 0], translation[:, 1], 
                        color=colors[i], linewidth=2, label=name, alpha=0.7)
                # 标记起点
                ax4.scatter(translation[0, 0], translation[0, 1], 
                          color=colors[i], s=50, marker='o', alpha=0.8)
        
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_title('轨迹重叠显示')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        plt.tight_layout()
        save_path = self.output_dir / "instances_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"比较图保存到: {save_path}")
    
    def create_dashboard(self, instance_files: List[str]):
        """创建综合仪表盘"""
        logger.info("创建综合仪表盘...")
        
        # 收集所有实例的统计信息
        stats = []
        
        for instance_file in instance_files:
            try:
                instance_data = self.load_instance_data(instance_file)
                motion_info = self.extract_motion_info(instance_data)
                geometry_info = self.extract_geometry_info(instance_data)
                
                file_name = Path(instance_file).stem
                
                # 收集统计信息
                stat = {
                    'name': file_name,
                    'num_points': geometry_info.get('num_points', 0),
                    'num_frames': motion_info.get('num_frames', 0),
                    'trajectory_length': 0,
                    'avg_speed': 0,
                    'valid_frames': 0
                }
                
                # 计算轨迹长度和平均速度
                if "translation" in motion_info:
                    translation = motion_info["translation"]
                    if len(translation) > 1:
                        distances = np.linalg.norm(np.diff(translation, axis=0), axis=1)
                        stat['trajectory_length'] = np.sum(distances)
                        stat['avg_speed'] = stat['trajectory_length'] / (len(translation) - 1)
                
                # 计算有效帧数
                if "frame_validity" in motion_info:
                    fv = motion_info["frame_validity"]
                    stat['valid_frames'] = np.sum(fv)
                else:
                    stat['valid_frames'] = stat['num_frames']
                
                stats.append(stat)
                
            except Exception as e:
                logger.error(f"处理实例 {instance_file} 时出错: {str(e)}")
                continue
        
        # 生成仪表盘
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 实例概览表格
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')
        
        # 创建表格数据
        table_data = []
        headers = ['实例名称', '点数', '帧数', '轨迹长度(m)', '平均速度', '有效帧比例']
        
        for stat in stats:
            valid_ratio = stat['valid_frames'] / stat['num_frames'] if stat['num_frames'] > 0 else 0
            table_data.append([
                stat['name'][:20],  # 限制名称长度
                f"{stat['num_points']:,}",
                f"{stat['num_frames']}",
                f"{stat['trajectory_length']:.2f}",
                f"{stat['avg_speed']:.3f}",
                f"{valid_ratio:.2%}"
            ])
        
        table = ax1.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax1.set_title('实例概览', fontsize=14, pad=20)
        
        # 点数分布
        ax2 = fig.add_subplot(gs[0, 2])
        point_counts = [stat['num_points'] for stat in stats]
        ax2.hist(point_counts, bins=min(10, len(stats)), alpha=0.7, color='skyblue')
        ax2.set_xlabel('点数')
        ax2.set_ylabel('频率')
        ax2.set_title('点数分布')
        ax2.grid(True, alpha=0.3)
        
        # 帧数分布
        ax3 = fig.add_subplot(gs[0, 3])
        frame_counts = [stat['num_frames'] for stat in stats]
        ax3.hist(frame_counts, bins=min(10, len(stats)), alpha=0.7, color='lightgreen')
        ax3.set_xlabel('帧数')
        ax3.set_ylabel('频率')
        ax3.set_title('帧数分布')
        ax3.grid(True, alpha=0.3)
        
        # 轨迹长度对比
        ax4 = fig.add_subplot(gs[1, :2])
        names = [stat['name'] for stat in stats]
        trajectory_lengths = [stat['trajectory_length'] for stat in stats]
        
        bars = ax4.bar(range(len(names)), trajectory_lengths, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
        ax4.set_xlabel('实例')
        ax4.set_ylabel('轨迹长度 (m)')
        ax4.set_title('轨迹长度对比')
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels(names, rotation=45, ha='right')
        
        # 在柱状图上显示数值
        for bar, value in zip(bars, trajectory_lengths):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 速度分布
        ax5 = fig.add_subplot(gs[1, 2])
        speeds = [stat['avg_speed'] for stat in stats]
        ax5.hist(speeds, bins=min(10, len(stats)), alpha=0.7, color='orange')
        ax5.set_xlabel('平均速度 (m/frame)')
        ax5.set_ylabel('频率')
        ax5.set_title('速度分布')
        ax5.grid(True, alpha=0.3)
        
        # 统计摘要
        ax6 = fig.add_subplot(gs[1, 3])
        ax6.axis('off')
        
        # 计算统计摘要
        total_instances = len(stats)
        total_points = sum(stat['num_points'] for stat in stats)
        avg_frames = np.mean([stat['num_frames'] for stat in stats])
        avg_trajectory = np.mean([stat['trajectory_length'] for stat in stats])
        
        summary_text = f"""
统计摘要:

总实例数: {total_instances}
总点数: {total_points:,}
平均帧数: {avg_frames:.1f}
平均轨迹长度: {avg_trajectory:.2f} m

点数范围: {min(point_counts):,} - {max(point_counts):,}
帧数范围: {min(frame_counts)} - {max(frame_counts)}
轨迹长度范围: {min(trajectory_lengths):.2f} - {max(trajectory_lengths):.2f} m
        """.strip()
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        # 有效帧比例
        ax7 = fig.add_subplot(gs[2, :])
        valid_ratios = [stat['valid_frames'] / stat['num_frames'] if stat['num_frames'] > 0 else 0 for stat in stats]
        
        bars = ax7.bar(range(len(names)), valid_ratios, color=plt.cm.RdYlGn(valid_ratios))
        ax7.set_xlabel('实例')
        ax7.set_ylabel('有效帧比例')
        ax7.set_title('有效帧比例')
        ax7.set_xticks(range(len(names)))
        ax7.set_xticklabels(names, rotation=45, ha='right')
        ax7.set_ylim(0, 1)
        
        # 在柱状图上显示百分比
        for bar, value in zip(bars, valid_ratios):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.1%}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Instance数据分析仪表盘', fontsize=18, y=0.98)
        plt.tight_layout()
        
        save_path = self.output_dir / "dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"仪表盘保存到: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Instance可视化工具")
    
    # 输入选项
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance_file", type=str, 
                      help="单个实例文件路径")
    group.add_argument("--instance_files", type=str, nargs="+",
                      help="多个实例文件路径")
    group.add_argument("--instance_dir", type=str,
                      help="包含实例文件的目录")
    
    # 输出选项
    parser.add_argument("--output_dir", type=str, default="./instance_visualization",
                       help="输出目录")
    
    # 实例类型（可选）
    parser.add_argument("--instance_type", type=str, 
                       choices=["smpl", "rigid"],
                       help="实例类型过滤")
    
    # 可视化选项
    parser.add_argument("--create_dashboard", action="store_true",
                       help="创建综合仪表盘")
    
    args = parser.parse_args()
    
    try:
        # 创建可视化器
        visualizer = InstanceVisualizer(args.output_dir)
        
        # 确定要处理的文件
        instance_files = []
        
        if args.instance_file:
            instance_files = [args.instance_file]
        elif args.instance_files:
            instance_files = args.instance_files
        elif args.instance_dir:
            # 搜索目录中的实例文件
            instance_dir = Path(args.instance_dir)
            if not instance_dir.exists():
                logger.error(f"目录不存在: {instance_dir}")
                return
            
            # 查找.pkl文件
            instance_files = list(instance_dir.glob("*.pkl"))
            if not instance_files:
                logger.error(f"在目录 {instance_dir} 中没有找到.pkl文件")
                return
            
            instance_files = [str(f) for f in instance_files]
        
        # 类型过滤
        if args.instance_type:
            instance_files = [f for f in instance_files if args.instance_type in Path(f).stem]
        
        if not instance_files:
            logger.error("没有找到要处理的实例文件")
            return
        
        logger.info(f"找到 {len(instance_files)} 个实例文件")
        
        # 验证文件存在
        valid_files = []
        for file_path in instance_files:
            if Path(file_path).exists():
                valid_files.append(file_path)
            else:
                logger.warning(f"文件不存在，跳过: {file_path}")
        
        instance_files = valid_files
        
        if not instance_files:
            logger.error("没有有效的实例文件")
            return
        
        # 执行可视化
        if len(instance_files) == 1:
            visualizer.visualize_instance(instance_files[0], args.instance_type)
        else:
            visualizer.visualize_multiple_instances(instance_files)
        
        # 创建仪表盘
        if args.create_dashboard and len(instance_files) > 1:
            visualizer.create_dashboard(instance_files)
        
        logger.info("=" * 60)
        logger.info("可视化完成！")
        logger.info(f"结果保存到: {args.output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"可视化过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()