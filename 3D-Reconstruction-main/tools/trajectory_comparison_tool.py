#!/usr/bin/env python3
"""
轨迹数据比较工具
比较实例自身的轨迹数据和外部JSON文件中的轨迹数据，并生成可视化图表

使用示例:
python trajectory_comparison_tool.py \
    --instance_file ./saved_instances/smpl_instance_0.pkl \
    --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \
    --trajectory_instance_id 1 \
    --output_dir ./trajectory_comparison
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_instance_trajectory_data(instance_file: str) -> Dict:
    """
    从实例文件中加载轨迹数据
    
    Args:
        instance_file: 实例文件路径
        
    Returns:
        Dict: 包含轨迹信息的字典
    """
    print(f"加载实例文件: {instance_file}")
    
    with open(instance_file, 'rb') as f:
        instance_data = pickle.load(f)
    
    # 提取运动数据
    motion = instance_data.get("motion", {})
    
    # 提取位置和旋转数据
    instances_trans = motion.get("instances_trans")  # [num_frames, 3]
    instances_quats = motion.get("instances_quats")  # [num_frames, 4] 或 [num_frames, 1, 4]
    smpl_qauts = motion.get("smpl_qauts")  # [num_frames, 23, 4]
    instances_fv = motion.get("instances_fv")  # [num_frames]
    
    # 确保数据是numpy数组
    if isinstance(instances_trans, torch.Tensor):
        instances_trans = instances_trans.cpu().numpy()
    if isinstance(instances_quats, torch.Tensor):
        instances_quats = instances_quats.cpu().numpy()
    if isinstance(smpl_qauts, torch.Tensor):
        smpl_qauts = smpl_qauts.cpu().numpy()
    if isinstance(instances_fv, torch.Tensor):
        instances_fv = instances_fv.cpu().numpy()
    
    # 处理形状
    if instances_quats.ndim == 3 and instances_quats.shape[1] == 1:
        instances_quats = instances_quats[:, 0, :]  # [num_frames, 4]
    
    num_frames = len(instances_trans) if instances_trans is not None else 0
    
    print(f"实例轨迹数据:")
    print(f"  帧数: {num_frames}")
    if instances_trans is not None:
        print(f"  位置数据形状: {instances_trans.shape}")
        print(f"  位置范围: X[{instances_trans[:, 0].min():.2f}, {instances_trans[:, 0].max():.2f}]")
        print(f"            Y[{instances_trans[:, 1].min():.2f}, {instances_trans[:, 1].max():.2f}]")
        print(f"            Z[{instances_trans[:, 2].min():.2f}, {instances_trans[:, 2].max():.2f}]")
    
    return {
        "frame_indices": np.arange(num_frames),
        "positions": instances_trans,
        "rotations": instances_quats,
        "smpl_rotations": smpl_qauts,
        "frame_validity": instances_fv,
        "metadata": instance_data.get("metadata", {})
    }


def load_json_trajectory_data(json_path: str, instance_id: str) -> Dict:
    """
    从JSON文件中加载轨迹数据
    
    Args:
        json_path: JSON文件路径
        instance_id: 实例ID
        
    Returns:
        Dict: 包含轨迹信息的字典
    """
    print(f"加载JSON轨迹数据: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if str(instance_id) not in data:
        available_ids = list(data.keys())
        raise ValueError(f"实例ID {instance_id} 不存在于轨迹数据中，可用ID: {available_ids}")
    
    instance_data = data[str(instance_id)]
    frame_annotations = instance_data["frame_annotations"]
    
    frame_indices = np.array(frame_annotations["frame_idx"])
    obj_to_world_matrices = frame_annotations["obj_to_world"]
    
    # 提取位置和旋转信息
    positions = []
    rotations = []
    
    for matrix in obj_to_world_matrices:
        matrix = np.array(matrix)
        # 提取位置 (最后一列的前三个元素)
        position = matrix[:3, 3]
        positions.append(position)
        
        # 提取旋转矩阵并转换为四元数
        rotation_matrix = matrix[:3, :3]
        quat = rotation_matrix_to_quaternion(rotation_matrix)
        rotations.append(quat)
    
    positions = np.array(positions)
    rotations = np.array(rotations)
    
    print(f"JSON轨迹数据:")
    print(f"  实例ID: {instance_id}")
    print(f"  类别: {instance_data.get('class_name', 'Unknown')}")
    print(f"  帧数: {len(frame_indices)}")
    print(f"  帧范围: {frame_indices.min()} - {frame_indices.max()}")
    print(f"  位置范围: X[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"            Y[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"            Z[{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
    
    return {
        "frame_indices": frame_indices,
        "positions": positions,
        "rotations": rotations,
        "class_name": instance_data.get("class_name", "Unknown"),
        "instance_id": instance_id
    }


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为四元数 (w, x, y, z)
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])


def align_trajectories(instance_traj: Dict, json_traj: Dict) -> Tuple[Dict, Dict]:
    """
    对齐两个轨迹的时间戳
    
    Args:
        instance_traj: 实例轨迹数据
        json_traj: JSON轨迹数据
        
    Returns:
        Tuple[Dict, Dict]: 对齐后的轨迹数据
    """
    print("对齐轨迹时间戳...")
    
    instance_frames = instance_traj["frame_indices"]
    json_frames = json_traj["frame_indices"]
    
    print(f"实例帧范围: {instance_frames.min()} - {instance_frames.max()} (共{len(instance_frames)}帧)")
    print(f"JSON帧范围: {json_frames.min()} - {json_frames.max()} (共{len(json_frames)}帧)")
    
    # 找到重叠的帧
    common_frames = np.intersect1d(instance_frames, json_frames)
    print(f"重叠帧数: {len(common_frames)}")
    
    if len(common_frames) == 0:
        print("警告: 没有重叠的帧，将尝试基于偏移量对齐")
        # 如果没有重叠，尝试基于偏移量对齐
        offset = json_frames.min() - instance_frames.min()
        print(f"计算的时间偏移量: {offset}")
        
        # 创建对齐后的帧索引
        aligned_instance_frames = instance_frames + offset
        common_frames = np.intersect1d(aligned_instance_frames, json_frames)
        print(f"偏移后重叠帧数: {len(common_frames)}")
        
        if len(common_frames) > 0:
            # 基于偏移对齐
            instance_mask = np.isin(aligned_instance_frames, common_frames)
            json_mask = np.isin(json_frames, common_frames)
        else:
            # 如果还是没有重叠，使用长度较短的序列
            min_length = min(len(instance_frames), len(json_frames))
            instance_mask = np.arange(min_length)
            json_mask = np.arange(min_length)
            print(f"使用前{min_length}帧进行比较")
    else:
        # 找到对应的索引
        instance_mask = np.isin(instance_frames, common_frames)
        json_mask = np.isin(json_frames, common_frames)
    
    # 创建对齐后的数据
    aligned_instance = {
        "frame_indices": instance_traj["frame_indices"][instance_mask],
        "positions": instance_traj["positions"][instance_mask] if instance_traj["positions"] is not None else None,
        "rotations": instance_traj["rotations"][instance_mask] if instance_traj["rotations"] is not None else None,
    }
    
    aligned_json = {
        "frame_indices": json_traj["frame_indices"][json_mask],
        "positions": json_traj["positions"][json_mask],
        "rotations": json_traj["rotations"][json_mask],
    }
    
    print(f"对齐完成，共{len(aligned_instance['frame_indices'])}个对应帧")
    
    return aligned_instance, aligned_json


def calculate_differences(aligned_instance: Dict, aligned_json: Dict) -> Dict:
    """
    计算两个对齐轨迹之间的差异
    """
    print("计算轨迹差异...")
    
    differences = {}
    
    # 位置差异
    if aligned_instance["positions"] is not None and aligned_json["positions"] is not None:
        pos_diff = aligned_instance["positions"] - aligned_json["positions"]
        pos_distance = np.linalg.norm(pos_diff, axis=1)
        
        differences["position"] = {
            "diff": pos_diff,
            "distance": pos_distance,
            "mean_distance": np.mean(pos_distance),
            "max_distance": np.max(pos_distance),
            "std_distance": np.std(pos_distance)
        }
        
        print(f"位置差异统计:")
        print(f"  平均距离: {differences['position']['mean_distance']:.4f}")
        print(f"  最大距离: {differences['position']['max_distance']:.4f}")
        print(f"  标准差: {differences['position']['std_distance']:.4f}")
    
    # 旋转差异 (简化为角度差异)
    if aligned_instance["rotations"] is not None and aligned_json["rotations"] is not None:
        # 计算四元数角度差异
        rot_diff = []
        for i in range(len(aligned_instance["rotations"])):
            q1 = aligned_instance["rotations"][i]
            q2 = aligned_json["rotations"][i]
            
            # 计算四元数之间的角度差异
            dot_product = np.abs(np.dot(q1, q2))
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_diff = 2 * np.arccos(dot_product)
            rot_diff.append(angle_diff)
        
        rot_diff = np.array(rot_diff)
        
        differences["rotation"] = {
            "angle_diff": rot_diff,
            "mean_angle": np.mean(rot_diff),
            "max_angle": np.max(rot_diff),
            "std_angle": np.std(rot_diff)
        }
        
        print(f"旋转差异统计:")
        print(f"  平均角度差: {np.degrees(differences['rotation']['mean_angle']):.2f}°")
        print(f"  最大角度差: {np.degrees(differences['rotation']['max_angle']):.2f}°")
        print(f"  标准差: {np.degrees(differences['rotation']['std_angle']):.2f}°")
    
    return differences


def plot_trajectory_comparison(aligned_instance: Dict, aligned_json: Dict, 
                             differences: Dict, output_dir: str):
    """
    绘制轨迹比较图表
    """
    print("生成可视化图表...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    frames = aligned_instance["frame_indices"]
    
    # 1. 位置轨迹对比图
    if aligned_instance["positions"] is not None and aligned_json["positions"] is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('位置轨迹对比', fontsize=16)
        
        # 3D轨迹图
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.plot(aligned_instance["positions"][:, 0], 
                aligned_instance["positions"][:, 1], 
                aligned_instance["positions"][:, 2], 
                'b-', label='实例轨迹', linewidth=2)
        ax.plot(aligned_json["positions"][:, 0], 
                aligned_json["positions"][:, 1], 
                aligned_json["positions"][:, 2], 
                'r-', label='JSON轨迹', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('3D轨迹对比')
        
        # X-Y平面图
        axes[0, 1].plot(aligned_instance["positions"][:, 0], 
                       aligned_instance["positions"][:, 1], 
                       'b-', label='实例轨迹', linewidth=2)
        axes[0, 1].plot(aligned_json["positions"][:, 0], 
                       aligned_json["positions"][:, 1], 
                       'r-', label='JSON轨迹', linewidth=2)
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        axes[0, 1].legend()
        axes[0, 1].set_title('X-Y平面轨迹')
        axes[0, 1].grid(True)
        
        # 各轴位置随时间变化
        axes[1, 0].plot(frames, aligned_instance["positions"][:, 0], 'b-', label='实例X', linewidth=1.5)
        axes[1, 0].plot(frames, aligned_json["positions"][:, 0], 'r--', label='JSON X', linewidth=1.5)
        axes[1, 0].plot(frames, aligned_instance["positions"][:, 1], 'g-', label='实例Y', linewidth=1.5)
        axes[1, 0].plot(frames, aligned_json["positions"][:, 1], 'm--', label='JSON Y', linewidth=1.5)
        axes[1, 0].plot(frames, aligned_instance["positions"][:, 2], 'c-', label='实例Z', linewidth=1.5)
        axes[1, 0].plot(frames, aligned_json["positions"][:, 2], 'y--', label='JSON Z', linewidth=1.5)
        axes[1, 0].set_xlabel('帧号')
        axes[1, 0].set_ylabel('位置')
        axes[1, 0].legend()
        axes[1, 0].set_title('位置随时间变化')
        axes[1, 0].grid(True)
        
        # 位置差异
        if "position" in differences:
            axes[1, 1].plot(frames, differences["position"]["distance"], 'k-', linewidth=2)
            axes[1, 1].axhline(y=differences["position"]["mean_distance"], 
                              color='r', linestyle='--', 
                              label=f'平均: {differences["position"]["mean_distance"]:.3f}')
            axes[1, 1].set_xlabel('帧号')
            axes[1, 1].set_ylabel('距离差异')
            axes[1, 1].legend()
            axes[1, 1].set_title('位置距离差异')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'position_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 旋转对比图
    if aligned_instance["rotations"] is not None and aligned_json["rotations"] is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('旋转轨迹对比', fontsize=16)
        
        # 四元数分量对比
        quat_labels = ['W', 'X', 'Y', 'Z']
        colors_instance = ['b', 'g', 'r', 'c']
        colors_json = ['navy', 'darkgreen', 'darkred', 'darkcyan']
        
        for i in range(4):
            axes[0, 0].plot(frames, aligned_instance["rotations"][:, i], 
                           color=colors_instance[i], linestyle='-', 
                           label=f'实例{quat_labels[i]}', linewidth=1.5)
            axes[0, 0].plot(frames, aligned_json["rotations"][:, i], 
                           color=colors_json[i], linestyle='--', 
                           label=f'JSON{quat_labels[i]}', linewidth=1.5)
        
        axes[0, 0].set_xlabel('帧号')
        axes[0, 0].set_ylabel('四元数分量')
        axes[0, 0].legend()
        axes[0, 0].set_title('四元数分量对比')
        axes[0, 0].grid(True)
        
        # 各分量差异
        quat_diff = aligned_instance["rotations"] - aligned_json["rotations"]
        for i in range(4):
            axes[0, 1].plot(frames, quat_diff[:, i], 
                           color=colors_instance[i], 
                           label=f'{quat_labels[i]}差异', linewidth=1.5)
        
        axes[0, 1].set_xlabel('帧号')
        axes[0, 1].set_ylabel('四元数分量差异')
        axes[0, 1].legend()
        axes[0, 1].set_title('四元数分量差异')
        axes[0, 1].grid(True)
        
        # 角度差异
        if "rotation" in differences:
            axes[1, 0].plot(frames, np.degrees(differences["rotation"]["angle_diff"]), 
                           'k-', linewidth=2)
            axes[1, 0].axhline(y=np.degrees(differences["rotation"]["mean_angle"]), 
                              color='r', linestyle='--', 
                              label=f'平均: {np.degrees(differences["rotation"]["mean_angle"]):.2f}°')
            axes[1, 0].set_xlabel('帧号')
            axes[1, 0].set_ylabel('角度差异 (度)')
            axes[1, 0].legend()
            axes[1, 0].set_title('旋转角度差异')
            axes[1, 0].grid(True)
        
        # 四元数模长对比
        instance_norms = np.linalg.norm(aligned_instance["rotations"], axis=1)
        json_norms = np.linalg.norm(aligned_json["rotations"], axis=1)
        
        axes[1, 1].plot(frames, instance_norms, 'b-', label='实例四元数模长', linewidth=2)
        axes[1, 1].plot(frames, json_norms, 'r--', label='JSON四元数模长', linewidth=2)
        axes[1, 1].axhline(y=1.0, color='g', linestyle=':', label='理想值(1.0)')
        axes[1, 1].set_xlabel('帧号')
        axes[1, 1].set_ylabel('四元数模长')
        axes[1, 1].legend()
        axes[1, 1].set_title('四元数模长对比')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'rotation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 统计摘要图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('轨迹差异统计摘要', fontsize=16)
    
    # 位置差异直方图
    if "position" in differences:
        axes[0].hist(differences["position"]["distance"], bins=30, alpha=0.7, 
                    color='blue', edgecolor='black')
        axes[0].axvline(differences["position"]["mean_distance"], 
                       color='red', linestyle='--', linewidth=2, 
                       label=f'平均值: {differences["position"]["mean_distance"]:.4f}')
        axes[0].axvline(differences["position"]["max_distance"], 
                       color='orange', linestyle='--', linewidth=2, 
                       label=f'最大值: {differences["position"]["max_distance"]:.4f}')
        axes[0].set_xlabel('位置距离差异')
        axes[0].set_ylabel('频次')
        axes[0].set_title('位置差异分布')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # 旋转差异直方图
    if "rotation" in differences:
        axes[1].hist(np.degrees(differences["rotation"]["angle_diff"]), bins=30, 
                    alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(np.degrees(differences["rotation"]["mean_angle"]), 
                       color='red', linestyle='--', linewidth=2, 
                       label=f'平均值: {np.degrees(differences["rotation"]["mean_angle"]):.2f}°')
        axes[1].axvline(np.degrees(differences["rotation"]["max_angle"]), 
                       color='orange', linestyle='--', linewidth=2, 
                       label=f'最大值: {np.degrees(differences["rotation"]["max_angle"]):.2f}°')
        axes[1].set_xlabel('旋转角度差异 (度)')
        axes[1].set_ylabel('频次')
        axes[1].set_title('旋转差异分布')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'statistics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存到: {output_path}")


def save_comparison_report(instance_traj: Dict, json_traj: Dict, 
                          differences: Dict, output_dir: str):
    """
    保存详细的比较报告
    """
    output_path = Path(output_dir)
    report_file = output_path / 'trajectory_comparison_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("轨迹数据比较报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 基本信息
        f.write("数据源信息:\n")
        f.write(f"实例轨迹帧数: {len(instance_traj['frame_indices'])}\n")
        f.write(f"JSON轨迹帧数: {len(json_traj['frame_indices'])}\n")
        if 'class_name' in json_traj:
            f.write(f"JSON轨迹类别: {json_traj['class_name']}\n")
        if 'instance_id' in json_traj:
            f.write(f"JSON实例ID: {json_traj['instance_id']}\n")
        f.write("\n")
        
        # 时间范围
        f.write("时间范围:\n")
        f.write(f"实例轨迹: 帧 {instance_traj['frame_indices'].min()} - {instance_traj['frame_indices'].max()}\n")
        f.write(f"JSON轨迹: 帧 {json_traj['frame_indices'].min()} - {json_traj['frame_indices'].max()}\n")
        f.write("\n")
        
        # 位置差异统计
        if "position" in differences:
            pos_diff = differences["position"]
            f.write("位置差异统计:\n")
            f.write(f"平均距离差异: {pos_diff['mean_distance']:.6f}\n")
            f.write(f"最大距离差异: {pos_diff['max_distance']:.6f}\n")
            f.write(f"距离差异标准差: {pos_diff['std_distance']:.6f}\n")
            f.write(f"距离差异中位数: {np.median(pos_diff['distance']):.6f}\n")
            f.write("\n")
            
            # 各轴差异
            pos_diff_xyz = pos_diff['diff']
            f.write("各轴位置差异:\n")
            f.write(f"X轴 - 平均: {np.mean(pos_diff_xyz[:, 0]):.6f}, 标准差: {np.std(pos_diff_xyz[:, 0]):.6f}\n")
            f.write(f"Y轴 - 平均: {np.mean(pos_diff_xyz[:, 1]):.6f}, 标准差: {np.std(pos_diff_xyz[:, 1]):.6f}\n")
            f.write(f"Z轴 - 平均: {np.mean(pos_diff_xyz[:, 2]):.6f}, 标准差: {np.std(pos_diff_xyz[:, 2]):.6f}\n")
            f.write("\n")
        
        # 旋转差异统计
        if "rotation" in differences:
            rot_diff = differences["rotation"]
            f.write("旋转差异统计:\n")
            f.write(f"平均角度差异: {np.degrees(rot_diff['mean_angle']):.3f}°\n")
            f.write(f"最大角度差异: {np.degrees(rot_diff['max_angle']):.3f}°\n")
            f.write(f"角度差异标准差: {np.degrees(rot_diff['std_angle']):.3f}°\n")
            f.write(f"角度差异中位数: {np.degrees(np.median(rot_diff['angle_diff'])):.3f}°\n")
            f.write("\n")
        
        # 数据质量评估
        f.write("数据质量评估:\n")
        if "position" in differences:
            pos_quality = "良好" if differences["position"]["mean_distance"] < 0.1 else "需要注意" if differences["position"]["mean_distance"] < 1.0 else "较差"
            f.write(f"位置一致性: {pos_quality} (平均差异: {differences['position']['mean_distance']:.4f})\n")
        
        if "rotation" in differences:
            rot_quality = "良好" if np.degrees(differences["rotation"]["mean_angle"]) < 5 else "需要注意" if np.degrees(differences["rotation"]["mean_angle"]) < 30 else "较差"
            f.write(f"旋转一致性: {rot_quality} (平均差异: {np.degrees(differences['rotation']['mean_angle']):.2f}°)\n")
        
        f.write("\n")
        f.write("注意: 较大的差异可能表明时间戳不匹配或数据来源不同\n")
    
    print(f"详细报告已保存到: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="轨迹数据比较工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 基本比较:
   python trajectory_comparison_tool.py \\
       --instance_file ./saved_instances/smpl_instance_0.pkl \\
       --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \\
       --trajectory_instance_id 1 \\
       --output_dir ./trajectory_comparison

2. 详细分析:
   python trajectory_comparison_tool.py \\
       --instance_file ./saved_instances/smpl_instance_0.pkl \\
       --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \\
       --trajectory_instance_id 1 \\
       --output_dir ./trajectory_comparison \\
       --verbose
        """
    )
    
    parser.add_argument(
        "--instance_file", type=str, required=True,
        help="实例文件路径 (.pkl)"
    )
    parser.add_argument(
        "--trajectory_json_path", type=str, required=True,
        help="轨迹JSON文件路径"
    )
    parser.add_argument(
        "--trajectory_instance_id", type=str, required=True,
        help="JSON文件中的实例ID"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./trajectory_comparison",
        help="输出目录"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="输出详细信息"
    )
    
    args = parser.parse_args()
    
    try:
        print("开始轨迹数据比较...")
        print("=" * 50)
        
        # 加载数据
        instance_traj = load_instance_trajectory_data(args.instance_file)
        json_traj = load_json_trajectory_data(args.trajectory_json_path, args.trajectory_instance_id)
        
        print("\n" + "=" * 50)
        
        # 对齐轨迹
        aligned_instance, aligned_json = align_trajectories(instance_traj, json_traj)
        
        print("\n" + "=" * 50)
        
        # 计算差异
        differences = calculate_differences(aligned_instance, aligned_json)
        
        print("\n" + "=" * 50)
        
        # 生成可视化
        plot_trajectory_comparison(aligned_instance, aligned_json, differences, args.output_dir)
        
        # 保存报告
        save_comparison_report(aligned_instance, aligned_json, differences, args.output_dir)
        
        print("\n" + "=" * 50)
        print("轨迹比较完成!")
        print(f"结果已保存到: {args.output_dir}")
        
        # 输出关键结论
        print("\n关键结论:")
        if "position" in differences:
            pos_mean = differences["position"]["mean_distance"]
            print(f"- 平均位置差异: {pos_mean:.4f} 单位")
            if pos_mean < 0.1:
                print("  → 位置一致性良好")
            elif pos_mean < 1.0:
                print("  → 位置存在中等差异，需要检查时间戳对齐")
            else:
                print("  → 位置差异较大，可能是不同的轨迹或时间戳严重不匹配")
        
        if "rotation" in differences:
            rot_mean = np.degrees(differences["rotation"]["mean_angle"])
            print(f"- 平均旋转差异: {rot_mean:.2f}°")
            if rot_mean < 5:
                print("  → 旋转一致性良好")
            elif rot_mean < 30:
                print("  → 旋转存在中等差异")
            else:
                print("  → 旋转差异较大")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()