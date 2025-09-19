# Camera pose manipulation and trajectory generation.
import os
import torch
import numpy as np
import math
from typing import Dict, List, Optional

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R


def interpolate_poses(key_poses: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Interpolate between key poses to generate a smooth trajectory.

    Args:
        key_poses (torch.Tensor): Tensor of shape (N, 4, 4) containing key camera poses.
        target_frames (int): Number of frames to interpolate.

    Returns:
        torch.Tensor: Interpolated poses of shape (target_frames, 4, 4).
    """
    device = key_poses.device
    key_poses = key_poses.cpu().numpy()

    # Separate translation and rotation
    translations = key_poses[:, :3, 3]
    rotations = key_poses[:, :3, :3]

    # Create time array
    times = np.linspace(0, 1, len(key_poses))
    target_times = np.linspace(0, 1, target_frames)

    # Interpolate translations
    interp_translations = np.stack(
        [np.interp(target_times, times, translations[:, i]) for i in range(3)], axis=-1
    )

    # Interpolate rotations using Slerp
    key_rots = R.from_matrix(rotations)
    slerp = Slerp(times, key_rots)
    interp_rotations = slerp(target_times).as_matrix()

    # Combine interpolated translations and rotations
    interp_poses = np.eye(4)[None].repeat(target_frames, axis=0)
    interp_poses[:, :3, :3] = interp_rotations
    interp_poses[:, :3, 3] = interp_translations

    return torch.tensor(interp_poses, dtype=torch.float32, device=device)


def look_at_rotation(
    direction: torch.Tensor, up: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])
) -> torch.Tensor:
    """Calculate rotation matrix to look at a specific direction."""
    # 确保输入张量在相同设备
    up = up.to(direction.device)  # [!code ++]
    front = torch.nn.functional.normalize(direction, dim=-1)
    right = torch.nn.functional.normalize(torch.cross(front, up), dim=-1)
    up = torch.cross(right, front)
    rotation_matrix = torch.stack([right, up, -front], dim=-1)
    return rotation_matrix


def get_interp_novel_trajectories(
    dataset_type: str,
    scene_idx: str,
    per_cam_poses: Dict[int, torch.Tensor],
    traj_type: str = "front_center_interp",
    target_frames: int = 100,
) -> torch.Tensor:
    original_frames = per_cam_poses[list(per_cam_poses.keys())[0]].shape[0]

    trajectory_generators = {
        "front_center_interp": front_center_interp,
        "s_curve": s_curve,
        "three_key_poses": three_key_poses_trajectory,
        # 新增轨迹类型
        "circle_trajectory": circle_trajectory,
        "spiral_trajectory": spiral_trajectory,
        "look_around_trajectory": look_around_trajectory,
        "fixed_path_trajectory": kitti_fixed_path,
        "analyze_center_trajectory":analyze_front_center_interp,
        "analyze_npz_trajectory":analyze_kitti_trajectory,
        "fixed_offset_1": fixed_offset_trajectory_1,
        "fixed_offset_2": fixed_offset_trajectory_2,
        "fixed_offset_3": fixed_offset_trajectory_3,
        "fixed_offset_4": fixed_offset_trajectory_4,
        "fixed_offset_5": fixed_offset_trajectory_5,
        "fixed_offset_6": fixed_offset_trajectory_6,
        "fixed_offset_7": fixed_offset_trajectory_7,
        "fixed_offset_8": fixed_offset_trajectory_8,
        "fixed_offset_9": fixed_offset_trajectory_9,
        "fixed_offset_10": fixed_offset_trajectory_10,
        "fixed_offset": fixed_offset_trajectory,
        "lane_change": smooth_lane_change_trajectory,
        "double_lane_change": double_lane_change_trajectory,
    }

    if traj_type not in trajectory_generators:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return trajectory_generators[traj_type](
        dataset_type, per_cam_poses, original_frames, target_frames
    )

def kitti_fixed_path(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    num_loops: int = 1,
    npz_path = "output/Kitti/dataset=Kitti/change_line_gt/camera_poses_eval/full_poses_2025-07-02_18-00-20.npz",
    position_offset: Optional[List[float]] = None,  # 新增：位置偏移 [x, y, z]
    rotation_offset: Optional[List[float]] = None,  # 新增：旋转偏移 [roll, pitch, yaw] (弧度)
) -> torch.Tensor:
    """
    从NPZ文件读取完整的相机轨迹，不做插值，直接使用原始数据
    
    Args:
        dataset_type (str): 数据集类型（此函数中未使用）
        per_cam_poses (Dict[int, torch.Tensor]): 每相机poses（此函数中未使用）
        original_frames (int): 原始帧数（此函数中未使用）
        target_frames (int): 目标帧数（如果超过原始帧数则重复或截断）
        num_loops (int): 循环次数（此函数中未使用）
        position_offset (List[float], optional): 位置偏移 [x, y, z]，单位米
        rotation_offset (List[float], optional): 旋转偏移 [roll, pitch, yaw]，单位弧度
        
    Returns:
        torch.Tensor: 原始轨迹数据，形状为 (actual_frames, 4, 4)
    """
    # 写死的NPZ文件路径
    # npz_path = "output/Kitti/dataset=Kitti/change_line_gt/camera_poses_eval/full_poses_2025-07-02_18-00-20.npz"
    
    print(f"🔍 Loading complete trajectory from NPZ (no interpolation):")

    # position_offset = [0, 0, 0]
    
    try:
        # 加载NPZ文件
        data = np.load(npz_path, allow_pickle=True)
        camera_poses = data['camera_poses']  # 形状: (N, 4, 4)
        cam_names = data['cam_names']        # 相机名称列表
        frame_indices = data['frame_indices'] # 帧索引
        
        print(f"   NPZ contains {len(camera_poses)} total poses")
        print(f"   Available cameras: {set(cam_names)}")
        
        # 寻找前视中心相机（尝试多种可能的命名）
        front_center_mask = None
        found_camera = None
        
        for candidate in ['CAM_LEFT', 'FRONT_CENTER', 'front_center', 'FRONT', 'front', '0', 'cam0']:
            mask = np.array([str(name) == candidate for name in cam_names])
            if mask.any():
                front_center_mask = mask
                found_camera = candidate
                break
        
        # 如果没找到，使用第一个相机
        if front_center_mask is None:
            front_center_mask = np.ones(len(cam_names), dtype=bool)
            front_center_mask[1:] = False  # 只保留第一个
            found_camera = str(cam_names[0])
        
        print(f"   Using camera: {found_camera}")
        
        # 提取前视中心相机的poses
        front_center_poses = camera_poses[front_center_mask]
        front_center_frames = np.array(frame_indices)[front_center_mask]
        
        print(f"   Found {len(front_center_poses)} poses for this camera")
        
        # 按帧索引排序
        sorted_indices = np.argsort(front_center_frames)
        front_center_poses = front_center_poses[sorted_indices]
        front_center_frames = front_center_frames[sorted_indices]
        
        print(f"   Frame range: {front_center_frames[0]} - {front_center_frames[-1]}")
        
        # 显示位置范围
        positions = front_center_poses[:, :3, 3]
        print(f"   Position ranges:")
        print(f"     X: [{positions[:, 0].min():.6f}, {positions[:, 0].max():.6f}] m")
        print(f"     Y: [{positions[:, 1].min():.6f}, {positions[:, 1].max():.6f}] m")
        print(f"     Z: [{positions[:, 2].min():.6f}, {positions[:, 2].max():.6f}] m")
        
        # 转换为torch tensor
        poses_tensor = torch.tensor(front_center_poses, dtype=torch.float32)
        
        # 确保设备一致性
        if per_cam_poses and len(per_cam_poses) > 0:
            sample_pose = per_cam_poses[list(per_cam_poses.keys())[0]]
            poses_tensor = poses_tensor.to(sample_pose.device)
        
        # 根据target_frames调整输出
        actual_frames = len(poses_tensor)
        
        if target_frames <= actual_frames:
            # 如果目标帧数少于或等于实际帧数，直接截取
            result = poses_tensor[:target_frames]
            print(f"   Truncated to {target_frames} frames (from {actual_frames})")
        else:
            # 如果目标帧数多于实际帧数，重复最后一帧
            result = torch.zeros(target_frames, 4, 4, dtype=poses_tensor.dtype, device=poses_tensor.device)
            result[:actual_frames] = poses_tensor
            # 用最后一帧填充剩余部分
            for i in range(actual_frames, target_frames):
                result[i] = poses_tensor[-1]
            print(f"   Extended to {target_frames} frames (repeated last frame)")
        
        # ==================== 新增：应用偏移 ====================
        if position_offset is not None or rotation_offset is not None:
            print(f"   Applying offsets...")
            result = apply_trajectory_offset(result, position_offset, rotation_offset)
        
        # 输出结果信息
        result_positions = result[:, :3, 3]
        print(f"   Output: {result.shape[0]} frames")
        print(f"   Start: {result_positions[0][0]:.3f}, {result_positions[0][1]:.3f}, {result_positions[0][2]:.3f}")
        print(f"   End:   {result_positions[-1][0]:.3f}, {result_positions[-1][1]:.3f}, {result_positions[-1][2]:.3f}")
        
        return result
        
    except Exception as e:
        print(f"Error loading from NPZ: {e}")
        # 如果NPZ加载失败，回退到原始的front_center_interp
        print("Falling back to front_center_interp")
        assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required for fallback"
        key_poses = per_cam_poses[0][::original_frames // 4]
        return interpolate_poses(key_poses, target_frames)


def apply_trajectory_offset(
    poses: torch.Tensor, 
    position_offset: Optional[List[float]] = None,
    rotation_offset: Optional[List[float]] = None
) -> torch.Tensor:
    """
    给轨迹应用位置和旋转偏移
    
    Args:
        poses: 原始poses张量，形状 (N, 4, 4)
        position_offset: 位置偏移 [x, y, z]，单位米
        rotation_offset: 旋转偏移 [roll, pitch, yaw]，单位弧度
        
    Returns:
        torch.Tensor: 应用偏移后的poses
    """
    import torch
    import math
    
    result = poses.clone()
    
    # 应用位置偏移
    if position_offset is not None:
        offset_tensor = torch.tensor(position_offset, dtype=poses.dtype, device=poses.device)
        print(f"     Position offset: {position_offset}")
        
        # 方法1：简单的全局偏移（在世界坐标系中）
        result[:, :3, 3] += offset_tensor
        
        # 方法2：相对于相机朝向的偏移（如果你想要相对偏移，可以启用这个）
        # for i in range(len(result)):
        #     # 获取当前相机的旋转矩阵
        #     rotation_matrix = result[i, :3, :3]
        #     # 将偏移转换到相机坐标系
        #     relative_offset = rotation_matrix @ offset_tensor
        #     result[i, :3, 3] += relative_offset
    
    # 应用旋转偏移
    if rotation_offset is not None:
        print(f"     Rotation offset (roll, pitch, yaw): {rotation_offset}")
        
        # 将欧拉角转换为旋转矩阵
        roll, pitch, yaw = rotation_offset
        
        # 创建旋转矩阵（ZYX顺序）
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch) 
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        
        # Roll (X轴旋转)
        R_x = torch.tensor([
            [1, 0, 0],
            [0, cos_r, -sin_r],
            [0, sin_r, cos_r]
        ], dtype=poses.dtype, device=poses.device)
        
        # Pitch (Y轴旋转)
        R_y = torch.tensor([
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p]
        ], dtype=poses.dtype, device=poses.device)
        
        # Yaw (Z轴旋转)
        R_z = torch.tensor([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1]
        ], dtype=poses.dtype, device=poses.device)
        
        # 组合旋转矩阵 (ZYX顺序)
        R_offset = R_z @ R_y @ R_x
        
        # 应用旋转偏移到每一帧
        for i in range(len(result)):
            # 原始旋转矩阵
            original_rotation = result[i, :3, :3]
            # 应用偏移旋转
            result[i, :3, :3] = R_offset @ original_rotation
    
    return result
    
def front_center_interp(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    num_loops: int = 1,
) -> torch.Tensor:
    """Interpolate key frames from the front center camera."""
    assert (
        0 in per_cam_poses.keys()
    ), "Front center camera (ID 0) is required for front_center_interp"
    key_poses = per_cam_poses[0][
        :: original_frames // 4
    ]  # Select every 4th frame as key frame
    return interpolate_poses(key_poses, target_frames)


def fixed_offset_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    translation_offset: list = [-4.0, 0.0, 0.0],
    rotation_offset: list = [0.0, 0.0, 0.0],
) -> torch.Tensor:
    """
    生成相对于前视相机的固定偏移轨迹

    Args:
        translation_offset (list): [x, y, z] 平移偏移量（米）
        rotation_offset (list): [pitch, yaw, roll] 旋转偏移量（度）
    """
    assert 0 in per_cam_poses.keys(), "需要前视中心相机（ID 0）"

    # 获取设备信息
    device = per_cam_poses[0].device

    # 转换偏移量为张量
    trans_offset = torch.tensor(translation_offset, device=device, dtype=torch.float32)
    rot_offset = torch.tensor(rotation_offset, device=device, dtype=torch.float32)

    # 确保original_frames至少为1
    original_frames = max(1, original_frames)
    # 计算步长，确保至少为1
    step = max(1, original_frames // 4)
    key_poses = per_cam_poses[0][::step]

    def convert_to_tensor(data, device):
        return torch.tensor(data, device=device, dtype=torch.float32)  # [!code ++]

    # 应用偏移量
    modified_poses = []
    for pose in key_poses:
        # 创建新位姿矩阵
        new_pose = torch.eye(4, device=device)

        rot_matrix = R.from_euler(
            "xyz", rot_offset.cpu().numpy(), degrees=True
        ).as_matrix()
        rot_matrix = rot_matrix.astype(np.float32)  # [!code ++]

        # 修改点3：保持矩阵乘法数据类型一致
        new_rot = pose[:3, :3] @ convert_to_tensor(rot_matrix, device)  # [!code ++]

        # 修改点4：确保平移偏移量数据类型正确
        trans_offset = convert_to_tensor(translation_offset, device)  # [!code ++]
        offset_trans = pose[:3, :3] @ trans_offset
        new_trans = pose[:3, 3] + offset_trans

        new_pose[:3, :3] = new_rot
        new_pose[:3, 3] = new_trans

        modified_poses.append(new_pose)
    # 确保至少有两个位姿才能插值
    if len(modified_poses) == 1:
        # 如果只有一个位姿，直接复制它来创建目标帧数
        return modified_poses[0].unsqueeze(0).repeat(target_frames, 1, 1)
    return interpolate_poses(torch.stack(modified_poses), target_frames)


def analyze_front_center_interp(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    num_loops: int = 1,
) -> torch.Tensor:
    """
    分析front_center_interp的逻辑，显示关键信息
    """
    print(f"🔍 Front Center Interp Analysis:")
    
    # 检查输入
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) required"
    front_poses = per_cam_poses[0]
    
    # 基本信息
    print(f"   Input: {len(front_poses)} poses -> Target: {target_frames} frames")
    print(f"   Original frames param: {original_frames}")
    
    # 关键帧选择逻辑
    step = original_frames // 4
    key_poses = front_poses[::step]
    print(f"   Step size: {step} -> Key frames: {len(key_poses)}")
    
    # 显示关键帧坐标
    print(f"   Key frame positions:")
    for i, pose in enumerate(key_poses):
        pos = pose[:3, 3]
        print(f"     [{i}] {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")
    
    # 进行插值
    result = interpolate_poses(key_poses, target_frames)
    
    # 输出结果
    result_start = result[0][:3, 3]
    result_end = result[-1][:3, 3]
    print(f"   Output: {result.shape[0]} frames")
    print(f"   Start: {result_start[0]:.3f}, {result_start[1]:.3f}, {result_start[2]:.3f}")
    print(f"   End:   {result_end[0]:.3f}, {result_end[1]:.3f}, {result_end[2]:.3f}")
    
    return result


def analyze_kitti_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    num_loops: int = 1,
) -> torch.Tensor:
    """
    分析kitti轨迹的逻辑
    """
    npz_path =  "output/Kitti/dataset=Kitti/training_20250630_162211_FollowLeadingVehicleWithObstacle_1/camera_poses_eval/full_poses_2025-07-02_18-00-20.npz"
    
    print(f"🔍 Kitti Trajectory Analysis:")
    
    try:
        # 加载数据
        data = np.load(npz_path, allow_pickle=True)
        camera_poses = data['camera_poses']
        cam_names = data['cam_names']
        
        # 找到前视中心相机
        front_mask = None
        for candidate in ['FRONT_CENTER', 'front_center', 'FRONT', 'front', '0']:
            mask = np.array([str(name) == candidate for name in cam_names])
            if mask.any():
                front_mask = mask
                break
        
        if front_mask is None:
            front_mask = np.ones(len(cam_names), dtype=bool)
            front_mask[1:] = False
        
        # 提取poses
        front_poses = camera_poses[front_mask]
        frame_indices = data['frame_indices']
        front_frames = np.array(frame_indices)[front_mask]
        
        # 排序
        sorted_indices = np.argsort(front_frames)
        front_poses = front_poses[sorted_indices]
        
        print(f"   NPZ input: {len(front_poses)} poses -> Target: {target_frames} frames")
        
        # 关键帧选择
        actual_frames = len(front_poses)
        step = max(1, actual_frames // 4)
        key_poses = front_poses[::step]
        
        print(f"   Step size: {step} -> Key frames: {len(key_poses)}")
        
        # 显示关键帧坐标
        print(f"   Key frame positions:")
        for i, pose in enumerate(key_poses):
            pos = pose[:3, 3]
            print(f"     [{i}] {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")
        
        # 转换为tensor并插值
        key_poses_tensor = torch.tensor(key_poses, dtype=torch.float32)
        if per_cam_poses and len(per_cam_poses) > 0:
            sample_pose = per_cam_poses[list(per_cam_poses.keys())[0]]
            key_poses_tensor = key_poses_tensor.to(sample_pose.device)
        
        result = interpolate_poses(key_poses_tensor, target_frames)
        
        # 输出结果
        result_start = result[0][:3, 3]
        result_end = result[-1][:3, 3]
        print(f"   Output: {result.shape[0]} frames")
        print(f"   Start: {result_start[0]:.3f}, {result_start[1]:.3f}, {result_start[2]:.3f}")
        print(f"   End:   {result_end[0]:.3f}, {result_end[1]:.3f}, {result_end[2]:.3f}")
        
        return result
        
    except Exception as e:
        print(f"   Error: {e}")
        return analyze_front_center_interp(dataset_type, per_cam_poses, original_frames, target_frames, num_loops)

# FIX_TRAJ= "output/streetgs/dataset=Kitti/training_20250628_171319_FollowLeadingVehicle_1/camera_poses_eval/full_poses_2025-07-09_00-52-47.npz"
FIX_TRAJ= "output/pvg/dataset=Kitti/training_20250628_171319_FollowLeadingVehicle_1/camera_poses/full_poses_2025-07-08_21-34-38.npz"


def fixed_offset_trajectory_1(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """基于kitti_fixed_path + [0.0, 0.0, 0.5]偏移"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.5, 0.0, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_2(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """基于kitti_fixed_path + [3.2, 0.0, 0.0]偏移"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, -3.2, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_3(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """基于kitti_fixed_path + [1.6, 0.0, 0.0]偏移"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, -1.6, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_4(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """基于kitti_fixed_path + [-3.2, 0.0, 0.0]偏移"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, 3.2, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_5(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """基于kitti_fixed_path + [-1.6, 0.0, 0.0]偏移"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, 1.6, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_6(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """基于kitti_fixed_path + [0.5, 0.0, 0.0]偏移"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, -0.5, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_7(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """基于kitti_fixed_path + [-0.5, 0.0, 0.0]偏移"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, 0.5, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_8(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """基于kitti_fixed_path + [0.5, 0.0, 0.0]偏移 + Y轴旋转15度"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, -0.5, 0.0],
        rotation_offset=[0.0, 0.0, math.radians(-15.0)],  # 转换为弧度
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_9(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """基于kitti_fixed_path + [-0.5, 0.0, 0.0]偏移 + Y轴旋转-15度"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, 0.5, 0.0],
        rotation_offset=[0.0, 0.0, math.radians(15.0)],  # 转换为弧度
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_10(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """基于kitti_fixed_path + [0.0, 0.0, -0.5]偏移"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[-0.5, 0.0, 0.0],
        npz_path = FIX_TRAJ
    )

def double_lane_change_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    first_change_start: int = 20,
    first_change_end: int = 50,
    second_change_start: int = 110,
    second_change_end: int = 130,
    lane_offset: float = 3.2,
    offset_vector: list = [0.0, -1.0, 0.0],  # 向左变道
    return_offset_vector: list = [0.0, 1.0, 0.0],  # 向右返回
    first_steer_angle: float = 5.0,   # 第一次变道时的转向角度(度)
    second_steer_angle: float = -5.0,  # 第二次变道时的转向角度(度)
) -> torch.Tensor:
    """
    生成带转向的变道-变回轨迹
    
    每次变道都包含：转向->回正 的过程
    转向从change_start开始，在change_end结束
    转向绕Y轴旋转（水平转向）
    """
    import math
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    assert 0 in per_cam_poses.keys(), "需要前视中心相机（ID 0）"
    assert (
        0
        <= first_change_start
        < first_change_end
        < second_change_start
        < second_change_end
        < target_frames
    ), "帧索引设置有误"
    
    # 获取设备信息
    device = per_cam_poses[0].device
    
    # 生成基础轨迹（使用front_center_interp）
    base_trajectory = front_center_interp(
        dataset_type, per_cam_poses, original_frames, target_frames
    )
    
    # 归一化第一次变道的偏移向量
    first_vector = torch.tensor(offset_vector, device=device, dtype=torch.float32)
    first_vector = first_vector / torch.norm(first_vector)
    first_full_offset = first_vector * lane_offset
    
    # 归一化第二次变道的偏移向量
    second_vector = torch.tensor(
        return_offset_vector, device=device, dtype=torch.float32
    )
    second_vector = second_vector / torch.norm(second_vector)
    second_full_offset = second_vector * lane_offset
    
    # 创建变道轨迹
    lane_change_trajectory = base_trajectory.clone()
    
    # 创建转向旋转（绕Y轴旋转 - 水平转向）
    # 第一次变道的转向旋转
    first_rot = R.from_euler('y', first_steer_angle, degrees=True)
    first_rot_matrix = torch.tensor(first_rot.as_matrix(), device=device, dtype=torch.float32)
    
    # 第二次变道的转向旋转
    second_rot = R.from_euler('y', second_steer_angle, degrees=True)
    second_rot_matrix = torch.tensor(second_rot.as_matrix(), device=device, dtype=torch.float32)
    
    # 单位旋转矩阵（直行状态）
    identity_rot = torch.eye(3, device=device, dtype=torch.float32)
    
    # 自定义球面线性插值函数
    def custom_slerp(rot1, rot2, t):
        """
        自定义的球面线性插值，兼容旧版本scipy
        """
        q1 = rot1.as_quat()
        q2 = rot2.as_quat()
        
        # 计算四元数点积
        dot = np.dot(q1, q2)
        
        # 如果点积为负，反转第二个四元数以确保最短路径
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # 如果四元数几乎相同，直接线性插值
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            result /= np.linalg.norm(result)
            return R.from_quat(result)
        
        # 计算插值角度
        theta_0 = np.arccos(np.abs(dot))
        sin_theta_0 = np.sin(theta_0)
        
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        # 球面线性插值
        result = s0 * q1 + s1 * q2
        return R.from_quat(result)
    
    # 计算每次变道的阶段长度
    first_duration = first_change_end - first_change_start
    first_steer_duration = first_duration // 2  # 转向阶段（前半段）
    first_return_duration = first_duration - first_steer_duration  # 回正阶段（后半段）
    
    second_duration = second_change_end - second_change_start
    second_steer_duration = second_duration // 2  # 转向阶段（前半段）
    second_return_duration = second_duration - second_steer_duration  # 回正阶段（后半段）
    
    # 应用变道和转向
    for frame_idx in range(target_frames):
        original_rotation = base_trajectory[frame_idx, :3, :3].clone()
        
        if frame_idx < first_change_start:
            # 第一次变道之前保持原始轨迹
            continue
            
        elif frame_idx < first_change_start + first_steer_duration:
            # 第一次变道：转向阶段（前半段）
            progress = (frame_idx - first_change_start) / first_steer_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # 平滑位移
            current_offset = first_full_offset * ((frame_idx - first_change_start) / first_duration)
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # 平滑转向
            start_rot = R.from_matrix(identity_rot.cpu().numpy())
            end_rot = R.from_matrix(first_rot_matrix.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # 应用平滑旋转
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
            
        elif frame_idx <= first_change_end:
            # 第一次变道：回正阶段（后半段）
            progress = (frame_idx - first_change_start - first_steer_duration) / first_return_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # 继续移动
            current_offset = first_full_offset * ((frame_idx - first_change_start) / first_duration)
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # 从转向回到直行
            start_rot = R.from_matrix(first_rot_matrix.cpu().numpy())
            end_rot = R.from_matrix(identity_rot.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # 应用平滑旋转
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
            
        elif frame_idx < second_change_start:
            # 在两次变道之间保持直行
            lane_change_trajectory[frame_idx, :3, 3] += first_full_offset
            # 保持直行状态
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ identity_rot
            
        elif frame_idx < second_change_start + second_steer_duration:
            # 第二次变道：转向阶段（前半段）
            progress = (frame_idx - second_change_start) / second_steer_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # 平滑返回
            total_progress = (frame_idx - second_change_start) / second_duration
            current_offset = first_full_offset + second_full_offset * total_progress
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # 平滑转向
            start_rot = R.from_matrix(identity_rot.cpu().numpy())
            end_rot = R.from_matrix(second_rot_matrix.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # 应用平滑旋转
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
            
        elif frame_idx <= second_change_end:
            # 第二次变道：回正阶段（后半段）
            progress = (frame_idx - second_change_start - second_steer_duration) / second_return_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # 继续返回
            total_progress = (frame_idx - second_change_start) / second_duration
            current_offset = first_full_offset + second_full_offset * total_progress
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # 从转向回到直行
            start_rot = R.from_matrix(second_rot_matrix.cpu().numpy())
            end_rot = R.from_matrix(identity_rot.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # 应用平滑旋转
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
            
        else:
            # 第二次变道之后保持直行状态
            # 不添加任何偏移，保持直行
            continue
    
    return lane_change_trajectory


def smooth_lane_change_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    start_frame: int = 20,
    end_frame: int = 50,
    lane_offset: float = -3.2,
    offset_vector: list = [0.0, 1.0, 0.0],  # 向左变道
    steer_angle: float = -5.0,  # 转向角度(度)
) -> torch.Tensor:
    """
    生成带转向的单次变道轨迹
    
    变道过程包含：转向->回正 的过程
    转向从start_frame开始，在end_frame结束
    转向绕Y轴旋转（水平转向）
    """
    import math
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    assert 0 in per_cam_poses.keys(), "需要前视中心相机（ID 0）"
    assert 0 <= start_frame < end_frame < target_frames, "帧索引设置有误"
    
    # 获取设备信息
    device = per_cam_poses[0].device
    
    # 生成基础轨迹（使用front_center_interp）
    base_trajectory = front_center_interp(
        dataset_type, per_cam_poses, original_frames, target_frames
    )
    
    # 归一化偏移向量
    norm_vector = torch.tensor(offset_vector, device=device, dtype=torch.float32)
    norm_vector = norm_vector / torch.norm(norm_vector)
    
    # 计算完整偏移量
    full_offset = norm_vector * lane_offset
    
    # 创建变道轨迹
    lane_change_trajectory = base_trajectory.clone()
    
    # 创建转向旋转（绕Y轴旋转 - 水平转向）
    steer_rot = R.from_euler('y', steer_angle, degrees=True)
    steer_rot_matrix = torch.tensor(steer_rot.as_matrix(), device=device, dtype=torch.float32)
    
    # 单位旋转矩阵（直行状态）
    identity_rot = torch.eye(3, device=device, dtype=torch.float32)
    
    # 自定义球面线性插值函数
    def custom_slerp(rot1, rot2, t):
        """
        自定义的球面线性插值，兼容旧版本scipy
        """
        q1 = rot1.as_quat()
        q2 = rot2.as_quat()
        
        # 计算四元数点积
        dot = np.dot(q1, q2)
        
        # 如果点积为负，反转第二个四元数以确保最短路径
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # 如果四元数几乎相同，直接线性插值
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            result /= np.linalg.norm(result)
            return R.from_quat(result)
        
        # 计算插值角度
        theta_0 = np.arccos(np.abs(dot))
        sin_theta_0 = np.sin(theta_0)
        
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        # 球面线性插值
        result = s0 * q1 + s1 * q2
        return R.from_quat(result)
    
    # 计算变道的阶段长度
    duration = end_frame - start_frame
    steer_duration = duration // 2  # 转向阶段（前半段）
    return_duration = duration - steer_duration  # 回正阶段（后半段）
    
    # 对每一帧应用平滑过渡的偏移和转向
    for frame_idx in range(target_frames):
        original_rotation = base_trajectory[frame_idx, :3, :3].clone()
        
        if frame_idx < start_frame:
            # 起始帧之前保持原始轨迹
            continue
            
        elif frame_idx < start_frame + steer_duration:
            # 转向阶段（前半段）
            progress = (frame_idx - start_frame) / steer_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # 平滑位移
            current_offset = full_offset * ((frame_idx - start_frame) / duration)
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # 平滑转向
            start_rot = R.from_matrix(identity_rot.cpu().numpy())
            end_rot = R.from_matrix(steer_rot_matrix.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # 应用平滑旋转
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
            
        elif frame_idx <= end_frame:
            # 回正阶段（后半段）
            progress = (frame_idx - start_frame - steer_duration) / return_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # 继续移动
            current_offset = full_offset * ((frame_idx - start_frame) / duration)
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # 从转向回到直行
            start_rot = R.from_matrix(steer_rot_matrix.cpu().numpy())
            end_rot = R.from_matrix(identity_rot.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # 应用平滑旋转
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
        else:
            # 结束帧之后保持新位置和直行状态
            lane_change_trajectory[frame_idx, :3, 3] += full_offset
            # 保持直行状态
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ identity_rot
    
    return lane_change_trajectory

def s_curve(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Create an S-shaped trajectory using the front three cameras."""
    assert all(
        cam in per_cam_poses.keys() for cam in [0, 1, 2]
    ), "Front three cameras (IDs 0, 1, 2) are required for s_curve"
    key_poses = torch.cat(
        [
            per_cam_poses[0][0:1],
            per_cam_poses[1][original_frames // 4 : original_frames // 4 + 1],
            per_cam_poses[0][original_frames // 2 : original_frames // 2 + 1],
            per_cam_poses[2][3 * original_frames // 4 : 3 * original_frames // 4 + 1],
            per_cam_poses[0][-1:],
        ],
        dim=0,
    )
    return interpolate_poses(key_poses, target_frames)


def three_key_poses_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """
    Create a trajectory using three key poses:
    1. First frame of front center camera
    2. Middle frame with interpolated rotation and position from camera 1 or 2
    3. Last frame of front center camera

    The rotation of the middle pose is calculated using Slerp between
    the start frame and the middle frame of camera 1 or 2.

    Args:
        dataset_type (str): Type of the dataset (e.g., "waymo", "pandaset", etc.).
        per_cam_poses (Dict[int, torch.Tensor]): Dictionary of camera poses.
        original_frames (int): Number of original frames.
        target_frames (int): Number of frames in the output trajectory.

    Returns:
        torch.Tensor: Trajectory of shape (target_frames, 4, 4).
    """
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required"
    assert (
        1 in per_cam_poses.keys() or 2 in per_cam_poses.keys()
    ), "Either camera 1 or camera 2 is required"

    # First key pose: First frame of front center camera
    start_pose = per_cam_poses[0][0]
    key_poses = [start_pose]

    # Select camera for middle frame
    middle_frame = int(original_frames // 2)
    chosen_cam = np.random.choice([1, 2])

    middle_pose = per_cam_poses[chosen_cam][middle_frame]

    # Calculate interpolated rotation for middle pose
    start_rotation = R.from_matrix(start_pose[:3, :3].cpu().numpy())
    middle_rotation = R.from_matrix(middle_pose[:3, :3].cpu().numpy())
    slerp = Slerp(
        [0, 1], R.from_quat([start_rotation.as_quat(), middle_rotation.as_quat()])
    )
    interpolated_rotation = slerp(0.5).as_matrix()

    # Create middle key pose with interpolated rotation and original translation
    middle_key_pose = torch.eye(4, device=start_pose.device)
    middle_key_pose[:3, :3] = torch.tensor(
        interpolated_rotation, device=start_pose.device
    )
    middle_key_pose[:3, 3] = middle_pose[:3, 3]  # Keep the original translation
    key_poses.append(middle_key_pose)

    # Third key pose: Last frame of front center camera
    key_poses.append(per_cam_poses[0][-1])

    # Stack the key poses and interpolate
    key_poses = torch.stack(key_poses)
    return interpolate_poses(key_poses, target_frames)


def circle_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    radius: float = 5.0,
    height: float = 2.0,
) -> torch.Tensor:
    """生成环绕场景的圆形轨迹"""
    # 修复1：正确获取中心点坐标
    center_pose = per_cam_poses[0][original_frames // 2]
    center = center_pose[:3, 3].cpu().numpy()  # [!code --]
    center = center_pose[:3, 3].cpu().numpy()  # [!code ++] 直接取位置坐标

    # 修复2：添加调试信息
    print(f"Center pose shape: {center_pose.shape}")  # 应为 (4,4)
    print(f"Center coordinates: {center}")  # 应显示三维坐标

    # 生成圆形轨迹参数
    angles = np.linspace(0, 2 * np.pi, 12)
    key_poses = []

    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + height  # [!code ++]

        # 确保坐标类型正确
        pose = torch.eye(4, device=center_pose.device)
        pose[:3, 3] = torch.tensor([x, y, z], device=center_pose.device)

        # 修复3：添加方向计算保护
        direction = center - pose[:3, 3].cpu().numpy()
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([0.0, 0.0, 1.0])  # 防止零向量

        pose[:3, :3] = look_at_rotation(
            torch.tensor(direction, device=center_pose.device)
        )
        key_poses.append(pose)

    return interpolate_poses(torch.stack(key_poses), target_frames)


def spiral_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    radius: float = 5.0,
    spiral_height: float = 3.0,
    num_turns: int = 2,
) -> torch.Tensor:
    """生成螺旋上升轨迹"""
    center_pose = per_cam_poses[0][original_frames // 2]
    center = center_pose[:3, 3].mean(dim=0).cpu().numpy()

    angles = np.linspace(0, num_turns * 2 * np.pi, 12)
    key_poses = []
    for i, angle in enumerate(angles):
        r = radius * (1 - i / len(angles))  # 半径逐渐缩小
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        z = center[2] + spiral_height * (i / len(angles))

        pose = torch.eye(4, device=center_pose.device)
        pose[:3, 3] = torch.tensor([x, y, z], device=center_pose.device)
        direction = center - pose[:3, 3].cpu().numpy()
        pose[:3, :3] = look_at_rotation(
            torch.tensor(direction, device=center_pose.device)
        )

        key_poses.append(pose)

    return interpolate_poses(torch.stack(key_poses), target_frames)


def look_around_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    elevation_range: tuple = (-30, 30),
    azimuth_range: tuple = (0, 360),
) -> torch.Tensor:
    """生成环绕观察轨迹（固定位置，旋转视角）"""
    center_pose = per_cam_poses[0][original_frames // 2]
    center = center_pose[:3, 3].cpu().numpy()

    # 生成视角参数
    elevations = np.linspace(*elevation_range, 6)
    azimuths = np.linspace(*azimuth_range, 6)

    key_poses = []
    for elev, azim in zip(elevations, azimuths):
        # 将球坐标转换为笛卡尔坐标
        r = np.linalg.norm(center)
        x = r * np.cos(np.radians(azim)) * np.cos(np.radians(elev))
        y = r * np.sin(np.radians(azim)) * np.cos(np.radians(elev))
        z = r * np.sin(np.radians(elev))

        pose = torch.eye(4, device=center_pose.device)
        pose[:3, 3] = torch.tensor([x, y, z], device=center_pose.device)
        direction = center - pose[:3, 3].cpu().numpy()
        pose[:3, :3] = look_at_rotation(
            torch.tensor(direction, device=center_pose.device)
        )

        key_poses.append(pose)

    return interpolate_poses(torch.stack(key_poses), target_frames)