# Utility functions for geometric transformations and projections.
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def transform_points(points, transform_matrix):
    """
    Apply a 4x4 transformation matrix to 3D points.

    Args:
        points: (N, 3) tensor of 3D points
        transform_matrix: (4, 4) transformation matrix

    Returns:
        (N, 3) tensor of transformed 3D points
    """
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    homo_points = torch.cat([points, ones], dim=1)  # N x 4
    transformed_points = torch.matmul(homo_points, transform_matrix.T)
    return transformed_points[:, :3]


def get_corners(l: float, w: float, h: float):
    """
    Get 8 corners of a 3D bounding box centered at origin.

    Args:
        l, w, h: length, width, height of the box

    Returns:
        (3, 8) array of corner coordinates
    """
    return np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
        ]
    )


def project_camera_points_to_image(points_cam, cam_intrinsics):
    """
    Project 3D points from camera space to 2D image space.

    Args:
        points_cam (np.ndarray): Shape (N, 3), points in camera space.
        cam_intrinsics (np.ndarray): Shape (3, 3), intrinsic matrix of the camera.

    Returns:
        tuple: (projected_points, depths)
            - projected_points (np.ndarray): Shape (N, 2), projected 2D points in image space.
            - depths (np.ndarray): Shape (N,), depth values of the projected points.
    """
    points_img = cam_intrinsics @ points_cam.T
    depths = points_img[2, :]
    projected_points = (points_img[:2, :] / (depths + 1e-6)).T

    return projected_points, depths


def cube_root(x):
    return torch.sign(x) * torch.abs(x) ** (1.0 / 3)


def spherical_to_cartesian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def uniform_sample_sphere(num_samples, device, inverse=False):
    """
    refer to https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    sample points uniformly inside a sphere
    """
    if not inverse:
        dist = torch.rand((num_samples,)).to(device)
        dist = cube_root(dist)
    else:
        dist = torch.rand((num_samples,)).to(device)
        dist = 1 / dist.clamp_min(0.02)
    thetas = torch.arccos(2 * torch.rand((num_samples,)) - 1).to(device)
    phis = 2 * torch.pi * torch.rand((num_samples,)).to(device)
    pts = spherical_to_cartesian(dist, thetas, phis)
    return pts


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix.

    Args:
        quat: Quaternion tensor of shape (..., 4), in (w, x, y, z) format

    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    # Extract components
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Compute terms
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # Build rotation matrix
    R = torch.stack(
        [
            1 - 2 * (y2 + z2),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (x2 + z2),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (x2 + y2),
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)

    return R


def quaternion_multiply(q1, q2):
    """
    四元数乘法
    q1, q2: [4] 形状的四元数，格式为(w,x,y,z)
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z])

def safe_quaternion_multiply(q1, q2):
    """
    安全的四元数乘法，处理不同的输入维度
    
    Args:
        q1, q2: 四元数张量，可以是 [4] 或 [1, 4] 或 [N, 4] 形状
    
    Returns:
        四元数乘积，保持输入的批次维度
    """
    # 记录原始形状
    q1_shape = q1.shape
    q2_shape = q2.shape
    
    # 确保至少是2D张量
    if q1.dim() == 1:
        q1 = q1.unsqueeze(0)
    if q2.dim() == 1:
        q2 = q2.unsqueeze(0)
    
    # 确保批次维度兼容
    if q1.shape[0] == 1 and q2.shape[0] > 1:
        q1 = q1.expand(q2.shape[0], -1)
    elif q2.shape[0] == 1 and q1.shape[0] > 1:
        q2 = q2.expand(q1.shape[0], -1)
    
    # 执行批量四元数乘法
    # q1 * q2 = [w1*w2 - x1*x2 - y1*y2 - z1*z2,
    #            w1*x2 + x1*w2 + y1*z2 - z1*y2,
    #            w1*y2 - x1*z2 + y1*w2 + z1*x2,
    #            w1*z2 + x1*y2 - y1*x2 + z1*w2]
    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    result = torch.stack([w, x, y, z], dim=1)
    
    # 恢复原始维度
    if len(q1_shape) == 1 and len(q2_shape) == 1:
        result = result.squeeze(0)
    
    return result


def combine_rotations(rotations_list, device='cuda'):
    """
    组合多个旋转（以轴角或四元数形式）
    
    Args:
        rotations_list: 旋转列表，每个旋转是字典 {"axis": [x,y,z], "angle": degrees}
        device: 设备
    
    Returns:
        组合后的四元数
    """
    from pytorch3d.transforms import axis_angle_to_quaternion
    
    # 初始化为单位四元数
    combined_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    
    for rotation in rotations_list:
        axis = rotation["axis"]
        angle_degrees = rotation["angle"]
        
        # 转换为弧度
        angle_radians = math.radians(angle_degrees)
        
        # 归一化旋转轴
        axis_tensor = torch.tensor(axis, device=device, dtype=torch.float32)
        axis_tensor = axis_tensor / torch.norm(axis_tensor)
        
        # 创建轴角表示并转换为四元数
        axis_angle = axis_tensor * angle_radians
        rotation_quat = axis_angle_to_quaternion(axis_angle)
        
        # 组合旋转
        combined_quat = safe_quaternion_multiply(rotation_quat, combined_quat)
    
    return combined_quat
