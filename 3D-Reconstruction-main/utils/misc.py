# Miscellaneous utility functions for exporting point clouds.
import importlib
import logging
import os

import numpy as np
import open3d as o3d
import torch
import torch.distributed as dist

logger = logging.getLogger()


def import_str(string: str):
    """Import a python module given string paths

    Args:
        string (str): The given paths

    Returns:
        Any: Imported python module / object
    """
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def export_points_to_ply(
    positions: torch.tensor,
    colors: torch.tensor,
    save_path: str,
    normalize: bool = False,
):
    """Export points to ply file

    Args:
        positions (torch.tensor): point positions (N, 3)
        colors (torch.tensor): point colors (N, 3)
        save_path (str): path to save ply file
        normalize (bool, optional): whether to normalize points. Defaults to False.
    """
    # detach tensors and move to cpu
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu()

    # normalize points
    if normalize:
        aabb_min = positions.min(0)[0]
        aabb_max = positions.max(0)[0]
        positions = (positions - aabb_min) / (aabb_max - aabb_min)

    # convert to numpy
    if isinstance(positions, torch.Tensor):
        positions = positions.numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.numpy()

    # clamp colors
    colors = np.clip(colors, a_min=0.0, a_max=1.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)


def export_gaussians_to_ply(model, path, name="point_cloud.ply", aabb=None):
    """Export gaussian splatting model to PLY file with all necessary attributes

    Args:
        model: Gaussian model
        path: Output directory
        name: Output filename
        aabb: Optional bounding box for filtering points
    """
    model.eval()
    filename = os.path.join(path, name)

    with torch.no_grad():
        # Get positions and filter by aabb if provided
        positions = model._means.detach()
        if aabb is not None:
            aabb = aabb.to(positions.device)
            aabb_min, aabb_max = aabb[:3], aabb[3:]
            vis_mask = torch.logical_and(
                positions >= aabb_min, positions < aabb_max
            ).all(-1)
        else:
            vis_mask = torch.ones_like(positions[:, 0], dtype=torch.bool)

        # Get all required attributes
        positions = positions[vis_mask].cpu().numpy()
        scales = model.get_scaling[vis_mask].cpu().numpy()  # 使用激活后的scale
        rotations = model.get_quats[vis_mask].cpu().numpy()  # 使用激活后的旋转
        opacities = model.get_opacity[vis_mask].cpu().numpy()

        # Get colors (handle both SH and direct color cases)
        if model.sh_degree > 0:
            colors = model.colors[vis_mask].cpu().numpy()  # This gets base colors
        else:
            colors = torch.sigmoid(model._features_dc[vis_mask]).cpu().numpy()

        # Ensure colors are in [0, 1]
        colors = np.clip(colors, 0, 1)

        # Create the point cloud structure
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Add custom attributes
        custom_data = {
            # Scale attributes
            "scale_0": scales[:, 0],
            "scale_1": scales[:, 1],
            "scale_2": scales[:, 2],
            # Rotation attributes (quaternion)
            "rot_0": rotations[:, 0],
            "rot_1": rotations[:, 1],
            "rot_2": rotations[:, 2],
            "rot_3": rotations[:, 3],
            # Color attributes
            "f_dc_0": colors[:, 0],
            "f_dc_1": colors[:, 1],
            "f_dc_2": colors[:, 2],
            # Opacity
            "opacity": opacities.squeeze(),
        }

        # Add SH coefficients if using SH
        if model.sh_degree > 0:
            sh_rest = model._features_rest[vis_mask].cpu().numpy()
            for i in range(sh_rest.shape[1]):
                for j in range(3):
                    custom_data[f"f_rest_{i}_{j}"] = sh_rest[:, i, j]

        # Write to file with custom attributes
        vertices = np.concatenate([positions, colors], axis=1)
        vertex_data = [tuple(v) for v in vertices]
        vertex_dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "f4"),
            ("green", "f4"),
            ("blue", "f4"),
        ]

        # Add custom attributes to dtype
        for key, value in custom_data.items():
            vertex_dtype.append((key, "f4"))
            vertex_data = [v + (f,) for v, f in zip(vertex_data, value)]

        vertex_array = np.array(vertex_data, dtype=vertex_dtype)

        # Write PLY file
        with open(filename, "wb") as f:
            # Write header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {len(vertex_array)}\n".encode())
            for name, dtype in vertex_dtype:
                f.write(f"property float {name}\n".encode())
            f.write(b"end_header\n")

            # Write data
            vertex_array.tofile(f)

    logger.info(f"Exported {len(positions)} gaussians to {filename}")
    return filename


def is_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process within the global process group.
    """
    return dist.get_rank() if is_enabled() else 0


def get_world_size():
    return dist.get_world_size() if is_enabled() else 1


def is_main_process() -> bool:
    """
    Returns:
        True if the current process is the main one.
    """
    return get_global_rank() == 0
