"""
融合新版数据读取和传统可视化逻辑的增强版点云查看工具
主要改进：
1. 保留JSON数据读取逻辑
2. 整合传统坐标转换逻辑
3. 添加调试输出
4. 增强错误处理
"""

import open3d as o3d
import numpy as np
import argparse
import json
import os


def read_instances_from_json(scene_path, frame_idx):
    """
    从JSON文件读取实例数据（带调试输出）

    参数：
        scene_path: 场景目录路径
        frame_idx: 帧序号（三位数格式）

    返回：
        bboxes: 边界框列表
        metadata: 元数据字典
    """
    print(f"[DEBUG] 开始读取实例数据，场景路径：{scene_path}，帧号：{frame_idx:03}")

    instances_dir = os.path.join(scene_path, "instances")
    frame_instances_path = os.path.join(instances_dir, "frame_instances.json")
    instances_info_path = os.path.join(instances_dir, "instances_info.json")

    # 检查文件存在性
    if not os.path.exists(frame_instances_path):
        raise FileNotFoundError(f"帧实例文件不存在：{frame_instances_path}")
    if not os.path.exists(instances_info_path):
        raise FileNotFoundError(f"实例信息文件不存在：{instances_info_path}")

    with open(frame_instances_path) as f:
        frame_instances = json.load(f)
    with open(instances_info_path) as f:
        instances_info = json.load(f)

    bboxes = []
    metadata = {}
    instance_ids = frame_instances.get(f"{frame_idx:03}", [])

    print(f"[DEBUG] 找到 {len(instance_ids)} 个实例")

    for instance_id in instance_ids:
        instance = instances_info[str(instance_id)]

        try:
            frame_indices = [
                int(idx) for idx in instance["frame_annotations"]["frame_idx"]
            ]
            idx = frame_indices.index(frame_idx)
        except ValueError:
            print(f"[WARNING] 实例 {instance_id} 无当前帧注释")
            continue

        obj_to_world = np.array(instance["frame_annotations"]["obj_to_world"][idx])
        box_size = np.array(instance["frame_annotations"]["box_size"][idx])

        # 调试输出原始数据
        print(f"\n[DEBUG] 实例 {instance_id} 原始数据：")
        print(f"obj_to_world:\n{obj_to_world}")
        print(f"box_size: {box_size}")

        bbox = create_bbox(
            center=obj_to_world[:3, 3],
            box_size=box_size,
            rotation_matrix=obj_to_world[:3, :3],
            object_type=instance["class_name"],
        )

        # 存储转换后的元数据
        metadata[bbox] = {
            "original_center": obj_to_world[:3, 3],
            "transformed_center": bbox.center,
            "extent": bbox.extent,
        }
        bboxes.append(bbox)

    return bboxes, metadata


def create_bbox(center, box_size, rotation_matrix, object_type):
    """
    创建带调试信息的边界框（整合传统转换逻辑）

    参数：
        center: 物体中心坐标（世界坐标系）
        box_size: [l, w, h] 尺寸
        rotation_matrix: 3x3旋转矩阵
        object_type: 物体类型
    """
    TYPE_COLORS = {
        "Car": (0, 1, 0),  # 绿色
        "Pedestrian": (1, 0, 0),  # 红色
        "Cyclist": (1, 1, 0),  # 黄色
        "Truck": (0.5, 0.5, 0),  # 橄榄色
        "Van": (0, 1, 1),  # 青色
        "Tram": (1, 0, 1),  # 品红色
    }

    # 初始化边界框
    bbox = o3d.geometry.OrientedBoundingBox()

    # 设置尺寸并添加高度偏移
    original_extent = [box_size[0], box_size[1], box_size[2]]
    bbox.extent = original_extent
    z_offset = box_size[2] / 2  # 高度方向偏移

    # 计算最终中心点
    transformed_center = center + np.array([0, 0, z_offset])
    bbox.center = transformed_center

    # 应用旋转矩阵
    print(f"\n[DEBUG] 应用旋转矩阵：\n{rotation_matrix}")
    bbox.rotate(rotation_matrix, center=bbox.center)

    # 设置颜色
    bbox.color = TYPE_COLORS.get(object_type, (0, 0, 1))

    # 调试输出
    print(f"创建 {object_type} 边界框：")
    print(f"原始中心：{center} → 变换后中心：{transformed_center}")
    print(f"尺寸：{original_extent} | 旋转矩阵：\n{rotation_matrix}")

    return bbox


def main():
    parser = argparse.ArgumentParser(description="增强版点云可视化工具")
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="场景路径，示例：/path/to/2025_02_20_drive_0003_sync",
    )
    parser.add_argument("--frame", type=int, required=True, help="帧序号（示例：0）")

    args = parser.parse_args()

    # 读取点云数据（带存在性检查）
    lidar_path = os.path.join(args.scene, "lidar", f"{args.frame:03}.bin")
    print(f"\n[INFO] 正在读取点云文件：{lidar_path}")

    if not os.path.exists(lidar_path):
        raise FileNotFoundError(f"点云文件不存在：{lidar_path}")

    point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    print(f"成功读取 {point_cloud.shape[0]} 个点")

    # 读取实例数据
    bboxes, metadata = read_instances_from_json(args.scene, args.frame)

    # 可视化设置
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    vis.add_geometry(pcd)

    # 添加坐标系
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

    # 添加边界框
    for bbox in bboxes:
        vis.add_geometry(bbox)

    # 设置渲染参数
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.background_color = np.array([0.1, 0.1, 0.1])

    print("\n[INFO] 启动可视化窗口...")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
