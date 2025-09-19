"""
适配最新目录结构的点云可视化工具
主要修改：
1. 移除lidar_idx参数
2. 适配新的文件命名格式（000.bin）
3. 保持核心可视化逻辑不变
"""

import open3d as o3d
import numpy as np
import argparse
import json
import os
import cv2


def load_point_cloud(scene_path, frame_idx):
    """加载并转换点云到世界坐标系"""
    # 加载原始点云
    bin_path = os.path.join(scene_path, "lidar", f"{frame_idx:03d}.bin")
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    # 读取ego pose（世界坐标系下的雷达姿态）
    ego_pose_path = os.path.join(scene_path, "ego_pose", f"{frame_idx:03d}.txt")
    ego_to_world = np.loadtxt(ego_pose_path)

    # 硬编码的雷达到ego的外参（与预处理一致）
    lidar_to_ego = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # lidar_to_ego = np.array(
    #     [[1, 0, 0, 0], [0, 1, 0, 0.1], [0, 0, 1, 1.6], [0, 0, 0, 1]]
    # )

    # 坐标转换：lidar->ego->world
    homog_points = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    world_points = (ego_to_world @ lidar_to_ego @ homog_points.T).T

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_points[:, :3])
    # 在load_point_cloud函数末尾添加
    print(f"[DEBUG] 坐标转换验证（第{frame_idx}帧）:")
    print(f"雷达到ego矩阵:\n{lidar_to_ego.round(3)}")
    print(f"Ego到世界矩阵:\n{ego_to_world.round(3)}")
    print(
        f"转换后点云范围：X({world_points[:,0].min():.1f}~{world_points[:,0].max():.1f}) "
        f"Y({world_points[:,1].min():.1f}~{world_points[:,1].max():.1f}) "
        f"Z({world_points[:,2].min():.1f}~{world_points[:,2].max():.1f})"
    )
    return pcd, lidar_to_ego, ego_to_world


def read_instances_from_json(scene_path, frame_idx):
    """
    从预处理生成的JSON文件读取实例数据（世界坐标系）

    参数：
        scene_path: 预处理后的场景目录路径（包含instances/子目录）
        frame_idx: 要读取的帧序号（三位数格式）

    返回：
        bboxes: 边界框列表（open3d.geometry.OrientedBoundingBox）
        metadata: 元数据字典
    """
    print(f"\n[DEBUG] 开始读取实例数据，场景路径：{scene_path}，帧号：{frame_idx}")

    instances_dir = os.path.join(scene_path, "instances")
    frame_instances_path = os.path.join(instances_dir, "frame_instances.json")
    instances_info_path = os.path.join(instances_dir, "instances_info.json")

    # 文件存在性检查
    print(f"[DEBUG] 检查文件存在性：")
    print(f" - Frame instances: {frame_instances_path}")
    print(f" - Instances info: {instances_info_path}")

    if not os.path.exists(frame_instances_path):
        raise FileNotFoundError(f"帧实例文件不存在：{frame_instances_path}")
    if not os.path.exists(instances_info_path):
        raise FileNotFoundError(f"实例信息文件不存在：{instances_info_path}")

    # 读取数据
    print("[DEBUG] 正在加载JSON文件...")
    with open(frame_instances_path) as f:
        frame_instances = json.load(f)
    with open(instances_info_path) as f:
        instances_info = json.load(f)
    print(f"[DEBUG] 成功加载 {len(instances_info)} 个实例的元数据")

    bboxes = []
    metadata = {}

    # 修正帧号匹配方式（JSON键为原始字符串格式）
    frame_key = str(frame_idx)
    instance_ids = frame_instances.get(frame_key, [])
    print(f"[DEBUG] 当前帧 {frame_idx} 包含 {len(instance_ids)} 个实例：{instance_ids}")

    for instance_id in instance_ids:
        # 转换为字符串匹配JSON键
        str_id = str(instance_id)
        instance = instances_info.get(str_id)

        if not instance:
            print(f"[WARNING] 实例 {instance_id} 不存在于instances_info.json，已跳过")
            continue

        # 获取当前帧在实例轨迹中的位置
        frame_indices = instance["frame_annotations"]["frame_idx"]
        try:
            idx_in_track = frame_indices.index(frame_idx)
        except ValueError:
            print(f"[ERROR] 实例 {instance_id} 不包含帧 {frame_idx} 的标注，已跳过")
            continue

        # 安全获取数据
        try:
            obj_to_world = np.array(
                instance["frame_annotations"]["obj_to_world"][idx_in_track]
            )
            box_size = np.array(instance["frame_annotations"]["box_size"][idx_in_track])
        except (IndexError, KeyError) as e:
            print(f"[ERROR] 实例 {instance_id} 数据获取失败：{str(e)}，已跳过")
            continue

        # 数据校验
        if obj_to_world.shape != (4, 4):
            print(
                f"[ERROR] 实例 {instance_id} 的变换矩阵维度错误：{obj_to_world.shape}，已跳过"
            )
            continue

        if len(box_size) != 3 or any(s <= 0 for s in box_size):
            print(f"[ERROR] 实例 {instance_id} 尺寸无效：{box_size}，已跳过")
            continue

        # # 调试输出
        # print(f"\n[DEBUG] 处理实例 {instance_id}:")
        # print(f"类型: {instance['class_name']}")
        # print(f"尺寸 (LWH): {box_size.tolist()}")
        # print(f"中心坐标转换:\n{obj_to_world.round(4)}")

        # 创建边界框
        try:
            bbox = create_bbox(
                center=obj_to_world[:3, 3],
                box_size=box_size,
                rotation_matrix=obj_to_world[:3, :3],
                object_type=instance["class_name"],
            )
        except Exception as e:
            print(f"[ERROR] 创建实例 {instance_id} 边界框失败：{str(e)}")
            continue

        # 存储元数据
        metadata[bbox] = {
            "object_type": instance["class_name"],
            "instance_id": instance_id,
            "original_extent": box_size,
            "transform_matrix": obj_to_world.tolist(),
        }
        bboxes.append(bbox)

    print(f"[INFO] 成功加载 {len(bboxes)} 个有效边界框")
    return bboxes, metadata


def create_bbox(center, box_size, rotation_matrix, object_type):
    """
    修正旋转矩阵应用方式的边界框创建
    """
    # print(f"\n[DEBUG] 创建 {object_type} 边界框（尺寸：{box_size}）")

    # 验证旋转矩阵有效性
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"无效旋转矩阵维度：{rotation_matrix.shape}")

    # 计算旋转矩阵的行列式（检测是否为真旋转矩阵）
    det = np.linalg.det(rotation_matrix)
    if not np.isclose(det, 1.0, atol=1e-3):
        print(f"[WARNING] 旋转矩阵行列式异常：{det:.4f}，可能包含反射成分")

    # 创建边界框（修正旋转应用顺序）
    bbox = o3d.geometry.OrientedBoundingBox()

    # 先设置中心点（修正高度偏移计算）
    original_center = np.array(center)
    z_offset = 0
    adjusted_center = original_center + np.array([0, 0, z_offset])  # 仅Z轴偏移
    print(f"中心点调整：{original_center} → {adjusted_center} (z_offset: {z_offset})")
    bbox.center = adjusted_center

    # 设置尺寸（LWH对应XYZ方向）
    bbox.extent = np.array([box_size[0], box_size[1], box_size[2]])

    # 应用旋转矩阵（修正旋转基准点）
    # print(f"应用旋转矩阵前边界框方向：\n{bbox.R}")
    bbox.R = rotation_matrix  # 直接设置旋转矩阵
    # print(f"应用旋转矩阵后方向：\n{bbox.R}")

    # 可视化验证点
    corners = np.asarray(bbox.get_box_points())
    # print(f"边界框角点坐标示例：\n{corners[0]} ... {corners[-1]}")

    # 设置颜色
    TYPE_COLORS = {
        "Car": (0, 1, 0),  # 绿色
        "Pedestrian": (1, 0, 0),  # 红色
        # ... 其他颜色定义保持不变 ...
    }
    bbox.color = TYPE_COLORS.get(object_type, (0, 0, 1))

    return bbox


def create_coordinate_frame(transform_matrix, size=1.0, axis_colors=None):
    """创建RGB三色坐标系（X红/Y绿/Z蓝）"""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    # 默认颜色方案
    if axis_colors is None:
        axis_colors = {
            "x": [1, 0, 0],  # 红色X轴
            "y": [0, 1, 0],  # 绿色Y轴
            "z": [0, 0, 1],  # 蓝色Z轴
        }

    # 分离各轴并着色（Open3D坐标系顶点顺序：X轴0-99，Y轴100-199，Z轴200-299）
    vertices = np.asarray(frame.vertices)

    # X轴（红色）
    frame.vertex_colors[0:100] = o3d.utility.Vector3dVector([axis_colors["x"]] * 100)
    # Y轴（绿色）
    frame.vertex_colors[100:200] = o3d.utility.Vector3dVector([axis_colors["y"]] * 100)
    # Z轴（蓝色）
    frame.vertex_colors[200:300] = o3d.utility.Vector3dVector([axis_colors["z"]] * 100)

    # 应用变换矩阵
    frame.transform(transform_matrix)
    return frame


def add_projection_validation(scene_path, frame_idx, pcd_world, ego_to_world):
    """投影验证函数（添加在main函数调用前）"""  # 读取内参文件（假设scene_path已定义）
    intrinsic_path = os.path.join(scene_path, "intrinsics", "0.txt")

    with open(intrinsic_path, "r") as f:
        # 读取所有行并转换为浮点数列表
        params = [float(line.strip()) for line in f.readlines()]

    # 解析前四个参数：fx, fy, cx, cy
    fx, fy, cx, cy = params[:4]

    # 构建内参矩阵
    CAM_INTRINSIC = np.array(
        [
            [fx, 0.0, cx],  # fx, 0, cx
            [0.0, fy, cy],  # 0, fy, cy
            [0.0, 0.0, 1.0],  # 最后一行固定
        ],
        dtype=np.float32,
    )

    # 雷达到相机的外参（ego到相机变换）
    ego_to_cam = np.array(
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0], [1.0, 0.0, 0.0, 0]]
    )

    ego_to_cam = np.array(
        [[-0.5, -0.866, 0.0, 0.0], [0.0, 0.0, -1.0, 0], [0.866, -0.5, 0.0, -1.0]]
    )

    # ego_to_cam = np.array(
    #     [[0.5, -0.866, 0.0, 0.0], [0.0, 0.0, -1.0, 0], [0.866, 0.5, 0.0, -1.0]]
    # )

    # ego_to_cam = np.array(
    #     [[-1.0, 0.0, 0.0, -0.5], [0.0, 0.0, -1.0, 0], [0.0, -1.0, 0.0, -0.5]]
    # )

    # ego_to_cam = get_ego_to_cam()
    print("ego_to_cam format:")
    print(ego_to_cam)
    # 读取相机图像
    img_path = os.path.join(scene_path, "images", f"{frame_idx:03d}_0.jpg")
    if not os.path.exists(img_path):
        print(f"[WARNING] 未找到相机图像：{img_path}")
        return
    img = cv2.imread(img_path)

    # 坐标系转换：世界坐标系 -> 相机坐标系
    world_to_ego = np.linalg.inv(ego_to_world)
    world_points = np.asarray(pcd_world.points)
    homog_points = np.hstack([world_points, np.ones((len(world_points), 1))])

    # 转换链：世界->ego->相机
    points_cam = (ego_to_cam @ world_to_ego @ homog_points.T).T[:, :3]

    # 投影到图像平面
    fx, fy = CAM_INTRINSIC[0, 0], CAM_INTRINSIC[1, 1]
    cx, cy = CAM_INTRINSIC[0, 2], CAM_INTRINSIC[1, 2]
    uv = (points_cam[:, :2] / points_cam[:, 2][:, None]) @ CAM_INTRINSIC[
        :2, :2
    ].T + CAM_INTRINSIC[:2, 2]
    uv = uv.astype(int)

    # 过滤有效投影点
    height, width = img.shape[:2]
    valid = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
        & (points_cam[:, 2] > 0)
    )
    uv = uv[valid]

    # 绘制投影点（红色）
    for u, v in uv:
        cv2.circle(img, (u, v), 2, (0, 0, 255), -1)

    # 显示结果
    cv2.imshow("Projection Validation", img)
    key = cv2.waitKey(0)  # 等待按键
    if key == ord('q'):  # 按下 'q' 键退出
        cv2.destroyAllWindows()


def main():
    # 命令行参数解析（适配新路径结构）
    parser = argparse.ArgumentParser(
        description="点云和边界框可视化工具（适配Volvo数据格式）"
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="预处理场景路径，示例：/home/zyf/Master-Thesis/.../2025_02_20_drive_0003_sync",
    )
    parser.add_argument(
        "--frame",
        type=int,
        required=True,
        help="要可视化的帧序号（三位数，示例：0, 10, 999）",
    )
    args = parser.parse_args()

    # 加载点云并获取转换矩阵
    pcd, lidar_to_ego, ego_to_world = load_point_cloud(
        args.scene, args.frame
    )  # 修改接收返回值
    # 新增投影验证（在可视化前添加）
    add_projection_validation(args.scene, args.frame, pcd, ego_to_world)
    # 创建各坐标系可视化
    coord_frames = []

    # # 世界坐标系（红色，2米大小）
    # world_frame = create_coordinate_frame(np.eye(4), size=2.0)
    # coord_frames.append(world_frame)

    # Ego坐标系（绿色，1.5米大小）
    ego_frame = create_coordinate_frame(ego_to_world, size=1.5)
    coord_frames.append(ego_frame)

    # 雷达坐标系（蓝色，1米大小）
    lidar_in_world = ego_to_world @ lidar_to_ego
    lidar_frame = create_coordinate_frame(lidar_in_world, size=1.0)
    coord_frames.append(lidar_frame)
    # 加载边界框
    bboxes, metadata = read_instances_from_json(args.scene, args.frame)

    # 可视化配置
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加所有几何体
    vis.add_geometry(pcd)
    for frame in coord_frames:
        vis.add_geometry(frame)
    for bbox in bboxes:
        vis.add_geometry(bbox)

    # 设置视角参数（示例参数，可根据需要调整）
    ctr = vis.get_view_control()
    ctr.set_front([0, -1, 0.5])  # 面向正北方向，俯角30度
    ctr.set_up([0, 0, 1])  # Z轴向上
    ctr.set_zoom(0.8)  # 初始缩放级别

    # 打印坐标系信息
    print("\n[DEBUG] 坐标系位置验证：")
    print(f"世界坐标系原点：{np.zeros(3)}")
    print(f"Ego坐标系原点：{ego_to_world[:3,3].round(3)}")
    print(f"雷达坐标系原点：{lidar_in_world[:3,3].round(3)}")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
