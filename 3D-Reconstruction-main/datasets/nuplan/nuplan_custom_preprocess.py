import json
import os
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image
import math
from utils.geometry import get_corners, project_camera_points_to_image
import yaml
from collections import defaultdict

# 添加KITTI类别定义
KITTI_LABELS = [
'vehicle', 'pedestrian', 'bicycle'
]


KITTI_NONRIGID_DYNAMIC_CLASSES = ['pedestrian', 'bicycle']

KITTI_RIGID_DYNAMIC_CLASSES = ['vehicle']

KITTI_DYNAMIC_CLASSES = KITTI_NONRIGID_DYNAMIC_CLASSES + KITTI_RIGID_DYNAMIC_CLASSES


class CustomProcessor(object):
    def __init__(
        self,
        load_dir: str,
        save_dir: str,
        split: str = "2025_02_20",
        split_file: str = None,
        process_keys: List[str] = None,
        process_id_list: List[str] = None,
        workers: int = 8,
    ):
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.split = split
        self.split_file = split_file
        self.process_keys = process_keys
        self.workers = workers

        # 定义相机名称和后缀的映射
        self.cam_mapping = {
            "SUB_RGB_1": "1",
            "SUB_RGB_2": "2",
            "SUB_RGB_3": "3",
            "SUB_RGB_4": "4",
            "SUB_RGB_5": "5",
            "SUB_RGB_6": "6",
            "SUB_RGB_7": "7",
            "SUB_RGB_8": "8",
        }

        # 定义相机名称和图像标签路径的映射
        self.image_label_mapping = {
            "SUB_RGB_1": "camera_1",
            "SUB_RGB_2": "camera_2",
            "SUB_RGB_3": "camera_3",
            "SUB_RGB_4": "camera_4",
            "SUB_RGB_5": "camera_5",
            "SUB_RGB_6": "camera_6",
            "SUB_RGB_7": "camera_7",
            "SUB_RGB_8": "camera_8",
        }
        # 读取场景列表
        if self.split_file is not None:
            with open(self.split_file, "r") as f:
                self.process_id_list = [
                    line.strip().split(",")[0] for line in f.readlines()
                ]
        else:
            self.process_id_list = (
                process_id_list  # 如果未提供 split_file，使用传入的 process_id_list
            )

        # 其他初始化代码
        # self.HW = (720, 1280)
        self.HW = (1080, 1920)
        self.cam_list = ["SUB_RGB_1", "SUB_RGB_2", "SUB_RGB_3", "SUB_RGB_4", 
                        "SUB_RGB_5", "SUB_RGB_6", "SUB_RGB_7", "SUB_RGB_8"]
        self.cam_configs = {
            "SUB_RGB_1": {
                "BLUEPRINT": "sensor.camera.rgb",
                "ATTRIBUTE": {"image_size_x": 1920, "image_size_y": 1080, "fov": 90},
                "TRANSFORM": {
                    "location": [0, 0.1, 0.0],
                    "rotation_matrix": [
                        [0, -1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]
                    ]
                },
            },
            "SUB_RGB_2": {
                "BLUEPRINT": "sensor.camera.rgb",
                "ATTRIBUTE": {"image_size_x": 1920, "image_size_y": 1080, "fov": 90},
                "TRANSFORM": {
                    "location": [0, -0.1, 0.0],
                    "rotation_matrix": [
                        [0, -1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]
                    ]
                },
            },
            "SUB_RGB_3": {
                "BLUEPRINT": "sensor.camera.rgb",
                "ATTRIBUTE": {"image_size_x": 1920, "image_size_y": 1080, "fov": 90},
                "TRANSFORM": {
                    "location": [0.1, 0, 0.0],
                    "rotation_matrix": [
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]
                    ]
                },
            },
            "SUB_RGB_4": {
                "BLUEPRINT": "sensor.camera.rgb",
                "ATTRIBUTE": {"image_size_x": 1920, "image_size_y": 1080, "fov": 90},
                "TRANSFORM": {
                    "location": [-0.1, 0, 0.0],
                    "rotation_matrix": [
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]
                    ]
                },
            },
            "SUB_RGB_5": {
                "BLUEPRINT": "sensor.camera.rgb",
                "ATTRIBUTE": {"image_size_x": 1920, "image_size_y": 1080, "fov": 90},
                "TRANSFORM": {
                    "location": [0, 0, 0.1],
                    "rotation_matrix": [
                        [0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]
                    ]
                },
            },
            "SUB_RGB_6": {
                "BLUEPRINT": "sensor.camera.rgb",
                "ATTRIBUTE": {"image_size_x": 1920, "image_size_y": 1080, "fov": 90},
                "TRANSFORM": {
                    "location": [0, 0, -0.1],
                    "rotation_matrix": [
                        [0, -1, 0],
                        [0, 0, 1],
                        [-1, 0, 0]
                    ]
                },
            },
            "SUB_RGB_7": {
                "BLUEPRINT": "sensor.camera.rgb",
                "ATTRIBUTE": {"image_size_x": 1920, "image_size_y": 1080, "fov": 90},
                "TRANSFORM": {
                    "location": [0.1, 0.1, 0.1],
                    "rotation_matrix": [
                        [0, -1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]
                    ]
                },
            },
            "SUB_RGB_8": {
                "BLUEPRINT": "sensor.camera.rgb",
                "ATTRIBUTE": {"image_size_x": 1920, "image_size_y": 1080, "fov": 90},
                "TRANSFORM": {
                    "location": [-0.1, -0.1, -0.1],
                    "rotation_matrix": [
                        [0, -1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]
                    ]
                },
            },
        }
        self.lidar_to_ego = self.build_transform_matrix(
            location=[0, 0, 1.60],  # 激光雷达安装位置 (x,y,z)
            rotation=[0, 0, 0],  # 安装角度 (pitch,yaw,roll)
        )
        self.create_folder()

    def convert(self):
        """转换数据."""
        if self.process_id_list is None:
            # 如果未提供 process_id_list，处理所有场景
            scene_names = os.listdir(os.path.join(self.load_dir, self.split))
        else:
            # 使用 process_id_list 处理指定场景
            scene_names = self.process_id_list

        for scene_name in scene_names:
            self.convert_one(scene_name)

    def convert_one(self, scene_name: str):
        """处理单个场景."""
        # 确保路径包含 2025_02_20_drive_0000_sync
        basedir = os.path.join(self.load_dir, self.split, scene_name)
        print(f"Processing scene: {basedir}")

        # 处理图像
        if "images" in self.process_keys:
            self.save_image(basedir, scene_name)
            print(f"Processing Image: {basedir}")

        # 处理标定数据
        if "calib" in self.process_keys:
            self.save_calib(basedir, scene_name)
            print(f"Processing Calib: {basedir}")

        # 其他处理逻辑
        # ...
        if "pose" in self.process_keys:
            self.save_pose(basedir, scene_name)
            print(f"Processing pose: {basedir}")
        if "lidar" in self.process_keys:
            self.save_lidar(basedir, scene_name)
            print(f"Processing lidar: {basedir}")
        if "dynamic_masks" in self.process_keys:
            self.save_dynamic_mask(basedir, scene_name)
            print(f"Processing dynamic_masks: {basedir}")
        if "objects" in self.process_keys:
            os.makedirs(f"{self.save_dir}/{scene_name}/instances", exist_ok=True)

            # 必须先保存实例信息
            self.save_instances_info(basedir, scene_name)

            # 然后生成帧实例映射
            self.save_frame_instances(basedir, scene_name)

    def calculate_intrinsic(self, calib_data: Dict[str, np.ndarray], cam_idx: int):
        """从 calib.txt 中读取相机内参."""
        # 获取对应的投影矩阵（P0, P1, P2, P3）
        proj_matrix = calib_data[f"P{cam_idx}"]

        # 提取前 3x3 部分作为内参矩阵
        intrinsic_matrix = proj_matrix[:3, :3]
        return intrinsic_matrix

    def calculate_extrinsic(self, cam_config, calib_data):
        """根据config计算相机外参，直接使用旋转矩阵"""
        location = cam_config["TRANSFORM"]["location"]
        rotation_matrix = np.array(cam_config["TRANSFORM"]["rotation_matrix"])
        
        # 构建4x4变换矩阵
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = location
        
        return extrinsic_matrix

    def read_2d_bbox_for_mask(self, filepath):
        """专门读取2D BBox用于生成动态掩码"""
        bboxes = []
        with open(filepath, "r") as f:
            for line in f:
                data = line.strip().split()
                if len(data) < 9:  # 只需保证有足够字段解析bbox
                    continue

                # 严格匹配KITTI的bbox字段位置
                bbox = {
                    "type": data[0],  # 需要类型信息用于分类掩码
                    "bbox_2d": [
                        int(float(data[5])),  # x1
                        int(float(data[6])),  # y1
                        int(float(data[7])),  # x2
                        int(float(data[8])),  # y2
                    ],
                }
                bboxes.append(bbox)
        return bboxes

    def read_label_for_json(self, filepath):
        """读取标签生成JSON格式数据（用于物体跟踪）"""
        objects = []
        with open(filepath, "r") as f:
            for line in f:
                data = line.strip().split()

                # 构建符合KITTI标准的对象信息
                obj_info = {
                    "type": data[0],
                    "obj_id": int(data[1]),
                    "transform": self._calculate_transform_matrix(
                        [float(x) for x in data[12:15]],  # location
                        float(data[15]),  # rotation_y
                        float(data[9]),
                    ).tolist(),
                    "dimensions": [float(x) for x in data[9:12]],  # h,l,w
                    "rotation_y": float(data[15]),
                    "location": [float(x) for x in data[12:15]],
                    "bbox_2d": [int(float(x)) for x in data[5:9]],  # x1,y1,x2,y2
                }
                objects.append(obj_info)
        return objects

    def _calculate_transform_matrix(self, location, rotation_y, h):
        """计算4x4变换矩阵（私有方法）"""
        ry = rotation_y
        c = np.cos(ry)
        s = np.sin(ry)
        tx, ty, tz = location
        tz += h / 2  # 修正高度中心

        transform = np.array(
            [[c, -s, 0, tx], [s, c, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]]
        )
        return transform

    def _calculate_bbox_rotation_in_world(self, rotation_y):
        """计算4x4变换矩阵（私有方法）"""
        ry = rotation_y
        R = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )
        transform = np.eye(4)
        transform[:3, :3] = R
        return transform

    def save_image(self, basedir: str, scene_name: str):
        """复制图像数据并转换为指定格式."""
        image_dir = os.path.join(basedir, "image")

        # 检查 image 目录是否存在
        if not os.path.exists(image_dir):
            print(f"Warning: Image directory not found: {image_dir}")
            return

        # 获取所有 .png 文件并按文件名排序
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

        # 创建目标目录
        os.makedirs(f"{self.save_dir}/{scene_name}/images", exist_ok=True)

        for image_file in image_files:
            try:
                # 提取帧号和相机编号（支持 000000_camera_4.png 格式）
                parts = image_file.split("_")
                if len(parts) < 3 or parts[2].split(".")[0] not in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                    # print(f"Skipping non-standard file: {image_file}")  # 跳过非标准文件
                    continue

                frame_idx = int(parts[0])  # 提取 000000
                camera_idx = int(parts[2].split(".")[0])  # 提取 4 或 5

                # 根据相机编号映射到目标编号
                if 1 <= camera_idx <= 8:
                    target_idx = camera_idx - 1
                else:
                    continue  # 忽略其他相机编号

                # 构建目标文件名（000_0.jpg 或 000_1.jpg）
                dst_filename = f"{str(frame_idx).zfill(3)}_{target_idx}.jpg"
                dst_path = os.path.join(
                    self.save_dir, scene_name, "images", dst_filename
                )

                # 读取源图像并保存为 .jpg 格式
                src_path = os.path.join(image_dir, image_file)
                image = Image.open(src_path)

                # 如果图像是 RGBA 模式，转换为 RGB 模式
                if image.mode == "RGBA":
                    image = image.convert("RGB")

                image.save(dst_path, "JPEG")
                # print(f"Saved {src_path} to {dst_path}")  # 调试信息
            except Exception as e:
                print(f"Error processing {image_file}: {e}")  # 错误处理
        # 处理新的掩码文件
        self.save_fine_dynamic_masks(basedir, scene_name)
        self.save_sky_masks(basedir, scene_name)

    def save_fine_dynamic_masks(self, basedir: str, scene_name: str):
        """保存精细动态掩码"""
        mask_dir = os.path.join(basedir, "mask")
        os.makedirs(
            f"{self.save_dir}/{scene_name}/fine_dynamic_mask/all", exist_ok=True
        )
        os.makedirs(
            f"{self.save_dir}/{scene_name}/fine_dynamic_mask/human", exist_ok=True
        )
        os.makedirs(
            f"{self.save_dir}/{scene_name}/fine_dynamic_mask/vehicle", exist_ok=True
        )

        # 获取所有帧号
        rigid_files = sorted(os.listdir(os.path.join(mask_dir, "rigid")))
        frame_nums = [f.split("_")[0] for f in rigid_files]

        for frame_num in frame_nums:
            for cam_suffix in  ["1", "2", "3", "4", "5", "6", "7", "8"]:
                # 读取并合并掩码
                rigid_path = os.path.join(
                    mask_dir, "rigid", f"{frame_num}_rigid_{cam_suffix}.png"
                )
                nonrigid_path = os.path.join(
                    mask_dir, "nonrigid", f"{frame_num}_nonrigid_{cam_suffix}.png"
                )

                rigid_mask = cv2.imread(rigid_path, cv2.IMREAD_GRAYSCALE)
                nonrigid_mask = cv2.imread(nonrigid_path, cv2.IMREAD_GRAYSCALE)

                # 合并所有动态物体
                all_mask = cv2.bitwise_or(rigid_mask, nonrigid_mask)

                # 保存掩码
                save_name = f"{str(int(frame_num)).zfill(3)}_{int(cam_suffix)-1}.png"
                cv2.imwrite(
                    f"{self.save_dir}/{scene_name}/fine_dynamic_mask/all/{save_name}",
                    all_mask,
                )
                cv2.imwrite(
                    f"{self.save_dir}/{scene_name}/fine_dynamic_mask/vehicle/{save_name}",
                    rigid_mask,
                )
                cv2.imwrite(
                    f"{self.save_dir}/{scene_name}/fine_dynamic_mask/human/{save_name}",
                    nonrigid_mask,
                )

    def save_sky_masks(self, basedir: str, scene_name: str):
        """保存天空掩码"""
        mask_dir = os.path.join(basedir, "mask")
        os.makedirs(f"{self.save_dir}/{scene_name}/sky_masks", exist_ok=True)

        sky_files = sorted(os.listdir(os.path.join(mask_dir, "sky")))
        for sky_file in sky_files:
            frame_num, _, cam_suffix = sky_file.split("_")
            sky_path = os.path.join(mask_dir, "sky", sky_file)

            # 读取并保存天空掩码
            sky_mask = cv2.imread(sky_path, cv2.IMREAD_GRAYSCALE)
            save_name = f"{str(int(frame_num)).zfill(3)}_{int(cam_suffix[0])-1}.png"
            cv2.imwrite(f"{self.save_dir}/{scene_name}/sky_masks/{save_name}", sky_mask)

    def save_calib(self, basedir: str, scene_name: str):
        """保存标定参数。"""
        calib_path = os.path.join(basedir, "calib.txt")

        # 读取calib.txt
        calib_data = self.read_calib_file(calib_path)

        # 创建 intrinsics 和 extrinsics 文件夹
        os.makedirs(f"{self.save_dir}/{scene_name}/intrinsics", exist_ok=True)
        os.makedirs(f"{self.save_dir}/{scene_name}/extrinsics", exist_ok=True)

        # 保存内参
        for i, cam_name in enumerate(self.cam_list):
            intrinsic = self.calculate_intrinsic(calib_data, i)
            # 将内参矩阵转换为指定格式
            formatted_intrinsic = self.format_intrinsic(intrinsic)
            np.savetxt(
                f"{self.save_dir}/{scene_name}/intrinsics/{i}.txt",
                formatted_intrinsic,
                fmt="%.15e",
            )

        # 保存外参，使用硬编码的相机配置
        for i, cam_name in enumerate(self.cam_list):
            cam_config = self.cam_configs[cam_name]
            extrinsic = self.calculate_extrinsic(cam_config, calib_data)
            np.savetxt(
                f"{self.save_dir}/{scene_name}/extrinsics/{i}.txt",
                extrinsic,
                fmt="%.15e",
            )

    def save_pose(self, basedir: str, scene_name: str):
        """保存 ego 位姿信息."""
        pose_dir = os.path.join(basedir, "ego_state")  # ego_state 路径
        # print(f"Checking pose directory: {pose_dir}")  # 调试信息

        # 检查 pose 目录是否存在
        if not os.path.exists(pose_dir):
            print(f"Warning: Pose directory not found: {pose_dir}")
            return

        # 获取所有 .txt 文件并按文件名排序
        pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith(".txt")])
        # print(f"Found pose files: {pose_files}")  # 调试信息

        # 如果 pose 目录为空，直接返回
        if not pose_files:
            print(f"Warning: No pose files found in {pose_dir}")
            return

        # 创建目标目录
        os.makedirs(f"{self.save_dir}/{scene_name}/ego_pose", exist_ok=True)

        for pose_file in pose_files:
            frame_idx = int(pose_file.split(".")[0])
            pose_path = os.path.join(pose_dir, pose_file)

            try:
                # 读取 ego 状态
                ego_state = self.read_ego_state(pose_path)

                # 生成IMU坐标系下的位姿矩阵
                imu_matrix = self.ego_state_to_matrix(ego_state)

                # 加载雷达外参（假设已通过calculate_extrinsic计算）
                lidar_extrinsic = self.lidar_to_ego

                # 转换到雷达坐标系：T_lidar = T_imu * inv(T_imu_to_lidar)
                lidar_matrix = imu_matrix @ np.linalg.inv(lidar_extrinsic)

                # 保存位姿矩阵
                save_path = f"{self.save_dir}/{scene_name}/ego_pose/{str(frame_idx).zfill(3)}.txt"
                np.savetxt(save_path, lidar_matrix, fmt="%.6f")
            except Exception as e:
                print(f"Error processing pose file {pose_path}: {e}")

    def save_lidar(self, basedir: str, scene_name: str):
        """复制激光雷达数据."""
        lidar_dir = os.path.join(basedir, "velodyne")
        # 获取所有 .bin 文件并按文件名排序
        lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".bin")])

        # 创建目标目录
        os.makedirs(f"{self.save_dir}/{scene_name}/lidar", exist_ok=True)

        for lidar_file in lidar_files:
            # 提取帧号（假设文件名为 000000_lidar_0.bin）
            frame_idx = int(lidar_file.split("_")[0])
            src_path = os.path.join(lidar_dir, lidar_file)
            dst_path = (
                f"{self.save_dir}/{scene_name}/lidar/{str(frame_idx).zfill(3)}.bin"
            )
            os.system(f"cp {src_path} {dst_path}")

    def save_dynamic_mask(self, basedir: str, scene_name: str):
        """生成符合KITTI标准的动态物体掩码"""
        label_dir = os.path.join(basedir, "image_label")

        # 创建输出目录
        save_path = f"{self.save_dir}/{scene_name}/dynamic_masks"
        os.makedirs(f"{save_path}/all", exist_ok=True)
        os.makedirs(f"{save_path}/human", exist_ok=True)
        os.makedirs(f"{save_path}/vehicle", exist_ok=True)

        # 处理每个标签文件
        for label_file in sorted(os.listdir(label_dir)):
            if not label_file.endswith(".txt"):
                continue

            # 修改帧号解析方式
            frame_part = label_file.split("_camera_")[0]  # 新解析方式
            frame_idx = int(frame_part)  # 现在得到正确帧号

            # 初始化掩码（左/右相机）
            masks = {
                "left": {
                    "all": np.zeros(self.HW, dtype=np.uint8),
                    "human": np.zeros(self.HW, dtype=np.uint8),
                    "vehicle": np.zeros(self.HW, dtype=np.uint8),
                },
                "right": {
                    "all": np.zeros(self.HW, dtype=np.uint8),
                    "human": np.zeros(self.HW, dtype=np.uint8),
                    "vehicle": np.zeros(self.HW, dtype=np.uint8),
                },
            }

            # 处理左右相机的标签文件
            for cam_suffix in ["0", "1"]:
                label_path = os.path.join(
                    label_dir, label_file.replace("_0.txt", f"_{cam_suffix}.txt")
                )

                if not os.path.exists(label_path):
                    continue

                # 读取2D bbox
                bboxes = self.read_2d_bbox_for_mask(label_path)

                # 生成掩码
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox["bbox_2d"]
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

                    # 确定相机类型
                    cam_type = "left" if cam_suffix == "0" else "right"

                    # 全动态物体
                    if bbox["type"] in KITTI_DYNAMIC_CLASSES:
                        cv2.fillPoly(
                            masks[cam_type]["all"], [pts.astype(np.int32)], 255
                        )

                    # 人类（非刚性）
                    if bbox["type"] in KITTI_NONRIGID_DYNAMIC_CLASSES:
                        cv2.fillPoly(
                            masks[cam_type]["human"], [pts.astype(np.int32)], 255
                        )

                    # 车辆（刚性）
                    if bbox["type"] in KITTI_RIGID_DYNAMIC_CLASSES:
                        cv2.fillPoly(
                            masks[cam_type]["vehicle"], [pts.astype(np.int32)], 255
                        )

            # 保存掩码文件
            for cam_type in ["left", "right"]:
                suffix = "0" if cam_type == "left" else "1"
                Image.fromarray(masks[cam_type]["all"]).save(
                    f"{save_path}/all/{str(frame_idx).zfill(3)}_{suffix}.png"
                )
                Image.fromarray(masks[cam_type]["human"]).save(
                    f"{save_path}/human/{str(frame_idx).zfill(3)}_{suffix}.png"
                )
                Image.fromarray(masks[cam_type]["vehicle"]).save(
                    f"{save_path}/vehicle/{str(frame_idx).zfill(3)}_{suffix}.png"
                )

    def save_instances_info(self, basedir: str, scene_name: str):
        """从激光雷达标签生成实例信息（包含ego车辆特殊处理）"""
        output_path = f"{self.save_dir}/{scene_name}/instances/instances_info.json"
        instances = defaultdict(
            lambda: {
                "id": None,
                "class_name": "",
                "frame_annotations": {
                    "frame_idx": [],
                    "obj_to_world": [],
                    "box_size": [],
                },
            }
        )

        # 处理每帧标签数据
        label_dir = os.path.join(basedir, "lidar_label")
        for label_file in sorted(os.listdir(label_dir)):
            if not label_file.endswith(".txt"):
                continue

            # 解析帧号并验证ego pose
            frame_id = int(label_file.split(".")[0])
            ego_pose_path = os.path.join(
                self.save_dir,
                scene_name,
                "ego_pose",
                f"{frame_id:03d}.txt",
            )

            if not os.path.exists(ego_pose_path):
                print(f"Warning: Skip frame {frame_id} due to missing ego pose")
                continue

            try:
                ego_to_world = np.loadtxt(ego_pose_path)
                if ego_to_world.shape != (4, 4):
                    raise ValueError(f"Invalid ego pose matrix at frame {frame_id}")
            except Exception as e:
                print(f"Error processing {ego_pose_path}: {str(e)}")
                continue

            # 处理单个物体
            label_path = os.path.join(label_dir, label_file)
            for obj in self.read_label_for_json(label_path):
                if obj["type"] in ["TrafficSigns", "TrafficLight"]:
                    continue
                # 类别名称转换
                class_name = obj["type"]
                if class_name == "Bicycle":
                    class_name = "Cyclist"  # 关键修改点
                instance_id = str(obj["obj_id"])

                rotation_in_world = self._calculate_bbox_rotation_in_world(
                    obj["rotation_y"]
                )

                obj_in_lidar = np.array(obj["transform"])
                obj_in_world = ego_to_world @ obj_in_lidar
                # new_instance_id = str(int(instance_id) - 1)
                # 更新实例信息
                instances[instance_id].update(
                    {
                        "id": int(instance_id),
                        "class_name": class_name,
                    }
                )
                instances[instance_id]["frame_annotations"]["frame_idx"].append(
                    frame_id
                )
                instances[instance_id]["frame_annotations"]["obj_to_world"].append(
                    obj_in_world.tolist()
                )
                instances[instance_id]["frame_annotations"]["box_size"].append(
                    [
                        obj["dimensions"][1],  # length
                        obj["dimensions"][2],  # width
                        obj["dimensions"][0],  # height
                    ]
                )

        # 保存最终结果
        with open(output_path, "w") as f:
            json.dump(
                {k: v for k, v in instances.items()}, f, indent=4, ensure_ascii=False
            )

    def build_transform_matrix(self, location, rotation):
        """根据位置和旋转构建4x4变换矩阵"""
        matrix = np.eye(4)

        # 转换旋转角度为弧度
        rx, ry, rz = np.radians(rotation)

        # 构建旋转矩阵（ZYX顺序）
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
        )

        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )

        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
        )

        R = Rz @ Ry @ Rx
        matrix[:3, :3] = R
        matrix[:3, 3] = location
        return matrix

    def save_frame_instances(self, basedir: str, scene_name: str):
        """生成严格排序的帧-实例映射关系"""
        frame_data = defaultdict(list)

        # 修改数据加载方式
        with open(f"{self.save_dir}/{scene_name}/instances/instances_info.json") as f:
            instances_info = json.load(f)  # 直接加载实例字典

        # 按数字顺序处理实例（解决JSON键默认无序问题）
        for instance_id in sorted(
            instances_info.keys(), key=lambda x: int(x)
        ):  # 字符串键转数字排序
            instance = instances_info[instance_id]
            frame_indices = instance["frame_annotations"]["frame_idx"]

            # 转换所有帧号为整数并去重
            unique_frames = sorted({int(idx) for idx in frame_indices})

            for frame_idx in unique_frames:
                frame_data[frame_idx].append(int(instance_id))

        # 双重排序保障
        sorted_frames = sorted(frame_data.items(), key=lambda x: x[0])
        final_data = {str(k): sorted(v) for k, v in sorted_frames}  # 实例ID排序

        output_path = f"{self.save_dir}/{scene_name}/instances/frame_instances.json"
        with open(output_path, "w") as f:
            json.dump(final_data, f, indent=4, sort_keys=True)

    # 辅助函数
    def read_calib_file(self, filepath: str) -> Dict[str, np.ndarray]:
        """读取 calib.txt 文件."""
        calib_data = {}
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("P"):
                    key, values = line.strip().split(":")
                    matrix = np.array([float(x) for x in values.split()]).reshape(3, 4)
                    calib_data[key] = matrix
        return calib_data

    def format_intrinsic(self, P):
        """格式化内参矩阵."""
        return np.array(
            [
                P[0, 0],  # fx
                P[1, 1],  # fy
                P[0, 2],  # cx
                P[1, 2],  # cy
                0,  # p1
                0,  # p2
                0,  # k1
                0,  # k2
                0,  # k3
            ]
        )

    def parse_transform(self, transform):
        """解析变换矩阵."""
        matrix = np.eye(4)
        matrix[:3, :4] = transform.reshape(3, 4)
        return matrix

    def read_ego_state(self, filepath: str) -> Dict[str, Dict[str, float]]:
        """读取 ego 状态信息."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ego state file not found: {filepath}")

        with open(filepath, "r") as f:
            content = f.read().strip()
            # 提取 Location
            location_start = content.find("Location(")
            location_end = content.find(")", location_start)
            location_str = content[location_start : location_end + 1]

            # 调试：打印 rotation_str
            # print("Location string:", location_str)

            # 检查 rotation_str 是否包含 pitch, yaw, roll
            if (
                "x=" not in location_str
                or "y=" not in location_str
                or "z=" not in location_str
            ):
                raise ValueError(f"Invalid rotation format in file: {filepath}")
            x = float(location_str.split("x=")[1].split(",")[0])
            y = float(location_str.split("y=")[1].split(",")[0])
            z = float(location_str.split("z=")[1].split(")")[0])

            # 提取 Rotation
            rotation_start = content.find("Rotation(")
            rotation_end = content.find(")", rotation_start)
            rotation_str = content[rotation_start : rotation_end + 1]

            # 调试：打印 rotation_str
            # print("Rotation string:", rotation_str)

            # 检查 rotation_str 是否包含 pitch, yaw, roll
            if (
                "pitch=" not in rotation_str
                or "yaw=" not in rotation_str
                or "roll=" not in rotation_str
            ):
                raise ValueError(f"Invalid rotation format in file: {filepath}")

            pitch = float(rotation_str.split("pitch=")[1].split(",")[0])
            yaw = float(rotation_str.split("yaw=")[1].split(",")[0])
            roll = float(rotation_str.split("roll=")[1].split(")")[0])

            # 解析 Velocity
            velocity_start = content.find("Velocity: {")
            velocity_end = content.find("}", velocity_start)
            velocity_str = content[velocity_start : velocity_end + 1]
            velocity_x = float(velocity_str.split("x': ")[1].split(",")[0])
            velocity_y = float(velocity_str.split("y': ")[1].split(",")[0])
            velocity_z = float(velocity_str.split("z': ")[1].split("}")[0])

            # 解析 Acceleration
            acceleration_start = content.find("Acceleration: {")
            acceleration_end = content.find("}", acceleration_start)
            acceleration_str = content[acceleration_start : acceleration_end + 1]
            acceleration_x = float(acceleration_str.split("x': ")[1].split(",")[0])
            acceleration_y = float(acceleration_str.split("y': ")[1].split(",")[0])
            acceleration_z = float(acceleration_str.split("z': ")[1].split("}")[0])

            return {
                "Transform": {
                    "Location": {"x": x, "y": y, "z": z},
                    "Rotation": {"pitch": pitch, "yaw": yaw, "roll": roll},
                },
                "Velocity": {"x": velocity_x, "y": velocity_y, "z": velocity_z},
                "Acceleration": {
                    "x": acceleration_x,
                    "y": acceleration_y,
                    "z": acceleration_z,
                },
            }

    def ego_state_to_matrix(self, ego_state):
        """将ego state转换为4x4矩阵."""
        # 提取位置和旋转信息
        location = ego_state["Transform"]["Location"]
        rotation = ego_state["Transform"]["Rotation"]

        # 转换为弧度
        pitch = math.radians(rotation["pitch"])
        yaw = math.radians(rotation["yaw"])
        roll = math.radians(rotation["roll"])

        # 构建旋转矩阵
        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)],
            ]
        )

        Ry = np.array(
            [
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)],
            ]
        )

        Rz = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1],
            ]
        )

        R = Rz @ Ry @ Rx

        # 构建4x4变换矩阵
        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[:3, 3] = [location["x"], location["y"], location["z"]]

        return matrix

    def create_folder(self):
        """创建输出目录结构."""
        scene_list = self.get_scene_list()
        for scene_name in scene_list:
            os.makedirs(f"{self.save_dir}/{scene_name}/images", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/intrinsics", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/extrinsics", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/ego_pose", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/lidar", exist_ok=True)
            os.makedirs(
                f"{self.save_dir}/{scene_name}/dynamic_masks/all", exist_ok=True
            )
            os.makedirs(
                f"{self.save_dir}/{scene_name}/dynamic_masks/human", exist_ok=True
            )
            os.makedirs(
                f"{self.save_dir}/{scene_name}/dynamic_masks/vehicle", exist_ok=True
            )
            os.makedirs(f"{self.save_dir}/{scene_name}/instances", exist_ok=True)

    def get_scene_list(self):
        """获取场景列表."""
        return sorted(
            [
                d
                for d in os.listdir(self.load_dir)
                if os.path.isdir(os.path.join(self.load_dir, d))
            ]
        )
