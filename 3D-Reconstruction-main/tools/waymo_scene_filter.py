#!/usr/bin/env python3
"""
Waymo数据集场景筛选脚本
筛选出行人密集区域和十字路口场景
"""

import os
import glob
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import transform_utils
import numpy as np
import argparse
from tqdm import tqdm
import json
from collections import defaultdict
import math

class WaymoSceneFilter:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.pedestrian_threshold = 5  # 行人密集区域阈值
        self.intersection_angle_threshold = 60  # 十字路口角度阈值（度）
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_scenes': 0,
            'pedestrian_dense_scenes': 0,
            'intersection_scenes': 0,
            'both_criteria_scenes': 0
        }
    
    def load_tfrecord_files(self):
        """加载所有tfrecord文件"""
        pattern = os.path.join(self.data_dir, "*.tfrecord")
        files = glob.glob(pattern)
        print(f"找到 {len(files)} 个 tfrecord 文件")
        return files
    
    def count_pedestrians_in_frame(self, frame):
        """统计帧中的行人数量"""
        pedestrian_count = 0
        
        # 遍历所有激光雷达标签
        for laser_label in frame.laser_labels:
            if laser_label.type == open_dataset.Label.TYPE_PEDESTRIAN:
                pedestrian_count += 1
        
        # 遍历所有相机标签
        for camera_label in frame.camera_labels:
            for label in camera_label.labels:
                if label.type == open_dataset.Label.TYPE_PEDESTRIAN:
                    pedestrian_count += 1
        
        return pedestrian_count
    
    def detect_intersection(self, frame):
        """检测是否为十字路口场景"""
        # 获取道路线信息
        road_lines = []
        
        # 从地图特征中提取道路线
        if hasattr(frame, 'map_features'):
            for map_feature in frame.map_features:
                if map_feature.HasField('road_line'):
                    road_lines.append(map_feature.road_line)
                elif map_feature.HasField('road_edge'):
                    road_lines.append(map_feature.road_edge)
        
        # 如果道路线数量少于4条，可能不是十字路口
        if len(road_lines) < 4:
            return False
        
        # 计算道路线的方向角度
        angles = []
        for road_line in road_lines:
            if len(road_line.polyline) >= 2:
                # 计算第一段和最后一段的方向
                start_point = road_line.polyline[0]
                end_point = road_line.polyline[-1]
                
                dx = end_point.x - start_point.x
                dy = end_point.y - start_point.y
                
                if dx != 0 or dy != 0:
                    angle = math.atan2(dy, dx) * 180 / math.pi
                    angles.append(angle)
        
        # 检查是否存在近似垂直的道路线组合
        if len(angles) >= 2:
            for i in range(len(angles)):
                for j in range(i + 1, len(angles)):
                    angle_diff = abs(angles[i] - angles[j])
                    angle_diff = min(angle_diff, 360 - angle_diff)
                    
                    # 如果角度差接近90度，认为是十字路口
                    if abs(angle_diff - 90) < self.intersection_angle_threshold:
                        return True
        
        return False
    
    def analyze_scene_complexity(self, frames):
        """分析场景复杂度"""
        total_pedestrians = 0
        max_pedestrians_per_frame = 0
        intersection_detected = False
        
        for frame in frames:
            # 统计行人数量
            pedestrian_count = self.count_pedestrians_in_frame(frame)
            total_pedestrians += pedestrian_count
            max_pedestrians_per_frame = max(max_pedestrians_per_frame, pedestrian_count)
            
            # 检测十字路口
            if self.detect_intersection(frame):
                intersection_detected = True
        
        avg_pedestrians = total_pedestrians / len(frames) if frames else 0
        
        return {
            'total_pedestrians': total_pedestrians,
            'avg_pedestrians_per_frame': avg_pedestrians,
            'max_pedestrians_per_frame': max_pedestrians_per_frame,
            'is_intersection': intersection_detected
        }
    
    def process_tfrecord(self, tfrecord_file):
        """处理单个tfrecord文件"""
        dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
        scenes_info = []
        
        current_scene_frames = []
        current_context_name = None
        
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(data.numpy())
            
            # 检查是否是新场景
            if current_context_name is None:
                current_context_name = frame.context.name
            
            if frame.context.name != current_context_name:
                # 处理上一个场景
                if current_scene_frames:
                    scene_analysis = self.analyze_scene_complexity(current_scene_frames)
                    scene_analysis['context_name'] = current_context_name
                    scene_analysis['frame_count'] = len(current_scene_frames)
                    scenes_info.append(scene_analysis)
                
                # 开始新场景
                current_scene_frames = [frame]
                current_context_name = frame.context.name
            else:
                current_scene_frames.append(frame)
        
        # 处理最后一个场景
        if current_scene_frames:
            scene_analysis = self.analyze_scene_complexity(current_scene_frames)
            scene_analysis['context_name'] = current_context_name
            scene_analysis['frame_count'] = len(current_scene_frames)
            scenes_info.append(scene_analysis)
        
        return scenes_info
    
    def filter_scenes(self, scenes_info):
        """根据条件筛选场景"""
        pedestrian_dense_scenes = []
        intersection_scenes = []
        both_criteria_scenes = []
        
        for scene in scenes_info:
            is_pedestrian_dense = (
                scene['avg_pedestrians_per_frame'] >= self.pedestrian_threshold or
                scene['max_pedestrians_per_frame'] >= self.pedestrian_threshold * 2
            )
            
            is_intersection = scene['is_intersection']
            
            if is_pedestrian_dense:
                pedestrian_dense_scenes.append(scene)
            
            if is_intersection:
                intersection_scenes.append(scene)
            
            if is_pedestrian_dense and is_intersection:
                both_criteria_scenes.append(scene)
        
        return {
            'pedestrian_dense': pedestrian_dense_scenes,
            'intersection': intersection_scenes,
            'both_criteria': both_criteria_scenes
        }
    
    def save_results(self, filtered_scenes):
        """保存筛选结果"""
        # 保存详细结果
        output_file = os.path.join(self.output_dir, 'filtered_scenes.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_scenes, f, indent=2, ensure_ascii=False)
        
        # 保存场景列表
        for category, scenes in filtered_scenes.items():
            list_file = os.path.join(self.output_dir, f'{category}_scenes.txt')
            with open(list_file, 'w', encoding='utf-8') as f:
                for scene in scenes:
                    f.write(f"{scene['context_name']}\n")
        
        # 保存统计信息
        stats_file = os.path.join(self.output_dir, 'statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
    
    def run(self):
        """运行主程序"""
        print("开始处理Waymo数据集...")
        
        # 加载tfrecord文件
        tfrecord_files = self.load_tfrecord_files()
        
        if not tfrecord_files:
            print("未找到tfrecord文件！")
            return
        
        all_scenes_info = []
        
        # 处理每个tfrecord文件
        for tfrecord_file in tqdm(tfrecord_files, desc="处理文件"):
            try:
                scenes_info = self.process_tfrecord(tfrecord_file)
                all_scenes_info.extend(scenes_info)
                self.stats['total_scenes'] += len(scenes_info)
            except Exception as e:
                print(f"处理文件 {tfrecord_file} 时出错: {e}")
                continue
        
        print(f"总共处理了 {self.stats['total_scenes']} 个场景")
        
        # 筛选场景
        filtered_scenes = self.filter_scenes(all_scenes_info)
        
        # 更新统计信息
        self.stats['pedestrian_dense_scenes'] = len(filtered_scenes['pedestrian_dense'])
        self.stats['intersection_scenes'] = len(filtered_scenes['intersection'])
        self.stats['both_criteria_scenes'] = len(filtered_scenes['both_criteria'])
        
        # 保存结果
        self.save_results(filtered_scenes)
        
        # 打印统计信息
        print("\n=== 筛选结果统计 ===")
        print(f"总场景数: {self.stats['total_scenes']}")
        print(f"行人密集场景: {self.stats['pedestrian_dense_scenes']}")
        print(f"十字路口场景: {self.stats['intersection_scenes']}")
        print(f"同时满足两个条件的场景: {self.stats['both_criteria_scenes']}")
        print(f"\n结果已保存到: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='筛选Waymo数据集中的特定场景')
    parser.add_argument('--data_dir', type=str, default='data/waymo/raw',
                        help='Waymo数据集目录路径')
    parser.add_argument('--output_dir', type=str, default='data/waymo/filtered',
                        help='输出目录路径')
    parser.add_argument('--pedestrian_threshold', type=int, default=5,
                        help='行人密集区域阈值')
    parser.add_argument('--intersection_angle_threshold', type=int, default=60,
                        help='十字路口角度阈值（度）')
    
    args = parser.parse_args()
    
    # 创建筛选器
    filter_tool = WaymoSceneFilter(args.data_dir, args.output_dir)
    filter_tool.pedestrian_threshold = args.pedestrian_threshold
    filter_tool.intersection_angle_threshold = args.intersection_angle_threshold
    
    # 运行筛选
    filter_tool.run()


if __name__ == "__main__":
    main()