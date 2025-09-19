import os
import cv2
import glob
import numpy as np
from pathlib import Path
import re

def create_video_from_images(image_folder, output_video_path, fps=10):
    """
    将指定文件夹中的前置相机图片转换为视频
    
    Args:
        image_folder: 包含图片的文件夹路径
        output_video_path: 输出视频的路径
        fps: 视频帧率，默认10fps
    """
    # 获取所有前置相机图片（以_0.jpg结尾的文件）
    image_pattern = os.path.join(image_folder, '*_0.jpg')
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"警告: 在 {image_folder} 中未找到前置相机图片")
        return False
    
    # 按文件名中的数字排序
    def extract_number(filename):
        # 提取文件名中的数字部分用于排序
        match = re.search(r'(\d+)_0\.jpg', os.path.basename(filename))
        return int(match.group(1)) if match else 0
    
    image_files.sort(key=extract_number)
    
    # 读取第一张图片以获取尺寸
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"错误: 无法读取图片 {image_files[0]}")
        return False
    
    height, width, channels = first_image.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"正在处理 {len(image_files)} 张图片...")
    
    # 逐张读取图片并写入视频
    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        if img is None:
            print(f"警告: 无法读取图片 {image_file}")
            continue
        
        # 确保图片尺寸一致
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
        
        # 显示进度
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(image_files)} 张图片")
    
    # 释放资源
    out.release()
    print(f"视频已保存至: {output_video_path}")
    return True

def main():
    # 基础路径
    base_path = "data/waymo/processed/training"
    output_dir = "data/waymo/videos"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理场景500到600
    start_scene = 500
    end_scene = 600
    
    successful_videos = 0
    failed_videos = 0
    
    print(f"开始处理场景 {start_scene} 到 {end_scene} 的前置相机图片...")
    print("=" * 60)
    
    for scene_num in range(start_scene, end_scene + 1):
        scene_folder = os.path.join(base_path, str(scene_num), "images")
        output_video = os.path.join(output_dir, f"scene_{scene_num:03d}_front_camera.mp4")
        
        print(f"\n处理场景 {scene_num}:")
        print(f"  输入文件夹: {scene_folder}")
        print(f"  输出视频: {output_video}")
        
        # 检查输入文件夹是否存在
        if not os.path.exists(scene_folder):
            print(f"  跳过: 文件夹 {scene_folder} 不存在")
            failed_videos += 1
            continue
        
        # 创建视频
        success = create_video_from_images(scene_folder, output_video, fps=10)
        
        if success:
            successful_videos += 1
            print(f"  ✓ 成功创建视频")
        else:
            failed_videos += 1
            print(f"  ✗ 创建视频失败")
    
    print("\n" + "=" * 60)
    print(f"处理完成!")
    print(f"成功创建视频: {successful_videos} 个")
    print(f"失败: {failed_videos} 个")
    print(f"总共处理场景: {end_scene - start_scene + 1} 个")
    print(f"视频保存位置: {output_dir}")

if __name__ == "__main__":
    main()