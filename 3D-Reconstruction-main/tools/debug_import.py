#!/usr/bin/env python3
"""
调试导入问题的测试脚本
"""

import sys
import os
from pathlib import Path

print("=== 调试信息 ===")
print(f"Python版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")
print(f"Python路径: {sys.path}")
print()

# 检查关键文件是否存在
files_to_check = [
    "scene_editing/insert_instance.py",
    "scene_editing/scene_editing.py",
    "utils/misc.py",
    "datasets/driving_dataset.py"
]

print("=== 文件存在性检查 ===")
for file_path in files_to_check:
    exists = os.path.exists(file_path)
    print(f"{file_path}: {'✓' if exists else '✗'}")

print()

# 尝试导入关键模块
print("=== 模块导入测试 ===")

try:
    from utils.misc import import_str
    print("✓ utils.misc 导入成功")
except Exception as e:
    print(f"✗ utils.misc 导入失败: {e}")

try:
    from datasets.driving_dataset import DrivingDataset
    print("✓ datasets.driving_dataset 导入成功")
except Exception as e:
    print(f"✗ datasets.driving_dataset 导入失败: {e}")

try:
    from omegaconf import OmegaConf
    print("✓ omegaconf 导入成功")
except Exception as e:
    print(f"✗ omegaconf 导入失败: {e}")

# 检查scene_editing模块
print()
print("=== scene_editing 模块检查 ===")
try:
    # 添加当前目录到路径
    current_dir = Path(__file__).parent
    project_root = current_dir.parent if current_dir.name == "scene_editing" else current_dir
    sys.path.insert(0, str(project_root))
    
    import scene_editing.scene_editing as se
    print("✓ scene_editing.scene_editing 导入成功")
    
    # 检查关键函数是否存在
    required_functions = [
        'save_smpl_instance',
        'save_rigid_instance', 
        'load_instance_data',
        'insert_smpl_instance',
        'insert_rigid_instance'
    ]
    
    for func_name in required_functions:
        if hasattr(se, func_name):
            print(f"  ✓ 函数 {func_name} 存在")
        else:
            print(f"  ✗ 函数 {func_name} 不存在")
            
except Exception as e:
    print(f"✗ scene_editing.scene_editing 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 检查完成 ===")