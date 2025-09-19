import numpy as np
import os

def analyze_specific_npy_files(folder_path, file_number):
    # 构建文件名
    in_file = f"{file_number}_in.npy"
    out_file = f"{file_number}_out.npy"
    
    # 检查文件是否存在
    if not os.path.exists(os.path.join(folder_path, in_file)):
        print(f"输入文件 {in_file} 不存在")
        return
    if not os.path.exists(os.path.join(folder_path, out_file)):
        print(f"输出文件 {out_file} 不存在")
        return
    
    # 分析输入文件
    print("\n=== 输入文件分析 ===")
    analyze_single_file(folder_path, in_file)
    
    # 分析输出文件
    print("\n=== 输出文件分析 ===")
    analyze_single_file(folder_path, out_file)

def analyze_single_file(folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    try:
        data = np.load(file_path)
        
        print(f"\n文件: {filename}")
        print(f"数据类型: {data.dtype}")
        print(f"数组形状: {data.shape}")
        print(f"数组大小: {data.size}")
        print(f"维度数: {data.ndim}")
        
        if data.size > 0:
            print(f"最小值: {np.min(data)}")
            print(f"最大值: {np.max(data)}")
            print(f"平均值: {np.mean(data)}")
            print(f"标准差: {np.std(data)}")
            
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {str(e)}")

# 使用示例
motion_folder = "/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/motion"
file_number = 20  # 可以修改为您想分析的编号
analyze_specific_npy_files(motion_folder, file_number)