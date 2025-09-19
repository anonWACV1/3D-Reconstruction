import argparse
import numpy as np
from plyfile import PlyData, PlyElement
import os

def read_ply_file(file_path):
    """
    读取PLY文件并打印其参数
    
    参数:
        file_path: PLY文件的路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            return
            
        # 读取PLY文件
        ply_data = PlyData.read(file_path)
        
        # 打印基本信息
        print(f"\n{'='*50}")
        print(f"PLY文件: {file_path}")
        print(f"{'='*50}")
        
        # 尝试获取文件格式和版本（使用安全的方式）
        try:
            if hasattr(ply_data, 'format'):
                print(f"\n文件格式: {ply_data.format}")
            else:
                # 尝试其他方式确定格式
                with open(file_path, 'rb') as f:
                    header = f.read(100)  # 读取前100个字节
                    if b'format ascii' in header:
                        print("\n文件格式: ascii")
                    elif b'format binary' in header:
                        print("\n文件格式: binary")
                    else:
                        print("\n文件格式: 未知")
                        
            if hasattr(ply_data, 'version'):
                print(f"文件版本: {ply_data.version}")
        except Exception as e:
            print(f"获取文件格式信息失败: {e}")
            
        # 遍历每个元素（通常是'vertex'和'face'）
        for element in ply_data.elements:
            print(f"\n元素: {element.name}")
            print(f"元素数量: {len(element)}")
            
            # 打印属性信息
            print("属性:")
            for prop in element.properties:
                try:
                    prop_name = prop.name
                    if hasattr(prop, 'val_dtype'):
                        prop_type = prop.val_dtype
                    else:
                        prop_type = "未知类型"
                    print(f"  - {prop_name}: {prop_type}")
                except Exception as e:
                    print(f"  - 获取属性信息失败: {e}")
            
            # 如果是顶点元素，打印一些统计信息
            if element.name == 'vertex':
                try:
                    # 检查是否有x,y,z坐标
                    coord_props = ['x', 'y', 'z']
                    available_coords = [prop.name for prop in element.properties if prop.name in coord_props]
                    
                    if len(available_coords) == 3:  # 有完整的xyz坐标
                        vertices = np.vstack([element[prop] for prop in available_coords]).T
                        if len(vertices) > 0:
                            print("\n顶点统计:")
                            print(f"  - 顶点数量: {len(vertices)}")
                            print(f"  - 坐标范围 X: [{vertices[:, 0].min()}, {vertices[:, 0].max()}]")
                            print(f"  - 坐标范围 Y: [{vertices[:, 1].min()}, {vertices[:, 1].max()}]")
                            print(f"  - 坐标范围 Z: [{vertices[:, 2].min()}, {vertices[:, 2].max()}]")
                    else:
                        print("\n顶点统计:")
                        print(f"  - 顶点数量: {len(element)}")
                        print(f"  - 可用坐标: {', '.join(available_coords)}")
                    
                    # 检查是否有颜色信息
                    color_props = [prop.name for prop in element.properties if prop.name in ['red', 'green', 'blue', 'alpha', 'r', 'g', 'b', 'a']]
                    if color_props:
                        print("\n颜色信息: 存在")
                        print(f"  - 颜色通道: {', '.join(color_props)}")
                except Exception as e:
                    print(f"\n处理顶点数据时出错: {e}")
            
            # 如果是面元素
            elif element.name == 'face':
                try:
                    print("\n面统计:")
                    print(f"  - 面数量: {len(element)}")
                    
                    # 获取第一个面的顶点数以检查多边形类型
                    if len(element) > 0:
                        first_face = element[0]
                        # 处理不同格式的面数据
                        if hasattr(first_face, 'vertex_indices'):
                            vertices_per_face = len(first_face.vertex_indices)
                        elif len(first_face.dtype.names) > 0:
                            # 尝试找到存储顶点索引的字段
                            for field in first_face.dtype.names:
                                if isinstance(first_face[field], np.ndarray):
                                    vertices_per_face = len(first_face[field])
                                    break
                            else:
                                vertices_per_face = None
                        else:
                            vertices_per_face = None
                            
                        if vertices_per_face:
                            if vertices_per_face == 3:
                                polygon_type = "三角形"
                            elif vertices_per_face == 4:
                                polygon_type = "四边形"
                            else:
                                polygon_type = f"{vertices_per_face}边形"
                            print(f"  - 多边形类型: {polygon_type}")
                        else:
                            print("  - 无法确定多边形类型")
                except Exception as e:
                    print(f"\n处理面数据时出错: {e}")
        
        # 打印其他可能存在的元素
        other_elements = [elem.name for elem in ply_data.elements if elem.name not in ['vertex', 'face']]
        if other_elements:
            print(f"\n其他元素: {', '.join(other_elements)}")
        
        print("\n分析完成!")
        
    except Exception as e:
        print(f"读取PLY文件时出错: {e}")
        
        # 尝试读取文件头以获取更多信息
        try:
            print("\n尝试读取文件头:")
            with open(file_path, 'rb') as f:
                header_lines = []
                for _ in range(20):  # 读取前20行作为头部
                    line = f.readline().decode('utf-8', errors='ignore').strip()
                    if not line or line == 'end_header':
                        header_lines.append(line)
                        break
                    header_lines.append(line)
                    
                print('\n'.join(header_lines))
        except Exception as header_error:
            print(f"读取文件头时出错: {header_error}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='读取PLY文件并打印其参数')
    parser.add_argument('file_path', type=str, help='PLY文件的路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 读取并分析PLY文件
    read_ply_file(args.file_path)

if __name__ == "__main__":
    main()