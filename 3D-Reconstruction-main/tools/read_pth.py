import torch
import os
import argparse
from collections import OrderedDict

def load_and_print_pth(pth_path, verbose=True):
    """
    加载PyTorch的.pth文件并打印其中的参数信息
    
    参数:
        pth_path (str): .pth文件的路径
        verbose (bool): 是否打印详细信息，默认为True
    
    返回:
        加载的参数字典
    """
    # 检查文件是否存在
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"找不到文件: {pth_path}")
    
    # 加载参数
    print(f"\n正在加载文件: {pth_path}")
    state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
    
    # 判断是模型状态字典还是整个模型
    if isinstance(state_dict, OrderedDict) or isinstance(state_dict, dict):
        print("文件包含参数字典 (state_dict)\n")
        params_count = 0
        total_size = 0
        
        # 打印参数信息
        print("参数列表:")
        print("-" * 100)
        print(f"{'参数名':<50} {'形状':<30} {'数据类型':<15} {'参数量':<10}")
        print("-" * 100)
        
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                shape_str = str(list(tensor.shape))
                dtype_str = str(tensor.dtype).split('.')[-1]
                num_params = tensor.numel()
                params_count += num_params
                total_size += num_params * tensor.element_size()
                
                if verbose:
                    print(f"{key:<50} {shape_str:<30} {dtype_str:<15} {num_params:<10,}")
        
        # 打印总参数数量和模型大小
        print("-" * 100)
        print(f"总参数数量: {params_count:,}")
        print(f"模型大小: {total_size / (1024 * 1024):.2f} MB")
        
    elif hasattr(state_dict, 'state_dict'):
        print("文件包含完整的模型对象\n")
        print(f"模型类型: {type(state_dict)}")
        
        if hasattr(state_dict, 'parameters'):
            params_count = sum(p.numel() for p in state_dict.parameters())
            print(f"总参数数量: {params_count:,}")
        
        if verbose:
            print("\n模型结构:")
            print(state_dict)
    else:
        print(f"加载的数据类型: {type(state_dict)}")
        if verbose:
            print("\n内容:")
            print(state_dict)
    
    return state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch模型参数读取工具')
    parser.add_argument('pth_path', type=str, help='.pth文件的路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='是否显示详细信息')
    args = parser.parse_args()
    
    try:
        state_dict = load_and_print_pth(args.pth_path, args.verbose)
    except Exception as e:
        print(f"错误: {e}")