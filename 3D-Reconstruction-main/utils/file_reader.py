import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models.video_utils import compute_psnr  # 确保正确导入函数
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import imageio
import shutil
from skimage.metrics import structural_similarity as ssim


def create_comparison_video(npy_dir, png_dir, view, output_path="comparison.mp4", fps=30, exclude_rows=0):
    """
    生成三图对比视频
    
    参数：
    npy_dir: .npy文件目录
    png_dir: .png文件目录
    output_path: 输出视频路径
    fps: 视频帧率
    exclude_rows: 排除的底部行数
    """
    # 创建临时目录
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 生成所有对比图
    frame_paths = []
    for i in tqdm(range(150), desc="生成对比图"):
        npy_path = os.path.join(npy_dir, f"new_frame{i:04d}.npy")
        png_path = os.path.join(png_dir, f"{i:06d}_new_view_{view}.png")
        
        if not (os.path.exists(npy_path) and os.path.exists(png_path)):
            continue
            
        # 生成对比图并保存
        fig = create_combined_figure(npy_path, png_path, exclude_rows)
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)
    
    # 生成视频
    with imageio.get_writer(output_path, fps=fps) as writer:
        for path in tqdm(frame_paths, desc="生成视频"):
            image = imageio.imread(path)
            writer.append_data(image)
    
    print(f"视频已保存至: {output_path}")

    shutil.rmtree(temp_dir)
    print(f"已清理临时文件: {temp_dir}")


def create_combined_figure(npy_path, png_path, exclude_rows):
    """
    创建三图对比的Figure对象
    """
    # 加载数据
    render = np.load(npy_path)
    gt = np.array(Image.open(png_path).convert('RGB'))
    gt = np.array(Image.fromarray(gt).resize((render.shape[1], render.shape[0]))) / 255.0
    
    # 裁剪
    H, W = render.shape[:2]
    render = render[:H-exclude_rows]
    gt = gt[:H-exclude_rows]
    
    # 计算差异
    diff = np.abs(render - gt).mean(axis=-1)
    
    # 创建画布
    fig = plt.figure(figsize=(18, 6), dpi=100)
    
    # 渲染结果
    plt.subplot(131)
    plt.imshow(np.clip(render, 0, 1))
    plt.title("Rendered")
    
    # 真实图像
    plt.subplot(132)
    plt.imshow(gt)
    plt.title("Ground Truth")
    
    # 差异图
    plt.subplot(133)
    plt.imshow(diff, cmap='jet', vmin=0, vmax=0.3)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Difference Map")
    
    # 统一设置
    for ax in fig.axes:
        ax.axis('off')
        ax.axhline(y=H-exclude_rows-1, color='r', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig

def batch_calculate_ssim(npy_dir, png_dir, view, output_csv="ssim_results.csv", exclude_rows=30):
    """
    批量计算SSIM（修复空输出问题）
    """
    start_frame, end_frame = get_frame_range(npy_dir, png_dir, view)
    
    file_pairs = []
    for i in range(start_frame, end_frame+1):
        npy_name = f"new_frame{i:04d}.npy"
        png_name = f"{i:06d}_new_view_{view}.png"
        npy_path = os.path.join(npy_dir, npy_name)
        png_path = os.path.join(png_dir, png_name)
        
        if os.path.exists(npy_path) and os.path.exists(png_path):
            file_pairs.append((npy_path, png_path, i))
        else:
            print(f"文件缺失: {npy_name} 或 {png_name}")
            print(f"文件缺失: {npy_path} 或 {png_path}")

    results = []
    
    for npy_path, png_path, frame_id in tqdm(file_pairs, desc="计算SSIM进度"):
        result = {
            "frame_id": frame_id,
            "ssim": None,  # 初始化默认值
            "npy_path": npy_path,
            "png_path": png_path,
            "error": None
        }
        try:
            # 数据加载
            render = np.load(npy_path)
            img = Image.open(png_path).convert('RGB')
            img = img.resize((render.shape[1], render.shape[0]))
            gt = np.array(img)
            
            # 预处理
            H, W = render.shape[:2]
            render_crop = (np.clip(render[:H-exclude_rows], 0, 1) * 255).astype(np.uint8)
            gt_crop = gt[:H-exclude_rows]
            
            # 计算SSIM
            result["ssim"] = ssim(gt_crop, render_crop, 
                                channel_axis=2, 
                                data_range=255)
            
        except Exception as e:
            result["error"] = str(e)
            print(f"帧 {frame_id} 错误: {str(e)}")
        
        results.append(result)

    # 确保列存在
    df = pd.DataFrame(results, columns=["frame_id", "ssim", "npy_path", "png_path", "error"])
    
    # 保存结果
    df.to_csv(output_csv, index=False)
    
    # 统计信息
    valid_ssim = df[df['ssim'].notnull()]['ssim']
    print(f"\n有效帧数: {len(valid_ssim)}")
    if not valid_ssim.empty:
        print(f"平均SSIM: {valid_ssim.mean():.4f}")
        print(f"最高SSIM: {valid_ssim.max():.4f}")
        print(f"最低SSIM: {valid_ssim.min():.4f}")
    else:
        print("无有效SSIM数据")
    
    return df

def get_frame_range(npy_dir, png_dir, view):
    """动态获取有效帧范围"""
    # 获取npy文件列表
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    npy_frames = {int(f[9:13]) for f in npy_files if f.startswith('new_frame')}
    
    # 获取png文件列表
    png_files = [f for f in os.listdir(png_dir) if f.endswith(f'_new_view_{view}.png')]
    png_frames = {int(f[:6]) for f in png_files}  # 假设文件名格式为000000_new_view_X.png
    
    # 取交集并排序
    common_frames = sorted(npy_frames & png_frames)
    
    if not common_frames:
        raise ValueError(f"目录 {npy_dir} 和 {png_dir} 中没有匹配的帧")
    
    return min(common_frames), max(common_frames)


def batch_calculate_psnr(npy_dir, png_dir, view, output_csv="psnr_results.csv", exclude_rows=30):
    """
    批量计算PSNR
    
    参数：
    npy_dir: .npy文件目录（渲染结果）
    png_dir: .png文件目录（真实图像）
    output_csv: 结果保存路径
    exclude_rows: 要排除的底部行数
    """
    start_frame, end_frame = get_frame_range(npy_dir, png_dir, view)
    
    file_pairs = []
    for i in range(start_frame, end_frame+1):
        npy_name = f"new_frame{i:04d}.npy"
        png_name = f"{i:06d}_new_view_{view}.png" # 注意文件名格式差异
        npy_path = os.path.join(npy_dir, npy_name)
        png_path = os.path.join(png_dir, png_name)
        
        if os.path.exists(npy_path) and os.path.exists(png_path):
            file_pairs.append((npy_path, png_path, i))
        else:
            print(f"文件缺失: {npy_name} 或 {png_name}")
            print(f"文件缺失: {npy_path} 或 {png_path}")

    # 准备结果存储
    results = []
    
    # 批量处理
    for npy_path, png_path, frame_id in tqdm(file_pairs, desc="计算PSNR进度"):
        try:
            # 加载数据
            render = np.load(npy_path)  # [H, W, 3]
            img = Image.open(png_path).convert('RGB')
            img = img.resize((render.shape[1], render.shape[0]))
            gt = np.array(img).astype(np.float32) / 255.0
            
            # 裁剪底部
            H, W = render.shape[:2]
            render_crop = render[:H-exclude_rows]
            gt_crop = gt[:H-exclude_rows]
            
            # 计算PSNR
            psnr = compute_psnr(
                torch.from_numpy(render_crop).permute(2,0,1),
                torch.from_numpy(gt_crop).permute(2,0,1)
            )
            
            results.append({
                "frame_id": frame_id,
                "psnr": psnr,
                "npy_path": npy_path,
                "png_path": png_path
            })
            
        except Exception as e:
            print(f"处理帧 {frame_id} 出错: {str(e)}")
            results.append({
                "frame_id": frame_id,
                "psnr": None,
                "error": str(e)
            })

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # 打印统计信息
    valid_psnr = df[df['psnr'].notnull()]['psnr']
    print(f"\n处理完成！有效帧数: {len(valid_psnr)}")
    print(f"平均PSNR: {valid_psnr.mean():.2f} dB")
    print(f"最高PSNR: {valid_psnr.max():.2f} dB")
    print(f"最低PSNR: {valid_psnr.min():.2f} dB")
    
    return df

def plot_comparison(npy_path, png_path, exclude_rows=0, save_path=None):
    """
    可视化对比（排除底部指定行数）
    """
    # 加载数据
    render = np.load(npy_path)
    gt = np.array(Image.open(png_path).convert('RGB'))
    
    # 调整尺寸并裁剪
    gt = np.array(Image.fromarray(gt).resize((render.shape[1], render.shape[0])))
    H, W = render.shape[0], render.shape[1]
    
    render_cropped = render[:H-exclude_rows, :, :]
    gt_cropped = gt[:H-exclude_rows, :, :] / 255.0
    
    # 创建画布
    plt.figure(figsize=(18, 6))
    
    # 渲染结果（裁剪后）
    plt.subplot(131)
    plt.imshow(np.clip(render_cropped, 0, 1))
    plt.title(f"Rendered (Cropped)\nShape: {render_cropped.shape}")
    
    # 真实图像（裁剪后）
    plt.subplot(132)
    plt.imshow(gt_cropped)
    plt.title(f"Ground Truth (Cropped)\nShape: {gt_cropped.shape}")
    
    # 差异图（裁剪后）
    diff = np.abs(render_cropped - gt_cropped)
    plt.subplot(133)
    im = plt.imshow(diff.mean(axis=-1), cmap='jet', vmin=0, vmax=0.3)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Difference (Cropped)\nMean: {diff.mean():.4f}")
    
    # 绘制裁剪线
    for ax in plt.gcf().axes:
        ax.axhline(y=H-exclude_rows-1, color='r', linestyle='--', linewidth=2, alpha=0.7)
        ax.axis('off')
    
    # # 保存或显示
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     print(f"对比图已保存至: {save_path}")
    plt.show()

def calculate_custom_psnr(npy_path, png_path):
    """
    计算自定义.npy文件与.png图像的PSNR
    
    参数：
    npy_path: .npy文件路径（渲染输出）
    png_path: .png文件路径（真实图像）
    
    返回：
    psnr_value: PSNR值 (dB)
    """
    # 加载.npy数据
    render_data = np.load(npy_path)  # 形状应为 [H, W, 3]
    
    # 处理渲染数据
    if render_data.dtype == np.float32:
        render_rgb = np.clip(render_data, 0.0, 1.0)  # 限制在[0,1]范围
    else:
        raise ValueError(f"不支持的.npy数据类型: {render_data.dtype}")

    # 加载并处理PNG图像
    png_img = Image.open(png_path)
    if png_img.mode == 'RGBA':
        png_img = png_img.convert('RGB')
    
    # 调整尺寸匹配
    if png_img.size != (render_rgb.shape[1], render_rgb.shape[0]):
        png_img = png_img.resize((render_rgb.shape[1], render_rgb.shape[0]))
    
    png_array = np.array(png_img).astype(np.float32) / 255.0  # [H, W, 3]

    # 转换为Tensor
    tensor_render = torch.from_numpy(render_rgb).permute(2, 0, 1)  # [3, H, W]
    tensor_png = torch.from_numpy(png_array).permute(2, 0, 1)      # [3, H, W]

    # 计算PSNR
    return compute_psnr(tensor_render, tensor_png)


def calculate_psnr(img1_path, npz_path, dataset_key='rgbs', target_index=0, exclude_rows=0, visualize=True):
    """
    增强版PSNR计算函数，支持可视化和裁剪
    
    参数：
    img1_path: PNG图像路径 
    npz_path: NPZ文件路径
    dataset_key: NPZ文件中RGB数据的键名，默认为'rgbs'
    target_index: 要比较的NPZ数据中的图像索引，默认为0
    exclude_rows: 要排除的底部行数，默认为0
    visualize: 是否生成可视化对比图，默认为True
    
    返回：
    psnr_value: PSNR值 (dB)
    """
    # 加载并预处理数据
    npz_data = np.load(npz_path)
    rgb_npz = npz_data[dataset_key]
    
    # 检查索引有效性
    if target_index >= rgb_npz.shape[0]:
        raise IndexError(f"索引{target_index}超出范围(总样本数: {rgb_npz.shape[0]})")
    
    # 提取目标图像
    render = rgb_npz[target_index]  # [H, W, 3]
    if render.dtype == np.uint8:
        render = render.astype(np.float32) / 255.0
    else:
        render = np.clip(render, 0.0, 1.0)
    
    # 加载并处理PNG
    img_pil = Image.open(img1_path).convert('RGB')
    img_pil = img_pil.resize((render.shape[1], render.shape[0]))
    gt = np.array(img_pil).astype(np.float32) / 255.0
    
    # 裁剪底部行
    H, W = render.shape[:2]
    render_cropped = render[:H-exclude_rows]
    gt_cropped = gt[:H-exclude_rows]
    
    # 计算PSNR
    tensor_render = torch.from_numpy(render_cropped).permute(2, 0, 1)
    tensor_gt = torch.from_numpy(gt_cropped).permute(2, 0, 1)
    psnr = compute_psnr(tensor_render, tensor_gt)
    
    # 可视化
    if visualize:
        plt.figure(figsize=(18, 6))
        
        # 渲染结果
        plt.subplot(131)
        plt.imshow(render_cropped)
        plt.title(f"Rendered (Cropped)\nPSNR: {psnr:.2f}dB")
        
        # 真实图像
        plt.subplot(132)
        plt.imshow(gt_cropped)
        plt.title("Ground Truth")
        
        # 差异图
        plt.subplot(133)
        diff = np.abs(render_cropped - gt_cropped).mean(axis=-1)
        plt.imshow(diff, cmap='jet', vmin=0, vmax=0.2)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"Difference Map\nMax: {diff.max():.2f}")
        
        for ax in plt.gcf().axes:
            ax.axhline(y=H-exclude_rows-1, color='r', linestyle='--', alpha=0.6)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return psnr

def calculate_masked_metrics(npy_dir, png_dir, view, mask_dir, output_csv="masked_metrics.csv", exclude_rows=0, visualize=False, vis_dir=None):
    """
    计算掩码区域的PSNR和SSIM
    """
    # 在函数开头添加可视化目录检查
    if visualize and not vis_dir:
        raise ValueError("启用可视化时必须指定vis_dir参数")
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    start_frame, end_frame = get_frame_range(npy_dir, png_dir, view)
    
    file_triplets = []
    for i in range(start_frame, end_frame+1):
        npy_name = f"new_frame{i:04d}.npy"
        png_name = f"{i:06d}_new_view_{view}.png"
        mask_name = f"{i:06d}_nonrigid_{view}.png"
        
        npy_path = os.path.join(npy_dir, npy_name)
        png_path = os.path.join(png_dir, png_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        if all(os.path.exists(p) for p in [npy_path, png_path, mask_path]):
            file_triplets.append((npy_path, png_path, mask_path, i))
        else:
            print(f"文件缺失: {npy_name}, {png_name} 或 {mask_name}")
            print(f"文件缺失:\n - NPY路径: {npy_path}\n - PNG路径: {png_path}\n - 掩码路径: {mask_path}\n")

    results = []
    
    for npy_path, png_path, mask_path, frame_id in tqdm(file_triplets, desc="处理进度"):
        result = {"frame_id": frame_id, "psnr": None, "ssim": None, "error": None}
        try:
            # 加载数据
            render = np.load(npy_path)  # [H, W, 3] float32
            gt = np.array(Image.open(png_path).convert('RGB')) / 255.0  # [H, W, 3] float32
            mask = np.array(Image.open(mask_path).convert('L'))  # [H, W] uint8
            
            # 统一尺寸
            H, W = render.shape[:2]
            gt = np.array(Image.fromarray((gt*255).astype(np.uint8)).resize((W, H))) / 255.0
            mask = np.array(Image.fromarray(mask).resize((W, H))) > 128  # 二值化
            
            # 应用掩码
            masked_render = render * mask[..., None]  # [H, W, 3]
            masked_gt = gt * mask[..., None]
            
            # 裁剪底部
            masked_render = masked_render[:H-exclude_rows]
            masked_gt = masked_gt[:H-exclude_rows]
            mask = mask[:H-exclude_rows]
            
            # 提取掩码区域的像素
            mask_flat = mask.flatten()
            render_pixels = masked_render.reshape(-1, 3)[mask_flat]
            gt_pixels = masked_gt.reshape(-1, 3)[mask_flat]
            
            # 转换为Tensor并计算
            tensor_render = torch.from_numpy(render_pixels).permute(1,0)
            tensor_gt = torch.from_numpy(gt_pixels).permute(1,0)
            result["psnr"] = compute_psnr(tensor_render, tensor_gt)

            # 计算SSIM（仅掩码区域）
            # 转换数据类型
            render_uint8 = (np.clip(masked_render, 0, 1) * 255).astype(np.uint8)  # 添加这行
            gt_uint8 = (masked_gt * 255).astype(np.uint8)  # 添加这行
            
            # 创建仅包含掩码区域的图像块
            y, x = np.where(mask)
            if len(y) == 0 or len(x) == 0:  # 添加空掩码检查
                raise ValueError("掩码区域为空")
                
            min_y, max_y = np.min(y), np.max(y)
            min_x, max_x = np.min(x), np.max(x)
            
            # 提取ROI区域
            render_roi = render_uint8[min_y:max_y+1, min_x:max_x+1]
            gt_roi = gt_uint8[min_y:max_y+1, min_x:max_x+1]
            mask_roi = mask[min_y:max_y+1, min_x:max_x+1]
            
            # 添加尺寸验证
            if render_roi.shape != gt_roi.shape:
                raise ValueError(f"ROI尺寸不匹配: render {render_roi.shape} vs gt {gt_roi.shape}")
            
            result["ssim"] = ssim(gt_roi, render_roi,
                                channel_axis=2,
                                data_range=255,
                                win_size=11,
                                use_sample_covariance=False,
                                mask=mask_roi)
            
            # 新增可视化部分
            if visualize:
                fig = plt.figure(figsize=(18, 6), dpi=100)
                
                # 渲染结果
                plt.subplot(131)
                plt.imshow(np.clip(masked_render, 0, 1))
                plt.title(f"Masked Render\nPSNR: {result['psnr']:.2f}dB")
                
                # 真实图像
                plt.subplot(132)
                plt.imshow(masked_gt)
                plt.title("Masked Ground Truth")
                
                # 差异图
                plt.subplot(133)
                diff = np.abs(masked_render - masked_gt).mean(axis=-1)
                plt.imshow(diff, cmap='jet', vmin=0, vmax=0.3)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title(f"Difference Map\nSSIM: {result['ssim']:.4f}")
                
                # # 绘制掩码边界
                # for ax in fig.axes:
                #     y, x = np.where(mask)
                #     if len(y) > 0 and len(x) > 0:
                #         rect = plt.Rectangle((x.min(), y.min()), 
                #                            x.max()-x.min(), 
                #                            y.max()-y.min(),
                #                            fill=False, 
                #                            edgecolor='lime', 
                #                            linewidth=2)
                #         ax.add_patch(rect)
                #     ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"frame_{frame_id:04d}_masked_comparison.png"), bbox_inches='tight')
                plt.close()


        except Exception as e:
            result["error"] = str(e)
            print(f"帧 {frame_id} 错误: {str(e)}")
        
        results.append(result)

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # 打印统计信息
    valid_psnr = df['psnr'].dropna()
    valid_ssim = df['ssim'].dropna()
    
    # 过滤异常值（PSNR > 100dB 或 SSIM > 0.999 视为异常）
    filtered_psnr = valid_psnr[valid_psnr < 100]
    filtered_ssim = valid_ssim[valid_ssim < 0.999]
    
    print("\nPSNR统计:")
    print(f"总有效帧数: {len(valid_psnr)}")
    print(f"过滤后帧数: {len(filtered_psnr)} (排除{len(valid_psnr)-len(filtered_psnr)}个异常值)")
    if not filtered_psnr.empty:
        print(f"平均: {filtered_psnr.mean():.2f} dB")
        print(f"范围: [{filtered_psnr.min():.2f}, {filtered_psnr.max():.2f}]")
    
    print("\nSSIM统计:")
    print(f"总有效帧数: {len(valid_ssim)}")
    print(f"过滤后帧数: {len(filtered_ssim)} (排除{len(valid_ssim)-len(filtered_ssim)}个异常值)")
    if not filtered_ssim.empty:
        print(f"平均: {filtered_ssim.mean():.4f}")
        print(f"范围: [{filtered_ssim.min():.4f}, {filtered_ssim.max():.4f}]")
    
    return df

def batch_calculate_masked_metrics(npy_base_dir, png_base_dir, mask_base_dir, output_dir="masked_metrics", exclude_rows=30, views=range(10)):
    """
    批量计算多视角的掩码指标
    
    参数：
    npy_base_dir: raw_fixed_offset文件夹的父目录
    png_base_dir: new_view图像目录的父目录
    mask_base_dir: 掩码文件根目录
    output_dir: 结果输出目录
    views: 要处理的视角编号列表（默认0-9）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for view in views:
        print(f"\n正在处理掩码视角 {view}...")
        # 构建路径
        npy_dir = os.path.join(npy_base_dir, f"raw_fixed_offset_{view+1}")
        png_dir = os.path.join(png_base_dir, "new_view")
        mask_dir = os.path.join(mask_base_dir, "nonrigid")  # 根据实际目录结构调整
        
        # 计算掩码指标
        results = calculate_masked_metrics(
            npy_dir=npy_dir,
            png_dir=png_dir,
            mask_dir=mask_dir,
            output_csv=os.path.join(output_dir, f"masked_metrics_view_{view}.csv"),
            exclude_rows=exclude_rows,
            visualize=False,  # 批量处理时建议关闭可视化
            view=view
        )

def create_masked_video(npy_dir, png_dir, mask_dir, output_path="masked_comparison.mp4", fps=30, exclude_rows=0):
    """
    生成掩码区域对比视频
    参数：
    output_path: 输出视频路径
    fps: 视频帧率
    """
    # 创建临时目录
    temp_dir = os.path.join(os.path.dirname(output_path), "masked_temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 生成所有对比图
    frame_paths = []
    for i in tqdm(range(150), desc="生成掩码对比图"):
        npy_path = os.path.join(npy_dir, f"new_frame{i:04d}.npy")
        png_path = os.path.join(png_dir, f"{i:06d}_new_view_3.png")
        mask_path = os.path.join(mask_dir, f"{i:06d}_rigid_3.png")
        
        if not all(os.path.exists(p) for p in [npy_path, png_path, mask_path]):
            continue
            
        # 生成对比图
        fig = plot_masked_comparison(npy_path, png_path, mask_path, exclude_rows)
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)
    
    # 生成视频
    with imageio.get_writer(output_path, fps=fps) as writer:
        for path in tqdm(sorted(frame_paths), desc="生成视频"):
            image = imageio.imread(path)
            writer.append_data(image)
    
    # 清理临时文件
    shutil.rmtree(temp_dir)
    print(f"\n视频已保存至: {output_path}")
    print(f"已清理临时文件: {temp_dir}")

def plot_masked_comparison(npy_path, png_path, mask_path, exclude_rows=0):
    """
    生成单帧掩码对比图
    """
    # 加载数据
    render = np.load(npy_path)
    gt = np.array(Image.open(png_path).convert('RGB')) / 255.0
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # 预处理
    H, W = render.shape[:2]
    gt = np.array(Image.fromarray((gt*255).astype(np.uint8)).resize((W, H))) / 255.0
    mask = np.array(Image.fromarray(mask).resize((W, H))) > 128
    
    # 应用掩码
    masked_render = render * mask[..., None]
    masked_gt = gt * mask[..., None]
    
    # 裁剪底部
    masked_render = masked_render[:H-exclude_rows]
    masked_gt = masked_gt[:H-exclude_rows]
    mask = mask[:H-exclude_rows]
    
    # 创建画布
    fig = plt.figure(figsize=(18, 6), dpi=100)
    
    # 渲染结果
    plt.subplot(131)
    plt.imshow(np.clip(masked_render, 0, 1))
    plt.title("Masked Render")
    
    # 真实图像
    plt.subplot(132)
    plt.imshow(masked_gt)
    plt.title("Masked Ground Truth")
    
    # 差异图
    plt.subplot(133)
    diff = np.abs(masked_render - masked_gt).mean(axis=-1)
    plt.imshow(diff, cmap='jet', vmin=0, vmax=0.3)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Difference Map")
    
    # 绘制掩码边界
    # y, x = np.where(mask)
    # if len(y) > 0 and len(x) > 0:
    #     for ax in fig.axes:
    #         rect = plt.Rectangle((x.min(), y.min()), 
    #                            x.max()-x.min(), 
    #                            y.max()-y.min(),
    #                            fill=False, 
    #                            edgecolor='lime',
    #                            linewidth=1)
    #         ax.add_patch(rect)
    
    for ax in fig.axes:
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def batch_calculate_metrics(npy_base_dir, png_base_dir, output_dir="metrics", exclude_rows=30, views=range(10)):
    """
    批量计算多视角的PSNR和SSIM
    
    参数：
    npy_base_dir: raw_fixed_offset文件夹的父目录
    png_base_dir: new_view图像目录的父目录
    output_dir: 结果输出目录
    views: 要处理的视角编号列表（默认0-9）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for view in views:
        print(f"\n正在处理视角 {view}...")
        npy_dir = os.path.join(npy_base_dir, f"raw_fixed_offset_{view+1}")
        png_dir = os.path.join(png_base_dir, f"new_view")
        
        # 计算PSNR
        psnr_df = batch_calculate_psnr(
            npy_dir=npy_dir,
            png_dir=png_dir,
            output_csv=os.path.join(output_dir, f"psnr_view_{view}.csv"),
            exclude_rows=exclude_rows, 
            view=view
        )
        
        # 计算SSIM
        ssim_df = batch_calculate_ssim(
            npy_dir=npy_dir,
            png_dir=png_dir,
            output_csv=os.path.join(output_dir, f"ssim_view_{view}.csv"),
            exclude_rows=exclude_rows , 
            view=view
        )



class ResultsReader:
    def __init__(self, file_path):
        """
        初始化结果读取器，支持HDF5和NPZ格式
        :param file_path: 文件路径（.h5/.hdf5/.npz）
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
            
        self.data = None
        self.file_type = None
        self._open_file()

    def _open_file(self):
        """打开文件并识别格式"""
        suffix = self.file_path.suffix.lower()
        
        if suffix in ['.h5', '.hdf5']:
            self.file_type = 'h5'
            self.data = h5py.File(self.file_path, 'r')
            print(f"成功加载HDF5文件: {self.file_path.name}")
        elif suffix == '.npz':
            self.file_type = 'npz'
            self.data = np.load(self.file_path, allow_pickle=True)
            print(f"成功加载NPZ文件: {self.file_path.name}")
        else:
            raise ValueError(f"不支持的格式: {suffix}")

        # 统一数据访问接口
        self.datasets = list(self.data.keys()) if self.file_type == 'h5' else list(self.data.files)
        print(f"包含的数据集: {self.datasets}")

    def get_dataset_info(self):
        """获取所有数据集的信息"""
        info = {}
        for name in self.datasets:
            if self.file_type == 'h5':
                dataset = self.data[name]
                info[name] = {
                    'shape': dataset.shape,
                    'dtype': dataset.dtype,
                    'size': dataset.size
                }
            else:  # npz
                arr = self.data[name]
                info[name] = {
                    'shape': arr.shape,
                    'dtype': arr.dtype,
                    'size': arr.size
                }
        return info

    def visualize_rgbs(self, max_images=5):
        """可视化RGB图像数据"""
        if 'rgbs' not in self.datasets:
            print("警告：文件中没有rgbs数据集")
            return

        rgbs = self.data['rgbs'][()] if self.file_type == 'h5' else self.data['rgbs']
        
        # 处理不同存储格式
        if rgbs.ndim == 5:  # (N, T, H, W, C)
            rgbs = rgbs[:,0]  # 取第一个时间步
        
        print(f"找到 {rgbs.shape[0]} 张RGB图像")
        
        plt.figure(figsize=(15, 5))
        for i in range(min(max_images, rgbs.shape[0])):
            plt.subplot(1, min(max_images, rgbs.shape[0]), i+1)
            plt.imshow(rgbs[i])
            plt.title(f"Frame {i}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def save_images(self, dataset_name, output_dir):
        """保存指定数据集中的图像到目录"""
        if dataset_name not in self.datasets:
            print(f"错误：数据集 {dataset_name} 不存在")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images = self.data[dataset_name][()] if self.file_type == 'h5' else self.data[dataset_name]
        
        # 处理不同时间维度
        if images.ndim == 5:  # (N, T, H, W, C)
            images = images[:,0]  # 取第一个时间步
        
        print(f"正在保存 {images.shape[0]} 张图像到 {output_path}...")
        
        for i in range(images.shape[0]):
            img = images[i]
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)
            plt.imsave(output_path / f"{dataset_name}_{i:04d}.png", img)
        
        print("保存完成！")

    def get_metrics(self):
        """获取评估指标"""
        metrics = {}
        for name in ['psnr', 'ssim', 'lpips']:
            if name in self.datasets:
                metrics[name] = self.data[name][()] if self.file_type == 'h5' else self.data[name].item()
        return metrics

    def close(self):
        """关闭文件"""
        if self.data:
            if self.file_type == 'h5':
                self.data.close()
            elif self.file_type == 'npz':
                self.data.close()
            print("文件已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 使用示例
if __name__ == "__main__":
    import argparse

    root_path = '/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti'
    data_path = 'training_20250508_103023_DynamicObjectCrossing_1'
    npy_folder = f"/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/{data_path}/videos/novel_30000/raw_fixed_offset_1"
    png_folder = f"/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data/custom_kitti/raw/2025_02_20/{data_path}/new_view"
    
    save_path = f'{root_path}/{data_path}/metrics'

    # batch_calculate_metrics(
    #     npy_base_dir="/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/training_20250508_103023_DynamicObjectCrossing_1/videos_eval/novel_30000",
    #     png_base_dir="/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data/custom_kitti/raw/2025_02_20/training_20250508_103023_DynamicObjectCrossing_1",
    #     output_dir=f"{save_path}/all_views_metrics",
    #     views=range(10)  # 0-9
    # )

    # mask_dir = "/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data/custom_kitti/raw/2025_02_20/training_20250505_160554_FollowLeadingVehicleWithObstacle_1/new_mask/rigid"
    # results = calculate_masked_metrics(
    #     npy_dir=npy_folder,
    #     png_dir=png_folder,
    #     mask_dir=mask_dir,
    #     output_csv=f"{save_path}/masked_metrics.csv",
    #     exclude_rows=30
    # )

    # batch_calculate_masked_metrics(
    #     npy_base_dir="/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/training_20250508_103023_DynamicObjectCrossing_1/videos_eval/novel_30000",
    #     png_base_dir="/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data/custom_kitti/raw/2025_02_20/training_20250508_103023_DynamicObjectCrossing_1",
    #     mask_base_dir="/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data/custom_kitti/raw/2025_02_20/training_20250508_103023_DynamicObjectCrossing_1/new_mask",  # 掩码文件根目录
    #     output_dir=f"{save_path}/all_masked_metrics",
    #     views=range(5)
    # )

    # create_masked_video(
    #     npy_dir=npy_folder,
    #     png_dir=png_folder,
    #     mask_dir=mask_dir,
    #     output_path=f"{save_path}/masked_comparison.mp4",
    #     fps=15,
    #     exclude_rows=30
    # )

    create_comparison_video(
    npy_dir=npy_folder,
    png_dir=png_folder,
    view=1,
    output_path=f"{save_path}/comparison_video.mp4",
    fps=10,
    exclude_rows=0
    )
    
    png_path = '/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data_generator/data/training_20250424_113506_SignalizedJunctionLeftTurn_5/image/000019_camera_0.png'
    npz_path = '/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/training_20250424_113506_SignalizedJunctionLeftTurn_5/render_results/full_set_trimmed_20250429_135411.npz'

    psnr_value = calculate_psnr(png_path, npz_path,target_index=0,
    exclude_rows=0,
    visualize=True)
    print(f"PSNR: {psnr_value:.2f} dB")

    # 使用示例
    npy_file = "/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/training_20250424_113506_SignalizedJunctionLeftTurn_5/videos_eval/novel_30000/raw_rgb/new_frame0130.npy"
    png_file = "/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data_generator/data/training_20250424_113506_SignalizedJunctionLeftTurn_5/new_view/000130_new_view_3.png"

    psnr = calculate_custom_psnr(npy_file, png_file)
    print(f"PSNR between render and GT: {psnr:.2f} dB")

    plot_comparison(npy_file, png_file)
    
    parser = argparse.ArgumentParser(description='HDF5结果文件分析工具')
    parser.add_argument('file', help='HDF5文件路径')
    parser.add_argument('--dataset', help='指定要操作的数据集名称')
    parser.add_argument('--save', help='保存图像到指定目录')
    args = parser.parse_args()

    try:
        with ResultsReader(args.file) as reader:
            # 显示基本信息
            print("\n文件信息：")
            print("="*40)
            for name, info in reader.get_dataset_info().items():
                print(f"{name}:")
                print(f"  |- 形状: {info['shape']}")
                print(f"  |- 类型: {info['dtype']}")
                print(f"  |- 大小: {info['size']:,}")
            
            # 显示评估指标
            print("\n评估指标：")
            print("="*40)
            metrics = reader.get_metrics()
            for k, v in metrics.items():
                print(f"{k.upper()}: {v:.4f}")

            # 可视化RGB图像
            if args.dataset == 'rgbs':
                reader.visualize_rgbs()
                
            # 保存图像
            if args.save and args.dataset:
                reader.save_images(args.dataset, args.save)
                
    except Exception as e:
        print(f"发生错误: {str(e)}")