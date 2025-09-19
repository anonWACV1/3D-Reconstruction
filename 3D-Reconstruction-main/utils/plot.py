import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 精度数据
precision_values = [0.9913, 0.9661, 0.9744, 0.9573, 0.9746, 0.9298, 0.9746, 0.9828, 0.9832, 0.9829, 0.9746]

# 位置描述和对应的坐标
positions = [
    "original",           # 0: 原点 (0, 0)
    "Front 0.5m",        # 1: 前方0.5m (0, 0.5)
    "back 0.5m",         # 2: 后方0.5m (0, -0.5)
    "right 3.2m",        # 3: 右方3.2m (3.2, 0)
    "right 1.6m",        # 4: 右方1.6m (1.6, 0)
    "left 3.2m",         # 5: 左方3.2m (-3.2, 0)
    "left 1.6m",         # 6: 左方1.6m (-1.6, 0)
    "right 0.5m",        # 7: 右方0.5m (0.5, 0)
    "left 0.5m",         # 8: 左方0.5m (-0.5, 0)
    "right 0.5m + 15°",  # 9: 右方0.5m旋转15° 
    "left 0.5m - 15°"    # 10: 左方0.5m旋转-15°
]

# 定义坐标系统（以原点为中心）
coordinates = np.array([
    [0, 0],        # 0: original
    [0, 0.5],      # 1: Front 0.5m
    [0, -0.5],     # 2: back 0.5m
    [3.2, 0],      # 3: right 3.2m
    [1.6, 0],      # 4: right 1.6m
    [-3.2, 0],     # 5: left 3.2m
    [-1.6, 0],     # 6: left 1.6m
    [0.5, 0],      # 7: right 0.5m
    [-0.5, 0],     # 8: left 0.5m
    [0.5*np.cos(np.radians(15)), 0.5*np.sin(np.radians(15))],  # 9: right 0.5m + 15°
    [-0.5*np.cos(np.radians(15)), -0.5*np.sin(np.radians(15))] # 10: left 0.5m - 15°
])

def create_heatmap_interpolation():
    """创建插值热力图"""
    # 创建网格用于插值
    x_min, x_max = -4, 4
    y_min, y_max = -1, 1
    
    # 创建更密集的网格
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 100)
    )
    
    # 使用不同的插值方法
    methods = ['linear', 'cubic', 'nearest']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('空间精度热力图 - 不同插值方法', fontsize=16, fontweight='bold')
    
    for i, method in enumerate(methods):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # 插值
        grid_precision = griddata(
            coordinates, precision_values, 
            (grid_x, grid_y), method=method, fill_value=np.nan
        )
        
        # 创建热力图
        im = ax.imshow(grid_precision, extent=[x_min, x_max, y_min, y_max], 
                      origin='lower', cmap='RdYlBu_r', alpha=0.8,
                      vmin=min(precision_values), vmax=max(precision_values))
        
        # 添加原始数据点
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                           c=precision_values, s=100, cmap='RdYlBu_r', 
                           edgecolors='black', linewidth=2,
                           vmin=min(precision_values), vmax=max(precision_values))
        
        # 添加数据标签
        for j, (x, y) in enumerate(coordinates):
            ax.annotate(f'{precision_values[j]:.3f}', 
                       (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{method.title()} 插值', fontsize=14, fontweight='bold')
        ax.set_xlabel('X方向 (米)', fontsize=12)
        ax.set_ylabel('Y方向 (米)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('精度值', fontsize=10)
    
    # 第四个子图：径向基函数插值
    ax = axes[1, 1]
    
    # 使用径向基函数进行平滑插值
    def rbf_interpolation(x, y, xi, yi, function='multiquadric', epsilon=1):
        """径向基函数插值"""
        from scipy.interpolate import Rbf
        rbf = Rbf(x, y, precision_values, function=function, epsilon=epsilon)
        return rbf(xi, yi)
    
    grid_precision_rbf = rbf_interpolation(
        coordinates[:, 0], coordinates[:, 1], 
        grid_x, grid_y, function='thin_plate'
    )
    
    im = ax.imshow(grid_precision_rbf, extent=[x_min, x_max, y_min, y_max], 
                  origin='lower', cmap='RdYlBu_r', alpha=0.8,
                  vmin=min(precision_values), vmax=max(precision_values))
    
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                       c=precision_values, s=100, cmap='RdYlBu_r', 
                       edgecolors='black', linewidth=2,
                       vmin=min(precision_values), vmax=max(precision_values))
    
    for j, (x, y) in enumerate(coordinates):
        ax.annotate(f'{precision_values[j]:.3f}', 
                   (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title('径向基函数插值', fontsize=14, fontweight='bold')
    ax.set_xlabel('X方向 (米)', fontsize=12)
    ax.set_ylabel('Y方向 (米)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('精度值', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def create_detailed_analysis():
    """创建详细分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('空间精度详细分析', fontsize=16, fontweight='bold')
    
    # 1. 原始数据分布
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(precision_values)), precision_values, 
                   color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(precision_values))))
    ax1.set_title('各位置精度值分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('位置编号', fontsize=12)
    ax1.set_ylabel('精度值', fontsize=12)
    ax1.set_xticks(range(len(precision_values)))
    ax1.set_xticklabels([f'P{i}' for i in range(len(precision_values))], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(precision_values):
        ax1.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 空间位置图
    ax2 = axes[0, 1]
    scatter = ax2.scatter(coordinates[:, 0], coordinates[:, 1], 
                         c=precision_values, s=200, cmap='RdYlBu_r', 
                         edgecolors='black', linewidth=2,
                         vmin=min(precision_values), vmax=max(precision_values))
    
    # 添加位置标签
    for i, (x, y) in enumerate(coordinates):
        ax2.annotate(f'P{i}\n{precision_values[i]:.3f}', 
                    (x, y), xytext=(0, 20), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax2.set_title('空间位置分布', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X方向 (米)', fontsize=12)
    ax2.set_ylabel('Y方向 (米)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 添加坐标轴
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('精度值', fontsize=10)
    
    # 3. 距离-精度关系
    ax3 = axes[1, 0]
    distances = np.sqrt(coordinates[:, 0]**2 + coordinates[:, 1]**2)
    ax3.scatter(distances, precision_values, s=100, alpha=0.7, color='blue')
    ax3.set_title('距离与精度关系', fontsize=14, fontweight='bold')
    ax3.set_xlabel('距离原点距离 (米)', fontsize=12)
    ax3.set_ylabel('精度值', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(distances, precision_values, 1)
    p = np.poly1d(z)
    ax3.plot(distances, p(distances), "r--", alpha=0.8, label=f'趋势线 (斜率={z[0]:.4f})')
    ax3.legend()
    
    # 4. 精度统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 计算统计信息
    stats_text = f"""
    精度统计信息：
    
    最高精度：{max(precision_values):.4f} (位置 {precision_values.index(max(precision_values))})
    最低精度：{min(precision_values):.4f} (位置 {precision_values.index(min(precision_values))})
    平均精度：{np.mean(precision_values):.4f}
    标准差：{np.std(precision_values):.4f}
    精度范围：{max(precision_values) - min(precision_values):.4f}
    
    位置分析：
    • 原点精度：{precision_values[0]:.4f}
    • 前后方向平均：{np.mean([precision_values[1], precision_values[2]]):.4f}
    • 左右方向平均：{np.mean([precision_values[i] for i in [3,4,5,6,7,8]]):.4f}
    • 旋转位置平均：{np.mean([precision_values[9], precision_values[10]]):.4f}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def create_contour_heatmap():
    """创建等高线热力图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('等高线热力图', fontsize=16, fontweight='bold')
    
    # 创建插值网格
    x_min, x_max = -4, 4
    y_min, y_max = -1, 1
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 100)
    )
    
    # 使用cubic插值
    grid_precision = griddata(
        coordinates, precision_values, 
        (grid_x, grid_y), method='cubic', fill_value=np.nan
    )
    
    # 填充等高线图
    contour_filled = ax1.contourf(grid_x, grid_y, grid_precision, 
                                 levels=20, cmap='RdYlBu_r', alpha=0.8)
    contour_lines = ax1.contour(grid_x, grid_y, grid_precision, 
                               levels=20, colors='black', alpha=0.4, linewidths=0.5)
    
    # 添加原始数据点
    scatter1 = ax1.scatter(coordinates[:, 0], coordinates[:, 1], 
                          c=precision_values, s=100, cmap='RdYlBu_r', 
                          edgecolors='black', linewidth=2,
                          vmin=min(precision_values), vmax=max(precision_values))
    
    ax1.set_title('填充等高线图', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X方向 (米)', fontsize=12)
    ax1.set_ylabel('Y方向 (米)', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    cbar1 = plt.colorbar(contour_filled, ax=ax1, shrink=0.8)
    cbar1.set_label('精度值', fontsize=10)
    
    # 线性等高线图
    contour_lines2 = ax2.contour(grid_x, grid_y, grid_precision, 
                                levels=15, cmap='RdYlBu_r', linewidths=2)
    ax2.clabel(contour_lines2, inline=True, fontsize=8, fmt='%.3f')
    
    scatter2 = ax2.scatter(coordinates[:, 0], coordinates[:, 1], 
                          c=precision_values, s=100, cmap='RdYlBu_r', 
                          edgecolors='black', linewidth=2,
                          vmin=min(precision_values), vmax=max(precision_values))
    
    ax2.set_title('等高线图（带标签）', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X方向 (米)', fontsize=12)
    ax2.set_ylabel('Y方向 (米)', fontsize=12)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('精度值', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# 运行所有可视化
if __name__ == "__main__":
    print("正在生成空间精度热力图...")
    print(f"数据点数量: {len(precision_values)}")
    print(f"精度范围: {min(precision_values):.4f} - {max(precision_values):.4f}")
    print("="*50)
    
    # 生成不同类型的图表
    create_heatmap_interpolation()
    create_detailed_analysis()
    create_contour_heatmap()
    
    print("所有图表已生成完成！")