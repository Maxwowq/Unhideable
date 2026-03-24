"""
Attention 可视化脚本
读取保存的 attention 数据并生成热图
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import zoom

# 默认的读取位置和保存位置
LAYER_DIR = "./layers/space"
OUTPUT_DIR = "./attention_map/space"

def load_attention_data(file_path: str):
    """加载 attention 数据"""
    data = np.load(file_path)
    return data["attention"]


def reconstruct_attention_matrix(attention_uint8):
    """
    从保存的数据反归一化 attention 矩阵

    Args:
        attention_uint8: 量化的 attention, shape [heads, asst_len, prompt_len]

    Returns:
        反归一化的 attention 矩阵, shape [heads, asst_len, prompt_len]
    """
    return attention_uint8.astype(np.float32) / 255.0


def apply_gamma_interpolation(attention, gamma=2.0):
    """
    使用幂函数插值（gamma校正）增强attention数值对比度

    Args:
        attention: attention矩阵, shape [heads, asst_len, prompt_len]
        gamma: 插值参数，gamma > 1 时高值更突出，gamma < 1 时低值更明显

    Returns:
        增强后的 attention 矩阵
    """
    # 使用幂函数插值: x^gamma
    # gamma > 1: 增强高值，压缩低值
    # gamma < 1: 增强低值，压缩高值
    enhanced = np.power(attention, gamma)
    return enhanced


def apply_spatial_interpolation(attention, scale_factor=2.0):
    """
    使用空间插值放大attention矩阵，让高亮线条更粗

    Args:
        attention: attention矩阵, shape [heads, asst_len, prompt_len]
        scale_factor: 缩放因子，>1 表示放大，高亮线条会更粗

    Returns:
        空间插值后的 attention 矩阵
    """
    if scale_factor <= 1.0:
        return attention

    heads, asst_len, prompt_len = attention.shape
    zoomed = np.zeros((heads, int(asst_len * scale_factor), int(prompt_len * scale_factor)), dtype=attention.dtype)


    for h in range(heads):
        # 使用双三次插值（order=3）进行平滑放大
        zoomed[h] = zoom(attention[h], scale_factor, order=3, mode='nearest', prefilter=True)

    return zoomed


def visualize_layer(layer_idx, layer_dir, output_path=None, gamma=2.0, scale=2.0):
    """
    可视化指定层的 attention
    
    Args:
        layer_idx: 层索引
        layer_dir: 层数据目录
        output_path: 保存路径（可选）
        gamma: 插值参数，gamma > 1 时高值更突出，gamma < 1 时低值更明显
        scale: 空间缩放因子，>1 时高亮线条更粗
    """
    file_path = Path(layer_dir) / f"layer_{layer_idx}.npz"
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return
    
    print(f"正在加载层 {layer_idx} 的数据...")
    attention_uint8 = load_attention_data(file_path)
    
    print(f"正在反归一化 attention 矩阵...")
    attention = reconstruct_attention_matrix(attention_uint8)

    heads, asst_len, prompt_len = attention.shape
    print(f"Attention 形状: [{heads}, {asst_len}, {prompt_len}]")
    
    # 对每个 head 用最大值归一化
    print(f"正在归一化 attention（每个 head 除以最大值）...")
    for h in range(heads):
        max_val = attention[h].max()
        if max_val > 0:
            attention[h] = attention[h] / max_val
    
    # 应用插值算法增强对比度
    if gamma != 1.0:
        print(f"正在应用 gamma 插值（gamma={gamma}）增强对比度...")
        attention = apply_gamma_interpolation(attention, gamma=gamma)
    
    # 应用空间插值放大，让高亮线条更粗
    if scale > 1.0:
        print(f"正在应用空间插值（scale={scale}）让高亮线条更粗...")
        attention = apply_spatial_interpolation(attention, scale_factor=scale)
    
    cols = min(4, heads)
    rows = (heads + cols - 1) // cols
    
    print(f"正在绘制图表（{heads} 个 heads，{rows}x{cols} 布局）...")
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    
    if heads == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    
    for h in range(heads):
        row = h // cols
        col = h % cols
        ax = axes[row][col]
        
        im = ax.imshow(attention[h], cmap='viridis', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'Head {h}')
        ax.set_xlabel('Prompt Position (Key)')
        ax.set_ylabel('Assistant Position (Query)')
        plt.colorbar(im, ax=ax, label='Attention')
    
    # 隐藏多余的子图
    for h in range(heads, rows * cols):
        row = h // cols
        col = h % cols
        axes[row][col].axis('off')
    
    gamma_text = f" (gamma={gamma})" if gamma != 1.0 else ""
    scale_text = f" (scale={scale})" if scale > 1.0 else ""
    plt.suptitle(f'Layer {layer_idx} Attention Patterns (Normalized{gamma_text}{scale_text})', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已保存至: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_all_layers(layer_dir, output_dir=None, gamma=2.0, scale=2.0):
    """
    可视化所有层的 attention
    
    Args:
        layer_dir: 层数据目录
        output_dir: 输出目录（可选）
        gamma: 插值参数，gamma > 1 时高值更突出，gamma < 1 时低值更明显
        scale: 空间缩放因子，>1 时高亮线条更粗
    """
    layer_path = Path(layer_dir)
    if not layer_path.exists():
        print(f"层数据目录不存在: {layer_dir}")
        return
    
    # 查找所有层数据文件
    layer_files = sorted(layer_path.glob("layer_*.npz"))
    layer_indices = [int(f.stem.split('_')[1]) for f in layer_files]
    
    if not layer_indices:
        print("没有找到层数据文件")
        return
    
    print(f"找到 {len(layer_indices)} 个层数据文件: {layer_indices}")
    
    # 创建输出目录
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 逐层可视化
    for layer_idx in tqdm(layer_indices, desc="处理所有层"):
        if output_dir:
            output_file = output_path / f"layer_{layer_idx}_attention.png"
            visualize_layer(layer_idx, layer_dir, str(output_file), gamma=gamma, scale=scale)
        else:
            visualize_layer(layer_idx, layer_dir, gamma=gamma, scale=scale)
    
    print(f"\n完成！共处理 {len(layer_indices)} 层")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Attention 可视化工具')
    parser.add_argument('--layer', type=int, default=None,
                        help='要可视化的层索引（与 --all 互斥）')
    parser.add_argument('--all', action='store_true',
                        help='可视化所有层')
    parser.add_argument('--layer-dir', type=str, default=LAYER_DIR,
                        help='层数据目录（默认：layers）')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='输出图片路径（单层）或输出目录（所有层）')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='插值参数（默认：1.0）。gamma > 1 时高值更突出，gamma < 1 时低值更明显，gamma=1.0 时不应用插值')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='空间缩放因子（默认：1.0）。scale > 1 时高亮线条更粗，scale=1.0 时不应用空间插值')
    
    args = parser.parse_args()
    
    # 参数校验
    if args.layer is None and not args.all:
        parser.error("必须指定 --layer 或 --all")
    if args.layer is not None and args.all:
        parser.error("--layer 和 --all 不能同时使用")
    
    if args.all:
        visualize_all_layers(args.layer_dir, args.output, gamma=args.gamma, scale=args.scale)
    else:
        visualize_layer(args.layer, args.layer_dir, args.output, gamma=args.gamma, scale=args.scale)


if __name__ == "__main__":
    main()
