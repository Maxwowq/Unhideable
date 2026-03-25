"""
注意力最大聚焦点散点图可视化脚本
读取保存的 attention 数据，计算并可视化每个 assistant token 在 prompt 侧的最大注意力聚焦点
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 默认的读取位置和保存位置
LAYER_DIR = "./layers/safe"
OUTPUT_DIR = "./max/safe"


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


def compute_max_attention_indices(attention):
    """
    计算每个 assistant token 在 prompt 侧的最大注意力权重索引

    Args:
        attention: attention矩阵, shape [heads, asst_len, prompt_len]
                  其中 asst_len 是 query 位置（assistant），prompt_len 是 key 位置（prompt）

    Returns:
        最大注意力索引数组, shape [heads, asst_len]
        对于每个 head 和每个 assistant 位置，返回该位置在 prompt 侧最大注意力的索引
    """
    # 使用向量化操作一次性计算所有最大注意力索引
    # axis=2 表示在 prompt_len 维度上找最大值
    max_indices = np.argmax(attention, axis=2)

    return max_indices


def visualize_layer_max_attention(layer_idx, layer_dir, output_path=None):
    """
    可视化指定层的注意力最大聚焦点
    
    Args:
        layer_idx: 层索引
        layer_dir: 层数据目录
        output_path: 保存路径（可选）
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
    
    # 计算最大注意力索引
    print(f"正在计算最大注意力索引...")
    max_indices = compute_max_attention_indices(attention)
    
    # 为每个 head 绘制散点图
    print(f"正在绘制图表（{heads} 个 heads）...")
    fig, axes = plt.subplots(heads, 1, figsize=(12, 3 * heads))
    
    if heads == 1:
        axes = [axes]
    
    for h in range(heads):
        ax = axes[h]
        x_positions = np.arange(asst_len)  # Assistant token 索引
        y_positions = max_indices[h]        # 对应的最大注意力 Prompt token 索引
        
        # 绘制散点图
        ax.scatter(x_positions, y_positions, c='steelblue', s=2, alpha=0.6, linewidth=0.5)
        ax.set_title(f'Head {h} - Maximum Attention Focus Points', fontsize=10)
        ax.set_xlabel('Assistant Position (Query)', fontsize=9)
        ax.set_ylabel('Prompt Position (Key)', fontsize=9)
        ax.set_xlim([-0.5, asst_len - 0.5])
        ax.set_ylim([-0.5, prompt_len - 0.5])
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'Layer {layer_idx} - Maximum Attention Focus Points', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        # 创建输出目录（如果不存在）
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已保存至: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_all_layers_max_attention(layer_dir, output_dir=None):
    """
    可视化所有层的注意力最大聚焦点
    
    Args:
        layer_dir: 层数据目录
        output_dir: 输出目录（可选）
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
            output_file = output_path / f"layer_{layer_idx}_max_attention.png"
            visualize_layer_max_attention(layer_idx, layer_dir, str(output_file))
        else:
            visualize_layer_max_attention(layer_idx, layer_dir)
    
    print(f"\n完成！共处理 {len(layer_indices)} 层")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='注意力最大聚焦点散点图可视化工具')
    parser.add_argument('--layer', type=int, default=None,
                        help='要可视化的层索引（与 --all 互斥）')
    parser.add_argument('--all', action='store_true',
                        help='可视化所有层')
    parser.add_argument('--layer-dir', type=str, default=LAYER_DIR,
                        help='层数据目录（默认：layers/safe）')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='输出图片路径（单层）或输出目录（所有层）')
    
    args = parser.parse_args()
    
    # 参数校验
    if args.layer is None and not args.all:
        parser.error("必须指定 --layer 或 --all")
    if args.layer is not None and args.all:
        parser.error("--layer 和 --all 不能同时使用")
    
    if args.all:
        visualize_all_layers_max_attention(args.layer_dir, args.output)
    else:
        visualize_layer_max_attention(args.layer, args.layer_dir, args.output)


if __name__ == "__main__":
    main()
