"""
注意力熵计算和可视化脚本
读取保存的 attention 数据，计算并可视化注意力熵
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 默认的读取位置和保存位置
LAYER_DIR = "./layers/safe"
OUTPUT_DIR = "./entropy/safe"


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


def compute_attention_entropy(attention):
    """
    计算注意力熵
    仅统计 'assistant' 对 'prompt' 的注意力权重熵值

    Args:
        attention: attention矩阵, shape [heads, asst_len, prompt_len]
                  其中 asst_len 是 query 位置（assistant），prompt_len 是 key 位置（prompt）

    Returns:
        熵值数组, shape [heads, asst_len]
        对于每个 head 和每个 assistant 位置，计算该位置对 prompt 的注意力分布熵
    """
    heads, asst_len, prompt_len = attention.shape

    # 初始化熵值数组
    entropy = np.zeros((heads, asst_len))

    for h in range(heads):
        for a in range(asst_len):
            # 获取第 h 个 head 中，第 a 个 assistant 位置对所有 prompt 位置的注意力
            attn_weights = attention[h, a, :]

            # 计算熵: H = -Σ p_i * log2(p_i)
            # 添加小常数避免 log(0)
            epsilon = 1e-10
            attn_weights = attn_weights + epsilon
            attn_weights = attn_weights / np.sum(attn_weights)  # 确保归一化

            h_val = -np.sum(attn_weights * np.log2(attn_weights))
            entropy[h, a] = h_val

    return entropy


def normalize_entropy_by_max(entropy, prompt_len):
    """
    使用最大熵归一化熵值
    最大熵 H_max = log2(prompt_len)，当所有注意力权重均匀分布时达到
    
    Args:
        entropy: 熵值数组, shape [heads, asst_len]
        prompt_len: prompt 长度，用于计算最大熵
        
    Returns:
        归一化后的熵值数组, shape [heads, asst_len]
    """
    max_entropy = np.log2(prompt_len)
    normalized_entropy = entropy / max_entropy
    return normalized_entropy


def visualize_layer_entropy(layer_idx, layer_dir, output_path=None):
    """
    可视化指定层的注意力熵
    
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
    
    # 计算注意力熵
    print(f"正在计算注意力熵...")
    entropy = compute_attention_entropy(attention)
    
    # 使用最大熵归一化
    print(f"正在归一化熵值（最大熵归一化）...")
    normalized_entropy = normalize_entropy_by_max(entropy, prompt_len)
    
    # 为每个 head 绘制熵值分布折线图
    print(f"正在绘制图表（{heads} 个 heads）...")
    fig, axes = plt.subplots(heads, 1, figsize=(10, 3 * heads))
    
    if heads == 1:
        axes = [axes]
    
    for h in range(heads):
        ax = axes[h]
        x_positions = np.arange(asst_len)
        
        ax.plot(x_positions, normalized_entropy[h], linewidth=1.5, color='steelblue')
        ax.set_title(f'Head {h} - Attention Entropy Distribution', fontsize=10)
        ax.set_xlabel('Assistant Position (Query)', fontsize=9)
        ax.set_ylabel('Normalized Entropy', fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加平均熵值的水平线
        avg_entropy = np.mean(normalized_entropy[h])
        ax.axhline(y=avg_entropy, color='red', linestyle='--', linewidth=1, alpha=0.7,
                   label=f'Mean: {avg_entropy:.3f}')
        ax.legend(fontsize=8)
    
    plt.suptitle(f'Layer {layer_idx} - Attention Entropy (Normalized by Max Entropy)', 
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


def visualize_all_layers_entropy(layer_dir, output_dir=None):
    """
    可视化所有层的注意力熵
    
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
            output_file = output_path / f"layer_{layer_idx}_entropy.png"
            visualize_layer_entropy(layer_idx, layer_dir, str(output_file))
        else:
            visualize_layer_entropy(layer_idx, layer_dir)
    
    print(f"\n完成！共处理 {len(layer_indices)} 层")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='注意力熵计算和可视化工具')
    parser.add_argument('--layer', type=int, default=None,
                        help='要可视化的层索引（与 --all 互斥）')
    parser.add_argument('--all', action='store_true',
                        help='可视化所有层')
    parser.add_argument('--layer-dir', type=str, default=LAYER_DIR,
                        help='层数据目录（默认：layers/space）')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='输出图片路径（单层）或输出目录（所有层）')
    
    args = parser.parse_args()
    
    # 参数校验
    if args.layer is None and not args.all:
        parser.error("必须指定 --layer 或 --all")
    if args.layer is not None and args.all:
        parser.error("--layer 和 --all 不能同时使用")
    
    if args.all:
        visualize_all_layers_entropy(args.layer_dir, args.output)
    else:
        visualize_layer_entropy(args.layer, args.layer_dir, args.output)


if __name__ == "__main__":
    main()
