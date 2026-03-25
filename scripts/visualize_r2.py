"""
注意力最大聚焦点 R² 计算和可视化脚本
读取保存的 attention 数据，计算最大注意力聚焦点，然后计算每个 head 的 R²，并绘制折线图
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

# 默认的读取位置和保存位置
LAYER_DIR = "./layers/verbatim"
OUTPUT_DIR = "./r2/verbatim"


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


def compute_r2_for_head(max_indices):
    """
    计算单个 head 的最大注意力索引序列的 R²（决定系数）
    使用线性回归计算与线性趋势的拟合程度

    Args:
        max_indices: 最大注意力索引数组, shape [asst_len]

    Returns:
        R² 值
    """
    x = np.arange(len(max_indices))
    y = max_indices

    # 计算线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # R² 是相关系数的平方
    r_squared = r_value ** 2

    return r_squared


def compute_layer_r2(attention):
    """
    计算指定层所有 head 的 R² 值

    Args:
        attention: attention矩阵, shape [heads, asst_len, prompt_len]

    Returns:
        R² 值数组, shape [heads]
    """
    # 计算最大注意力索引
    max_indices = compute_max_attention_indices(attention)

    heads = max_indices.shape[0]
    r2_values = np.zeros(heads)

    # 为每个 head 计算 R²
    for h in range(heads):
        r2_values[h] = compute_r2_for_head(max_indices[h])

    return r2_values


def compute_all_layers_r2(layer_dir):
    """
    计算所有层所有 head 的 R² 值

    Args:
        layer_dir: 层数据目录

    Returns:
        字典: {layer_idx: r2_values_array}
        其中 r2_values_array 的 shape 为 [heads]
    """
    layer_path = Path(layer_dir)
    if not layer_path.exists():
        print(f"层数据目录不存在: {layer_dir}")
        return {}

    # 查找所有层数据文件
    layer_files = sorted(layer_path.glob("layer_*.npz"))
    layer_indices = [int(f.stem.split('_')[1]) for f in layer_files]

    if not layer_indices:
        print("没有找到层数据文件")
        return {}

    print(f"找到 {len(layer_indices)} 个层数据文件: {layer_indices}")

    # 存储所有层的 R² 值
    all_r2 = {}

    # 逐层计算 R²
    for layer_idx in tqdm(layer_indices, desc="计算所有层的 R²"):
        file_path = layer_path / f"layer_{layer_idx}.npz"

        # 加载数据
        attention_uint8 = load_attention_data(file_path)
        attention = reconstruct_attention_matrix(attention_uint8)

        # 计算该层所有 head 的 R²
        r2_values = compute_layer_r2(attention)
        all_r2[layer_idx] = r2_values

    return all_r2


def visualize_r2_line_chart(all_r2, output_path=None):
    """
    绘制 R² 折线图

    Args:
        all_r2: 字典 {layer_idx: r2_values_array}
        output_path: 保存路径（可选）
    """
    if not all_r2:
        print("没有 R² 数据可绘制")
        return

    # 获取层索引并排序
    layer_indices = sorted(all_r2.keys())

    # 确定所有层的 head 数量是否一致
    num_heads_list = [len(all_r2[idx]) for idx in layer_indices]
    if len(set(num_heads_list)) != 1:
        print(f"警告: 不同层的 head 数量不一致: {set(num_heads_list)}")

    max_heads = max(num_heads_list)

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))

    # 为每个 head 绘制折线
    colors = plt.cm.tab20(np.linspace(0, 1, max_heads))

    for h in range(max_heads):
        r2_values_per_layer = []
        valid_layers = []

        for layer_idx in layer_indices:
            r2_values = all_r2[layer_idx]
            if h < len(r2_values):
                r2_values_per_layer.append(r2_values[h])
                valid_layers.append(layer_idx)

        if r2_values_per_layer:
            ax.plot(valid_layers, r2_values_per_layer,
                   marker='o', linewidth=2, markersize=4,
                   color=colors[h], label=f'Head {h}')

    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² (Coefficient of Determination)', fontsize=12, fontweight='bold')
    ax.set_title('R² Values for Maximum Attention Focus Points Across Layers',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    # 添加图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

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


def print_r2_statistics(all_r2):
    """
    打印 R² 统计信息

    Args:
        all_r2: 字典 {layer_idx: r2_values_array}
    """
    if not all_r2:
        print("没有 R² 数据")
        return

    layer_indices = sorted(all_r2.keys())

    print("\n" + "="*80)
    print("R² 统计信息")
    print("="*80)

    # 为每个 head 计算统计信息
    num_heads = max(len(all_r2[idx]) for idx in layer_indices)

    for h in range(num_heads):
        r2_values_per_layer = []

        for layer_idx in layer_indices:
            r2_values = all_r2[layer_idx]
            if h < len(r2_values):
                r2_values_per_layer.append(r2_values[h])

        if r2_values_per_layer:
            mean_r2 = np.mean(r2_values_per_layer)
            std_r2 = np.std(r2_values_per_layer)
            min_r2 = np.min(r2_values_per_layer)
            max_r2 = np.max(r2_values_per_layer)

            print(f"\nHead {h}:")
            print(f"  平均 R²: {mean_r2:.4f} ± {std_r2:.4f}")
            print(f"  最小值: {min_r2:.4f}")
            print(f"  最大值: {max_r2:.4f}")

    print("\n" + "="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='注意力最大聚焦点 R² 计算和可视化工具')
    parser.add_argument('--layer-dir', type=str, default=LAYER_DIR,
                        help='层数据目录（默认：layers/safe）')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='输出图片路径（默认：max_r2/safe）')

    args = parser.parse_args()

    # 计算所有层的 R²
    print(f"正在从 {args.layer_dir} 读取数据并计算 R²...")
    all_r2 = compute_all_layers_r2(args.layer_dir)

    if not all_r2:
        print("未能计算 R² 值")
        return

    # 打印统计信息
    print_r2_statistics(all_r2)

    # 绘制折线图
    print(f"\n正在绘制 R² 折线图...")
    output_file = Path(args.output) / "r2_line_chart.png"
    visualize_r2_line_chart(all_r2, str(output_file))

    print(f"\n完成！")


if __name__ == "__main__":
    main()
