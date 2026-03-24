"""
预实验脚本
基于一次实验的prompt，选取高价值的layer和head
"""

import os
from pathlib import Path
import json
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# 配置参数
SAVE_DIR = "layers/space"
MODEL_ID = "Qwen/Qwen3-8B"
CUDA_DEVICE = "0"
RESULT_PATH = "./cipher_attack/results/gpt_space.json"
ASSISTANT_TOKEN = "<|im_start|>assistant\n"


def save_topk_attn_hook(layer_idx: int, save_dir: Path, asst_start_idx: int):
    """
    创建 attention hook 函数

    Args:
        layer_idx: 层索引
        save_dir: 保存目录
        asst_start_idx: Assistant 起始索引
    """
    def hook(module: torch.nn.Module, input: torch.Tensor, output: Tuple[torch.Tensor, ...]) -> None:
        # output[1] 形状: [1, heads, seq_len, seq_len]
        with torch.no_grad():
            # 1. 提取权重：Assistant 作为 Query，Prompt 作为 Key
            # 形状: [heads, asst_len, prompt_len]
            attn = output[1][0, :, asst_start_idx:, :asst_start_idx]
            
            # 2. 量化为 INT8 (0-255) 并转到 CPU
            attn_uint8 = (attn * 255).to(torch.uint8).cpu().numpy()
            
            # 3. 保存压缩文件
            save_path = save_dir / f"layer_{layer_idx}.npz"
            np.savez_compressed(save_path, attention=attn_uint8)
            print(f"层 {layer_idx} 处理完成并保存，形状: {attn_uint8.shape}")

            del attn # 显式删除引用
    
    return hook


def load_experiment_data(file_path: str) -> dict:
    """
    加载实验数据
    
    Args:
        file_path: 结果文件路径
        
    Returns:
        包含 prompt 和 answer 的字典
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"结果文件不存在: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            raise ValueError("结果文件为空")
        
        first = json.loads(lines[0].strip())
        
        if "prompt" not in first or "answer" not in first:
            raise ValueError("数据格式不正确，缺少 'prompt' 或 'answer' 字段")
        
        return {
            "prompt": first["prompt"],
            "answer": first["answer"]
        }


def locate_assistant_start(
    full_id_list: List[int], 
    asst_token_ids: List[int],
    model_name: str
) -> int:
    """
    定位 Assistant 区域的起始位置
    
    Args:
        full_id_list: 完整的 token ID 列表
        asst_token_ids: Assistant 标记的 token ID 列表
        model_name: 模型名称（用于错误提示）
        
    Returns:
        Assistant 起始索引
    """
    # 从后往前找，确保定位到的是最后一个对话块（即真正的 Assistant 回复区）
    for i in range(len(full_id_list) - len(asst_token_ids), -1, -1):
        if full_id_list[i : i + len(asst_token_ids)] == asst_token_ids:
            return i + len(asst_token_ids)
    
    raise ValueError(
        f"未能自动定位 Assistant 边界，请检查 Tokenizer '{model_name}' "
        f"的 Chat Template 是否包含 '{ASSISTANT_TOKEN}'"
    )


def main():
    """主函数"""
    # 1. 设置 CUDA 环境
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE
    
    # 2. 创建保存目录
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 加载模型
    print(f"正在加载模型: {MODEL_ID}")
    # 配置 4-bit 量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, # 保持计算精度
        bnb_4bit_quant_type="nf4",             # 使用更精确的规范化浮点
        bnb_4bit_use_double_quant=True         # 二次量化以额外节省约 0.4 bit/param
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",                     # 自动分配到多张显卡或 CPU
        attn_implementation="eager"            # 必须保持 eager 模式以获取 attention
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("已设置 attention 实现为 'eager' 模式")
    print("模型加载完成")
    
    # 4. 加载实验数据
    print(f"正在读取结果文件: {RESULT_PATH}")
    data = load_experiment_data(RESULT_PATH)
    
    # 5. 构造对话并应用模板
    messages = [
        {"role": "user", "content": data["prompt"]},
        {"role": "assistant", "content": data["answer"]}
    ]
    
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(full_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    
    # 6. 定位 Assistant 边界
    full_id_list = input_ids[0].cpu().tolist()
    asst_token_ids = tokenizer.encode(ASSISTANT_TOKEN, add_special_tokens=False)
    
    asst_start_idx = locate_assistant_start(full_id_list, asst_token_ids, MODEL_ID)
    print(f"定位成功！Assistant 起始索引: {asst_start_idx}\n总序列长度: {seq_len}")

    # 7. 将输入移至设备
    inputs = inputs.to(model.device)

    # 8. 注册 hooks（每隔4层保存一层：0, 4, 8, ...）
    hooks = []
    for i in range(0, len(model.model.layers), 4):
        h = model.model.layers[i].self_attn.register_forward_hook(
            save_topk_attn_hook(i, save_dir, asst_start_idx)
        )
        hooks.append(h)

    # 8. 执行前向传播
    print("开始前向传播 (Prefill)... 这可能需要 1-2 分钟")
    try:
        with torch.no_grad():
            model(**inputs, output_attentions=False)
    except Exception as e:
        print(f"前向传播失败: {e}")
        raise
    finally:
        # 9. 清理 hooks
        for h in hooks:
            h.remove()

    print(f"\n实验完成！每隔4层的 Attention 数据已保存至: {SAVE_DIR}")


if __name__ == "__main__":
    main()
