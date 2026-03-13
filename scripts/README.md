### 使用方法

1. 首先使用`cipher_attack/`文件夹下的`prompts.py`生成完整的对话，结果保存在`cipher_attack/results/`
2. 然后运行`pre.py`，进行prefill并保存attetion到`layers/`文件夹
3. 最后使用`visualize_attention.py`将保存的注意力可视化