# commands for expriments
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run main.py \
--cfg_name scifact \
--llm_model Qwen/Qwen3-8B \
--llm_base_url http://localhost:8000/v1 \
--llm_max_gen_len 40960 \
--attack rtf \
--batch_size 10