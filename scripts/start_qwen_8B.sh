CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run vllm serve Qwen/Qwen3-8B \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.95 \
    --port 8000