# qwen3-32B
uv run prompts.py --model qwen --attack safe
uv run prompts.py --model qwen --attack space
uv run prompts.py --model qwen --attack verbatim

# gemini-3-flash-preview
uv run prompts.py --model gemini --attack caesar

# gpt-5.2
uv run prompts.py --model gpt --attack space
uv run prompts.py --model gpt --attack caesar