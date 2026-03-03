from configs.config_base import *

fiqa = VRConfig(
    {
        "data": {
            "force_rebuild": False,
            "datastorage_tool": "chroma",
            "data_dir_list": ["./data/fiqa/corpus.jsonl"],
            "description": {
                "name": "fiqa",
                "type": "Finance",  # 映射英文类别
                "intro": "a financial sentiment analysis benchmark derived from real-world sources such as StockTwits posts and financial news headlines.it enables models to understand market sentiment and investor opinions in financial contexts."
            }
        },
        "tool_llm": {
            "model": "gpt-4.1-mini",
            "base_url": "",
            "api_key": "",
            "reasoning": True,
            "temperature": 0.7,
            "top_p": 0.8
        },
        "retrieval": {
            "method": "mmr",
            "top_k": 10,
            "fetch_k": 40,
            "score_threshold": 0.75,
            "top_n": 5,
            "embed": {
                "provider": "hf",
                "model_name": "bge-large-en-v1.5",
                "model_dir": "",
                "retrival_database_batch_size": 256
            }
        },
        "reranker": {
            "provider": "hf",
            "model": "",
            "api_key": None
        },
        "extractor": {
            "provider": "hf",
            "model": "",
            "api_key": None
        }
    }
)
