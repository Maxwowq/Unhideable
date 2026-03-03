# LeakDojo

## 参考指令
1. 运行主流程
    ```
    nohup python main.py \
    --device cuda:1 \
    --cfg_name fiqa \
    --attack por \
    --attack_num 200 \
    --llm_model gpt-5.1 \
    --llm_base_url http://xxxxxxxx \
    --llm_api_key EMPTY \
    --attack_num 200 \
    --reranker \
    > por_scifact_010_200_gpt-5-1.log 2>&1 & 
    ```

### 说明

1. 实验设置保存在`configs/`文件路径下：
    - data: data_dir_list, description
    - retrieval: method(top_k/fetch_k/score_threshold/top_n), embed.provider/model_dir
    - reranker.model 设置重排模型