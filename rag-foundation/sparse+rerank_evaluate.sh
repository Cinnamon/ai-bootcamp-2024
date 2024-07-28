TOKENIZERS_PARALLELISM=false python -m scripts.main \
   --data_path qasper-test-v0.3.json \
   --output_path sparse+rerank_predictions.jsonl \
   --mode sparse \
   --force_index True \
   --retrieval_only True \
   --re_rank True \
   --top_k 10

python evaluate.py --predictions sparse+rerank_predictions.jsonl --gold ./qasper-test-v0.3.json --retrieval_only
