TOKENIZERS_PARALLELISM=false python -m scripts.main \
   --data_path ms_marco.json \
   --output_path sparse_predictions.jsonl \
   --mode sparse \
   --force_index True \
   --retrieval_only True \
   --top_k 10

python evaluate.py --predictions sparse_predictions.jsonl --gold ./ms_marco.json --retrieval_only
