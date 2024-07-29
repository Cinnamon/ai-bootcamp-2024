python -m scripts.main \
   --data_path ms_marco.json \
   --output_path semantic_predictions.jsonl \
   --mode semantic \
   --force_index True \
   --retrieval_only True \
   --top_k 5

python evaluate.py --predictions semantic_predictions.jsonl --gold ./ms_marco.json --retrieval_only
