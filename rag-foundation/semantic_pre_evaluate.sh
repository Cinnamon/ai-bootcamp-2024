python -m scripts.main \
   --data_path qasper-test-v0.3.json \
   --output_path semantic_predictions.jsonl \
   --mode semantic \
   --force_index False \
   --top_k 5

python evaluate.py --predictions semantic_predictions.jsonl --gold ./qasper-test-v0.3.json