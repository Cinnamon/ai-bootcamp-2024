python -m scripts.main \
   --data_path qasper-test-v0.3.json \
   --output_path semantic_predictions.jsonl \
   --mode semantic \
   --force_index True \
   --retrieval_only True \
   --top_k 5

python evaluate.py --predictions semantic_predictions.jsonl --gold ./qasper-test-v0.3.json --retrieval_only
