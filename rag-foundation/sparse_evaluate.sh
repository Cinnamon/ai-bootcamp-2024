python -m scripts.main \
   --data_path qasper-test-v0.3.json \
   --output_path sparse_predictions.jsonl \
   --mode sparse \
   --force_index True \
   --retrieval_only True \
   --top_k 5

python evaluate.py --predictions sparse_predictions.jsonl --gold ./qasper-test-v0.3.json --retrieval_only