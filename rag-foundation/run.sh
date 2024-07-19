python -m scripts.main \
   --data_path qasper-test-v0.3.json \
   --output_path predictions.jsonl \
   --mode sparse \
   --force_index  \
   --retrieval_only True \
   --top_k 1