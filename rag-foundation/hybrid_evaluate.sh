TOKENIZERS_PARALLELISM=false python -m scripts.main \
   --data_path qasper-test-v0.3.json \
   --output_path hybrid_predictions.jsonl \
   --mode hybrid \
   --force_index True \
   --retrieval_only True \
   --top_k 5

python evaluate.py --predictions hybrid_predictions.jsonl --gold ./qasper-test-v0.3.json --retrieval_only
