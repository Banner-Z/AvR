
python src/cot_dpo_data_scoring.py \
--input-file data/generated-data-path.jsonl \
--filtered-data-path data/output-path.jsonl \
--scoring-detail-path data/output-path-scoring_detail.jsonl \
--model-name path/to/reward/model \
--max-workers 80 \