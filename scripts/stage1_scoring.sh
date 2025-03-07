
python src/stage1_data_sampling.py \
--input-file data/generated-data-path.jsonl \
--filtered-data-path data/output-path.jsonl \
--scoring-detail-path data/output-path-scoring_detail.jsonl \
--model-name path/to/reward/model \
--max-workers 80 \
--output-structure dpo