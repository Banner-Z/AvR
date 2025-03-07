
python src/cot_data_generation_greedy.py \
--hf-datasets princeton-nlp/llama3-ultrafeedback-armorm \
--output-path data/output-path.jsonl \
--model-name path/to/stage1-model \
--rm-name path/to/reward/model \
--start-point 0 \
--sample-nums 10000 \
--max-workers 80 \
--template-version v2
