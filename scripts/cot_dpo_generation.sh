
python src/cot_dpo_data_generation.py \
--hf-datasets princeton-nlp/llama3-ultrafeedback-armorm \
--output-path data/path.jsonl \
--model-name sft/model/path \
--temperatures 0.7,0.7,0.7,0.7,0.7 \
--start-point 20000 \
--sample-nums 5000 \
--max-workers 80 \
