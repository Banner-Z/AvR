
python src/stage1_data_generation.py \
--hf-datasets princeton-nlp/llama3-ultrafeedback-armorm \
--output-path data/output-path.jsonl \
--model-name path/to/model \
--temperatures 0.7,0.7 \
--max-workers 32 \
--template-version v2 
# v1 and v2 are templates for single round and multiple rounds respectively. The main experiment uses the template for multiple rounds (v2) because it is more efficient in training.