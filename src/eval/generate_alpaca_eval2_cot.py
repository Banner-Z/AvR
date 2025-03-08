import datasets
from openai import OpenAI
import random
import json
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import re

def extract_final_answer(text):
    match = re.search(r'Final answer:\n\n(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def chat_model(message, model_name, ports, temperature=0.7):
    openai_api_key = "EMPTY"
    openai_api_base = "http://0.0.0.0:8055/v1"
    port = random.choice(ports)
    openai_api_base = openai_api_base.replace("8055", port)
    
    MODEl_NAME=model_name

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    try:
        chat_response = client.chat.completions.create(
            model=MODEl_NAME,
            messages=message,
            temperature=temperature,
            top_p=0.8,
            # max_completion_tokens=2048
        )
        response = json.loads(chat_response.model_dump_json())["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        return None
    return response

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]

model = "/model/path/"

# vllm ports
ports = ["11101", "11102", "11103", "11104", "11105", "11106", "11107", "11108"]

output_dir = "/output/dir/"
output_file = output_dir + "file-name.json"
os.makedirs(output_dir, exist_ok=True)

def generate(example):
    retry = 0
    final_result = None
    while retry < 5:
        retry += 1
        example["origin_output"] = chat_model([{"role": "user", "content": example["instruction"]}], model, ports)
        example["origin_model"] = model
        final_result = extract_final_answer(example["origin_output"])
        if final_result != None:
            break
    if final_result == None:
        print(f"format error!\n\n{example['origin_output']}")
        return None
    
    example["final_result"] = final_result
    return example

results = []
with open(output_file, 'w', encoding='utf-8') as o:
    with ThreadPoolExecutor(max_workers=64) as executor:
        future_to_data = {executor.submit(generate, d): d for d in eval_set}

        for future in tqdm(as_completed(future_to_data), total=len(eval_set)):
            # try:
            result = future.result()
            if result != None:
                results.append(result)
                o.write(json.dumps(result, ensure_ascii=False) + '\n')
                o.flush()
            # except Exception as e:
            #     print(f"Error processing data: {e}")
