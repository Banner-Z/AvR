from utils import parse_floats, get_json_data, find_failed_data, get_hf_datasets, extract_final_answer
from actions import chat_cot_model
import argparse
import os
import json
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

MODEL_NAME=""
TEMPERATURES=[]

def generate(d):
    result = []
    messages = d["chosen"][:]
    for temperature_1 in TEMPERATURES:
        record = {"messages": messages[:]}
        retry = 0
        final_result = None
        while retry < 2:
            retry += 1
            response = chat_cot_model([messages[0]], MODEL_NAME, temperature=temperature_1)
            if response == None:
                continue
            final_result = extract_final_answer(response)
            if final_result != None:
                break
        if final_result == None:
            print(f"format error!\n\n{response}")
            continue
        
        record["response"] = response
        record["final_result"] = final_result
        result.append(record)
    return result

def individual_data_inference(args):
    if args.input_file != None:
        data = get_json_data(args.input_file)
    elif args.hf_datasets != None:
        data = get_hf_datasets(args.hf_datasets)
    else:
        raise Exception("Datasets cannot be None!")
    max_points = min(len(data) - args.start_point, args.sample_nums)
    data = data[args.start_point: args.start_point + max_points]
    if os.path.exists(args.output_path):
        data = find_failed_data(data, args.output_path)

    # results = []
    with open(args.output_path, 'a+', encoding='utf-8') as o:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_data = {executor.submit(generate, d): d for d in data}

            for future in tqdm(as_completed(future_to_data), total=len(data)):
                try:
                    result = future.result()
                    if result == None:
                        continue
                    if isinstance(result, dict):
                        # results.append(result)
                        o.write(json.dumps(result, ensure_ascii=False) + '\n')
                        o.flush()
                        print("saved")
                    elif isinstance(result, list):
                        # results.extend(result)
                        for r in result:
                            if r == None:
                                continue
                            o.write(json.dumps(r, ensure_ascii=False) + '\n')
                            o.flush()
                except Exception as e:
                    print(f"Error processing data: {e}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', default=None, type=str, help="source data")
    parser.add_argument('--hf-datasets', default=None, type=str, help="source data")
    parser.add_argument('--model-name', required=True, type=str)
    parser.add_argument('--output-path', required=True, type=str)
    parser.add_argument('--max-workers', type=int, default=64)
    parser.add_argument('--start-point', type=int, default=0)
    parser.add_argument('--sample-nums', type=int, default=100000000)
    parser.add_argument('--temperatures', type=parse_floats, default=[0.7])

    args = parser.parse_args()
    MODEL_NAME = args.model_name
    TEMPERATURES = args.temperatures

    individual_data_inference(args)
