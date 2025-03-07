from utils import parse_floats, get_json_data, find_failed_data, get_hf_datasets
from actions import individual_judge, individual_refine, generate_response
import argparse
import os
import json
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_NAME=""
TEMPERATURES=[]
TEMPLATE_VERSION=""

def generate(d):
    messages = d["chosen"][:]
    result = []
    if TEMPLATE_VERSION == "v2":
        response = generate_response(messages[:-1], MODEL_NAME, temperature=0.7)
        if response == None:
            print(f"Failed generate response!")
            return None
        else:
            messages[-1] = {"role": "assistant", "content": response}
    for temperature_1 in TEMPERATURES:
        score, judgement = individual_judge(messages, MODEL_NAME, temperature_1, TEMPLATE_VERSION)
        if score == -1:
            if "source" in d:
                print(f"Failed source: {d['source']}")
            continue
        for temperature_2 in TEMPERATURES:
            refined_response = individual_refine(messages, judgement, MODEL_NAME, temperature_2, TEMPLATE_VERSION)
            if refined_response == None:
                continue
            result.append({"messages": messages, "judgement": judgement, "refined_response": refined_response, "score": score, "temperature_judge": temperature_1, "temperature_refine": temperature_2})
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
    parser.add_argument('--start-point', type=int, default=0, help="start point of the dataset")
    parser.add_argument('--sample-nums', type=int, default=100000000, help="the number of data items used in the dataset")
    parser.add_argument('--temperatures', type=parse_floats, default=[0.7])
    parser.add_argument('--template-version', type=str, default="v2", help="v1 and v2 are templates for single round and multiple rounds respectively. The main experiment uses the template for multiple rounds (v2) because it is more efficient in training.")

    args = parser.parse_args()
    MODEL_NAME = args.model_name
    TEMPERATURES = args.temperatures
    TEMPLATE_VERSION = args.template_version

    individual_data_inference(args)
