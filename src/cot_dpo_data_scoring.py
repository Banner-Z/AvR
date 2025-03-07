from utils import parse_floats, get_json_data, prepare_cot_sampling_data
from actions import rm_scoring
import argparse
import os
import json
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_NAME = ""
OUTPUT_STRUCTURE = ""

def scoring(d):
    messages = d["messages"]
    source_score = rm_scoring(messages, MODEL_NAME)
    detailed_result = {"messages": messages, "source_score": source_score, "refinements": []}
    best_score = source_score
    best_record = None
    worst_score = float("inf")
    worst_record = None
    for record in d["refinements"]:
        new_messages = messages[:]
        new_messages[-1] = {"role": "assistant", "content": record["final_result"]}
        score = rm_scoring(new_messages, MODEL_NAME)
        if score == None:
            continue
        record["score"] = score
        if score != None and score > best_score:
            best_score = score
            best_record = record
        if score < worst_score:
            worst_score = score
            worst_record = record
        detailed_result["refinements"].append(record)
    if source_score < best_score and best_score > worst_score:
        result = {"scoring_detail": detailed_result, "filtered_data": {"messages": messages, "source_score": source_score,"best_record": best_record, "best_score": best_score, "worst_record": worst_record, "worst_score": worst_score}}
    else:
        result = {"scoring_detail": detailed_result}
    return result
        
def individual_data_sampling(args):
    data = get_json_data(args.input_file)
    max_points = min(len(data), args.sample_nums)
    data = data[:max_points]
    data = prepare_cot_sampling_data(data)
    print(len(data))
    data = data[:]

    results = []
    with open(args.scoring_detail_path, 'a+', encoding='utf-8') as o1, open(args.filtered_data_path, 'a+') as o2:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_data = {executor.submit(scoring, d): d for d in data}

            for future in tqdm(as_completed(future_to_data), total=len(data)):
                try:
                    result = future.result()

                    if result == None:
                        continue
                    if "scoring_detail" in result:
                        o1.write(json.dumps(result["scoring_detail"], ensure_ascii=False) + '\n')
                    if "filtered_data" in result:
                        o2.write(json.dumps(result["filtered_data"], ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"Error processing data: {e}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, type=str, help="source data")
    parser.add_argument('--model-name', required=True, type=str)
    parser.add_argument('--scoring-detail-path', required=True, type=str)
    parser.add_argument('--filtered-data-path', required=True, type=str)
    parser.add_argument('--max-workers', type=int, default=64)
    parser.add_argument('--sample-nums', type=int, default=100000000)
    parser.add_argument('--output-structure', type=str, default="sft", help="sft or dpo")

    args = parser.parse_args()
    MODEL_NAME = args.model_name
    OUTPUT_STRUCTURE = args.output_structure

    individual_data_sampling(args)
