import json
import random
from tqdm import tqdm

def deduplicate_data(data):
    unique_lines = set()
    result = []

    for d in tqdm(data):
        line = json.dumps(d, ensure_ascii=False)
        if line and line not in unique_lines:
            unique_lines.add(line)
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Invalid JSON line: {line}")

    return result

def get_json_data(path):
    data = []
    with open(path, 'r') as f:
        for l in f.readlines():
            j = json.loads(l)
            data.append(j)
    print(f"load {str(len(data))} lines from {path}.")
    return data

import re
from collections import Counter

def prepare(data):
    result = []
    rejected_lenths = []
    chosen_lenths = []

    count = 0
    for d in data:
        message = d["messages"][:]
        if d["worst_record"]["score"] >= d["best_record"]["score"] or d["best_record"]["score"] <= d["source_score"]:
            print("error~")
            continue
        if len(d["best_record"]["final_result"]) > len(d["worst_record"]["final_result"]):
            continue
        result.append({
            "messages": d["messages"][:-1],
            "chosen": {"role": "assistant", "content": d["best_record"]["response"]},
            "rejected": {"role": "assistant", "content": d["worst_record"]["response"]}
        })
        chosen_lenths.append(len(d["best_record"]["response"]))
        rejected_lenths.append(len(d["worst_record"]["response"]))
        count += 1
    print(count)

    avg_rejected = sum(rejected_lenths) / len(rejected_lenths)
    avg_chosen = sum(chosen_lenths) / len(chosen_lenths)
    print(f"avg_rejected: {avg_rejected}, avg_chosen: {avg_chosen}")
    return result

def write_jsonl(data, output_file):
    with open(output_file, 'w') as o:
        for l in data:
            o.write(json.dumps(l, ensure_ascii=False) + '\n')

input_file = "data/input-path.jsonl"

output_file = "data/prepared-training-data.jsonl"

data = get_json_data(input_file)

print(len(data))
data = prepare(data)
print(len(data))
data = deduplicate_data(data)
print(len(data))
random.shuffle(data)
# write_jsonl(data, output_file)
json.dump(data, open(output_file, 'w'))