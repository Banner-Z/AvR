import json
import re
import random
import argparse
from collections import defaultdict

def parse_floats(input_string):
    try:
        return [float(item) for item in input_string.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("format error !")
    
def get_json_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            try:
                l = l.strip()
                j = json.loads(l)
                data.append(j)
            except Exception as e:
                continue
    print(f"load {str(len(data))} lines from {path}.")
    return data

def get_hf_datasets(path):
    from datasets import load_dataset
    ds = load_dataset(path)['train']
    data = []
    for d in ds:
        data.append(d)
    return data

def message_to_string(message, exchange=False):
    result = ""
    for m in message:
        if (m["role"] == "user" and not exchange) or (m["role"] == "assistant" and exchange):
            result += "### User:\n" + m["content"] + "\n"
        elif (m["role"] == "assistant" and not exchange) or (m["role"] == "user" and exchange):
            result += "\n### Assistant:\n" + m["content"] + "\n"
    return result

def extract_score(text):
    pattern = r'\[\[(\d+)\]\]'
    match = re.search(pattern, text)
    
    if match:
        score = int(match.group(1))
        return score
    else:
        return None

def find_failed_data(data, path):
    saved_data = get_json_data(path)
    if len(saved_data) == 0:
        return data
    result = []
    source = {}
    if "chosen_messages" in saved_data[0]:
        saved_message = set(str(d["chosen_messages"][:-1]) for d in saved_data)
    else:
        saved_message = set(str(d["messages"][:-1]) for d in saved_data)

    for d in data:
        if str(d["chosen"][:-1]) not in saved_message:
            result.append(d)
            if "source" in d:
                if d["source"] in source:
                    source[d["source"]] += 1
                else:
                    source[d["source"]] = 1
    print(source)
    return result

def prepare_reject_sampling_input(data):
    grouped_data = defaultdict(list)

    for entry in data:
        # Convert list of dicts to a hashable representation, like a JSON string
        messages = tuple(json.dumps(msg, sort_keys=True) for msg in entry["messages"])  # Convert each dict to a sorted JSON string
        judgement = entry.get("judgement")
        refined_response = entry.get("refined_response")
        score = entry.get("score")
        # source = entry.get("source")
        grouped_data[messages].append({
            "judgement": judgement,
            "refined_response": refined_response,
            "score": score,
            # "source": source
        })

    # Convert grouped data into the desired output format
    merged_data = [
        {"message": [json.loads(msg) for msg in messages], "refinements": value}
        for messages, value in grouped_data.items()
    ]

    return merged_data

def prepare_pairwise_reject_sampling_input(data):
    grouped_data = defaultdict(list)

    for entry in data:
        # Convert list of dicts to a hashable representation, like a JSON string
        messages = tuple(json.dumps(msg, sort_keys=True) for msg in entry["chosen_messages"])  # Convert each dict to a sorted JSON string
        judgement = entry.get("judgement")
        refined_response = entry.get("refined_response")
        score = entry.get("score")
        # source = entry.get("source")
        rejected_messages = entry.get("rejected_messages")
        grouped_data[messages].append({
            "rejected_messages": rejected_messages,
            "judgement": judgement,
            "refined_response": refined_response,
            "score": score,
            # "source": source
        })

    # Convert grouped data into the desired output format
    merged_data = [
        {"chosen_messages": [json.loads(msg) for msg in messages], "refinements": value}
        for messages, value in grouped_data.items()
    ]
    
    return merged_data

def prepare_cot_sampling_data(data):
    grouped_data = defaultdict(list)

    for entry in data:
        # Convert list of dicts to a hashable representation, like a JSON string
        messages = tuple(json.dumps(msg, sort_keys=True) for msg in entry["messages"])  # Convert each dict to a sorted JSON string
        response = entry.get("response")
        final_result = entry.get("final_result")
        grouped_data[messages].append({
            "response": response,
            "final_result": final_result
        })

    # Convert grouped data into the desired output format
    merged_data = [
        {"messages": [json.loads(msg) for msg in messages], "refinements": value}
        for messages, value in grouped_data.items()
    ]
    
    return merged_data

def extract_final_answer(text):
    match = re.search(r'Final answer:\n\n(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None