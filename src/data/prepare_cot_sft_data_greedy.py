import json
import random
from tqdm import tqdm

def deduplicate_data(data):
    unique_lines = set()
    result = []

    for d in tqdm(data):
        line = json.dumps(d, ensure_ascii=False)
        message = json.dumps(d["messages"])
        if line and message not in unique_lines:
            unique_lines.add(message)
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

# find the loop in output
def check_redundancy(text, threshold=60):
    def tokenize(text):
        if re.search(r'[a-zA-Z]', text):
            words = re.findall(r'\b\w+\b', text.lower())
        else:
            words = list(text)
        return words
    
    def check_symbol_and_number_redundancy(text):
        text = text.replace(' ', '')
        symbols = re.findall(r'[\d\W]+', text)
        symbol_count = Counter(symbols)
        for symbol, count in symbol_count.items():
            if count >= threshold:
                return True
        return False
    
    def check_word_redundancy(text):
        words = tokenize(text)
        word_count = Counter(words)
        for word, count in word_count.items():
            if count >= threshold:
                return True
        return False
    
    if check_symbol_and_number_redundancy(text) or check_word_redundancy(text):
        return True
    return False
    
def prepare(data):
    final_lenths = []
    source_lenths = []
    total_lenths = []
    iters = []
    records = []
    wrong_count = 0
    for d in data:
        message = d["messages"]
        if check_redundancy(d["messages"][-1]["content"]) or len(d["iterations"]) == 0:
            continue
        
        result = f"""<|Start of recursive criticism and improvement|>\n\n## Let's answer the question first:\n{message[-1]["content"]}\n\n"""
        source_lenths.append(len(message[-1]["content"]))
        
        if d["final_best"] == {}:
            if d["iterations"][0]["best_refine_index"] == None:
                result = None
                continue
            result += f"""## Now, let's try to criticize this answer:\n{d["iterations"][0]["judges"][d["iterations"][0]["best_refine_index"][0]]}\n\n"""
            best_answer = message[-1]["content"]
            continue
        else:
            best_answer = d["final_best"]["response"]
            best_score = d["final_best"]["score"]
            for i, iteration in enumerate(d["iterations"]):
                if iteration['best_score'] == best_score and i == len(d["iterations"]) - 1:
                    iters.append(i+1)
                    result = None
                    break
                elif iteration['best_score'] == best_score:
                    if d["iterations"][i+1]["best_refine_index"] == None:
                        result = None
                        break
                    result += f"""## Now, let's try to criticize this answer:\n{iteration["judges"][iteration["best_refine_index"][0]]}\n\n"""
                    result += f"""## Okey, let's improve the above answer based on the criticism:\n{best_answer}\n\n"""
                    result += f"""## Now, let's try to criticize this answer:\n{d["iterations"][i+1]["judges"][d["iterations"][i+1]["best_refine_index"][0]]}\n\n"""
                    break
                else:
                    result += f"""## Now, let's try to criticize this answer:\n{iteration["judges"][iteration["best_refine_index"][0]]}\n\n"""
                    result += f"""## Okey, let's improve the above answer based on the criticism:\n{iteration["refine_list"][iteration["best_refine_index"][0]][iteration["best_refine_index"][1]]}\n\n"""
            if result == None:
                continue
        result += """## Okay, now it’s almost done.\n\n<|End of recursive criticism and improvement|>\n\n"""
        result += f"""Final answer:\n\n{best_answer}"""

        final_lenths.append(len(best_answer))
        sft_message = message[:-1]
        sft_message.append({"role": "assistant", "content": result})
        records.append({"messages": sft_message, "score": d["final_best"]["score"] - d['origin_score']})
        source_lenths.append(len(message[-1]["content"]))
        total_lenths.append(len(result))

    final_records = [{"messages": r["messages"]} for r in records]
    print(len(total_lenths))
    print(sum(total_lenths) / len(total_lenths))
    print(len(source_lenths))
    print(sum(source_lenths) / len(source_lenths))
    print(len(final_lenths))
    print(sum(final_lenths) / len(final_lenths))
    print(len(iters))
    print(f"wrong_count: {wrong_count}")    
    return final_records

def write_jsonl(data, output_file):
    with open(output_file, 'w') as o:
        for l in data:
            o.write(json.dumps(l, ensure_ascii=False) + '\n')

input_file = "data/input-path.jsonl"

output_file = "data/prepared-training-data.json"

data = get_json_data(input_file)
print(len(data))
data = prepare(data)
data = deduplicate_data(data)
print(len(data))
random.shuffle(data)
# write_jsonl(data, output_file)
json.dump(data, open(output_file, 'w'))