import json
import random
from tqdm import tqdm

JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed above. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the last user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response at the end of your answer on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n"""

CORRECT_PROMPT = """Please revise the AI assistant's response based on the evaluation provided above, addressing any shortcomings mentioned in the review. Your revision should focus solely on improving the the assistant's answer to the last user question. Provide the revised response directly, without any additional commentary.\n\n"""

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

def message_to_string(message, exchange=False):
    result = ""
    for m in message:
        if (m["role"] == "user" and not exchange) or (m["role"] == "assistant" and exchange):
            result += "### User:\n" + m["content"] + "\n"
        elif (m["role"] == "assistant" and not exchange) or (m["role"] == "user" and exchange):
            result += "\n### Assistant:\n" + m["content"] + "\n"
    return result

def find_extreme_responses(data):
    if not data or len(data) < 2:
        return None, None, None, None

    sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)

    highest_score_response = sorted_data[0]
    lowest_score_response = sorted_data[-1]

    if highest_score_response['score'] <= lowest_score_response['score']:
        return None, None, None, None

    return highest_score_response['refined_response'], lowest_score_response['refined_response'], highest_score_response['score'], lowest_score_response['score']

import re
from collections import Counter

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
        # import pdb;pdb.set_trace()
        return True
    return False

def prepare(data):
    result = []
    rejected_lenths = []
    chosen_lenths = []

    response_pair_counts = 0
    judge_pair_counts = 0
    correct_pair_counts = 0

    ## origin setting
    count = 0
    for d in data:
        message = d["messages"][:]
        if check_redundancy(d["messages"][-1]["content"]):
            print(f"""!--!!\nerror in response !!!!\n{d["messages"][-1]["content"]}""")
            continue
        if d["refinement"] is None:
            count += 1
            continue
        if check_redundancy(d["refinement"]["refined_response"]):
            print(f"""!--!!\nerror in response !!!!\n{d["refinement"]["refined_response"]}""")
            continue

        if d["refinement"]["score"] - d["source_score"] <= 0:
            continue
        if d["refinement"]["score"] - d["source_score"] > 0:

            result.append({
                "messages": d["messages"][:-1],
                "chosen": {"role": "assistant", "content": d["refinement"]["refined_response"]},
                "rejected": d["messages"][-1]
            })
            response_pair_counts += 1

        if len(d["best_judgement"]["refinements"]) < 1 or len(d["worst_judgement"]["refinements"]) < 1:
            print("\n!-!-!\none of the judgement's refinment counts < 1\n")
            continue
        bj_scores = [x["score"] for x in d["best_judgement"]["refinements"]]
        avg_bj_scores = sum(bj_scores) / len(bj_scores)
        wj_scores = [x["score"] for x in d["worst_judgement"]["refinements"]]
        avg_wj_scores = sum(wj_scores) / len(wj_scores)

        judge_message = message[:]
        judge_message.append({"role": "user", "content": JUDGE_PROMPT})

        correct_message = judge_message[:]
        correct_message.append({"role": "assistant", "content": d["best_judgement"]["judgement"]})
        correct_message.append({"role": "user", "content": CORRECT_PROMPT})

        if d["best_judgement"]["refinements"] == None:
            continue

        chosen_correct, rejected_correct, chosen_score, rejected_score = find_extreme_responses(d["best_judgement"]["refinements"])

        if chosen_score!= None and chosen_score < d["source_score"]:
            print(f"\n!-!-!\nchosen_score < source_score:\n{json.dumps(correct_message, ensure_ascii=False, indent=2)}\n\n{chosen_correct}\n")
            continue

        if check_redundancy(d["best_judgement"]["judgement"]):
            print(f"""!--!!\nerror in judgement !!!!\n{d["best_judgement"]["judgement"]}""")
            continue

        if d["best_judgement"]["judgement"] == d["worst_judgement"]["judgement"] or avg_bj_scores <= avg_wj_scores or max(bj_scores) < d["source_score"]:
            print(f"\nno better judgement\n\n")
            continue
        else:
            if check_redundancy(d["worst_judgement"]["judgement"]):
                print(f"""!--!!\nerror in judgement !!!!\n{d["worst_judgement"]["judgement"]}""")
            else:
                result.append({
                    "messages": judge_message,
                    "chosen": {"role": "assistant", "content": d["best_judgement"]["judgement"]},
                    "rejected": {"role": "assistant", "content": d["worst_judgement"]["judgement"]}
                })
                judge_pair_counts += 1

        if chosen_correct != None and rejected_correct != None:
            if rejected_score == d["source_score"]:
                continue
            if check_redundancy(chosen_correct) or check_redundancy(rejected_correct):
                print(f"!--!!\nerror in correct !!!!\n{chosen_correct}\n\n{rejected_correct}")
            else:
                result.append({
                    "messages": correct_message,
                    "chosen": {"role": "assistant", "content": chosen_correct},
                    "rejected": {"role": "assistant", "content": rejected_correct}
                })
                rejected_lenths.append(len(rejected_correct))
                chosen_lenths.append(len(chosen_correct))
                correct_pair_counts += 1
                
    print(f"\nresponse_pair_counts: {response_pair_counts}\njudge_pair_counts: {judge_pair_counts}\ncorrect_pair_counts: {correct_pair_counts}")
    
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