import json
import random
from tqdm import tqdm

JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed above. Your evaluation should focus on the assistant's answer to the last user question and give the strengths and weaknesses of that answer. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response at the end of your answer on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n"""

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

def prepare(data):
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/llama3-ultrafeedback-armorm")['train']
    source_data = []
    for d in ds:
        source_data.append(d)
    source_data = source_data[:]
    correct_lenths = []
    source_lenths = []
    used_messages = []
    result = []
    for d in data:
        message = d["messages"]
        if d["refinement"]["score"] - d["source_score"] <= 0:
            continue
        if not (len(d["refinement"]["refined_response"]) - len(message[-1]["content"]) < 400 or len(d["refinement"]["refined_response"]) / len(message[-1]["content"]) < 1.2):
            continue
        sft_message = message[:]
        sft_message.append({"role": "user", "content": JUDGE_PROMPT})
        sft_message.append({"role": "assistant", "content": d["refinement"]["judgement"]})
        sft_message.append({"role": "user", "content": CORRECT_PROMPT})
        sft_message.append({"role": "assistant", "content": d["refinement"]["refined_response"]})
        result.append({"messages": sft_message})
        correct_lenths.append(len(d["refinement"]["refined_response"]))
        source_lenths.append(len(message[-1]["content"]))
        used_messages.append(json.dumps(message, ensure_ascii=False))
    print(len(correct_lenths))
    print(sum(correct_lenths) / len(correct_lenths))
    print(len(source_lenths))
    print(sum(source_lenths) / len(source_lenths))
    for d in tqdm(source_data):
        if json.dumps(d["chosen"], ensure_ascii=False) not in used_messages:
            result.append({"messages": d["chosen"]})
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
data = deduplicate_data(data)
print(len(data))
random.shuffle(data)
# write_jsonl(data, output_file)
json.dump(data, open(output_file, 'w'))