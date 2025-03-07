from utils import parse_floats, get_json_data, find_failed_data, get_hf_datasets
from actions import individual_judge, individual_refine, rm_scoring
import argparse
import os
import json
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_NAME=""
TEMPLATE_VERSION=""
RM_NAME = "/path/to/reward/model"

def generate(d, max_iterations=5):
    
    best_score = rm_scoring(d["chosen"][:], RM_NAME)
    best_refine = None
    best_path = []
    result_data = {'messages': d["chosen"][:], 'origin_score': best_score, 'iterations': [], 'final_best': {}, 'best_path': []}
    initial_response = d["chosen"][-1]["content"]
    current_response = initial_response
    previous_score = best_score
    previous_response = initial_response
    for iteration in range(max_iterations):
        iteration_data = {
            'iteration': iteration,
            'initial_response': current_response,
            'judges': [],
            'refine_list': [],
            'scores_list': [],
            'best_refine': None,
            'best_score': None,
            'best_refine_index': None
        }
        
        # Step 1: Judge the response twice
        messages = d["chosen"][:]
        messages[-1] = {"role": "assistant", "content": current_response}
        judges = [individual_judge(messages, MODEL_NAME, 0.7, TEMPLATE_VERSION)[1] for _ in range(2)]
        iteration_data['judges'] = judges

        # Step 2: Refine the response based on the judges' feedback
        refine_list = []
        for judgement in judges:
            if judgement == None:
                refines = None
            else:
                refines = [individual_refine(messages, judgement, MODEL_NAME, 0.7, TEMPLATE_VERSION) for _ in range(2)]
            refine_list.append(refines)
        iteration_data['refine_list'] = refine_list

        # Step 3: Score the refined responses
        scores_list = []
        best_refine_score = -float('inf')
        best_refine_index = None
        for i, refines in enumerate(refine_list):
            if refines == None:
                scores = None
            else:
                scores = []
                for j, refined in enumerate(refines):
                    if refined == None:
                        scores.append(None)
                        continue
                    rm_messages = messages[:]
                    rm_messages[-1] = {"role": "assistant", "content": refined}
                    s = rm_scoring(rm_messages, RM_NAME)
                    scores.append(s)
                    if s > best_refine_score:
                        best_refine_score = s
                        best_refine_index = (i, j)

            scores_list.append(scores)
        iteration_data['scores_list'] = scores_list

        # Step 4: Find the best refine based on score
        best_refine_index = best_refine_index
        if best_refine_index != None:
            best_refine = refine_list[best_refine_index[0]][best_refine_index[1]]
            best_score = best_refine_score
        else:
            best_refine = None

        iteration_data['best_refine_index'] = best_refine_index
        iteration_data['best_refine'] = best_refine
        iteration_data['best_score'] = best_score

        # Update the best path
        # best_path.append(best_refine_index + 1)

        # Append the iteration data
        result_data['iterations'].append(iteration_data)

        # Step 5: Use the best refine as the new response for the next iteration
        current_response = best_refine
        if best_refine == None or best_score <= previous_score:
            current_response = {'response': previous_response, 'score': previous_score}
            break
        else:
            previous_score = best_score
            previous_response = current_response
            result_data['final_best'] = {'response': current_response, 'score': best_score}

    # Step 6: Final result
    # result_data['best_path'] = best_path

    return result_data

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
                        # print("saved")
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
    parser.add_argument('--rm-name', required=True, type=str)
    parser.add_argument('--output-path', required=True, type=str)
    parser.add_argument('--max-workers', type=int, default=64)
    parser.add_argument('--start-point', type=int, default=0)
    parser.add_argument('--sample-nums', type=int, default=100000000)
    # parser.add_argument('--temperatures', type=parse_floats, default=[0.7])
    parser.add_argument('--template-version', type=str, default="v1")

    args = parser.parse_args()
    MODEL_NAME = args.model_name
    RM_NAME = args.rm_name

    # TEMPERATURES = args.temperatures
    TEMPLATE_VERSION = args.template_version

    individual_data_inference(args)
