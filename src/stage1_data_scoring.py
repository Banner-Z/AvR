from utils import parse_floats, get_json_data, prepare_reject_sampling_input
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
    messages = d["message"]
    source_score = rm_scoring(messages, MODEL_NAME)
    detailed_result = {"messages": messages, "source_score": source_score, "refinements": []}
    best_score = source_score
    best_record = None
    best_LC_record = None
    best_LC_score = source_score
    for record in d["refinements"]:
        new_messages = messages[:]
        new_messages[-1] = {"role": "assistant", "content": record["refined_response"]}
        score = rm_scoring(new_messages, MODEL_NAME)
        if score == None:
            continue
        record["score"] = score
        detailed_result["refinements"].append(record)
        if score != None and score >= best_score:
            best_score = score
            best_record = record
        # Filter responses with large length differences
        # if len(d["message"][-1]["content"]) - len(record["refined_response"]) > -400 or len(record["refined_response"]) / len(d["message"][-1]["content"]) < 1.2:
        #     if score != None and score >= best_LC_score:
        #         best_score = score
        #         best_LC_record = record
    detailed_result["best_record"] = best_record
    # detailed_result["best_LC_record"] = best_LC_record

    if best_record == None:
        result = {"scoring_detail": detailed_result}
        return result
    if OUTPUT_STRUCTURE == 'sft':
        result = {"scoring_detail": detailed_result, "filtered_data": {"messages": messages, "source_score": source_score,"refinement": best_record}}
        return result
    elif OUTPUT_STRUCTURE == 'dpo':
        j_dict = {}
        for r in detailed_result["refinements"]:
            j = json.dumps(r["judgement"], ensure_ascii=False)
            if j in j_dict:
                j_dict[j].append({"refined_response": r["refined_response"], "score": r["score"]})
            else:
                j_dict[j] = [{"refined_response": r["refined_response"], "score": r["score"]}]
        best_j = None
        best_j_score = -10000000
        worst_j = None
        worst_j_score = 10000000

        for j, r_list in j_dict.items():
            score_list = [i["score"] for i in r_list]
            j_score = sum(score_list) / len(score_list)
            if j_score > best_j_score:
                best_j_score = j_score
                best_j = {"judgement": json.loads(j), "refinements": r_list, "avg_score": j_score}
            if j_score < worst_j_score:
                worst_j_score = j_score
                worst_j = {"judgement": json.loads(j), "refinements": r_list, "avg_score": j_score}

        # for LC ablation. LC record was not used in the main experiment, because we did not find any benefit in our preliminary experiments.
        if best_LC_record == None:
            result = {"scoring_detail": detailed_result, "filtered_data": {"messages": messages, "source_score": source_score, "refinement": best_record, "best_judgement": best_j, "worst_judgement":worst_j, "LC_refinement": None, "best_LC_judgement": None, "worst_LC_judgement": None}}
        else:
            j_dict = {}
            for r in detailed_result["refinements"]:
                if len(d["message"][-1]["content"]) - len(r["refined_response"]) > -400 or len(r["refined_response"]) / len(d["message"][-1]["content"]) < 1.2:
                    j = json.dumps(r["judgement"], ensure_ascii=False)
                    if j in j_dict:
                        j_dict[j].append({"refined_response": r["refined_response"], "score": r["score"]})
                    else:
                        j_dict[j] = [{"refined_response": r["refined_response"], "score": r["score"]}]
            best_LC_j = None
            best_LC_j_score = -10000000
            worst_LC_j = None
            worst_LC_j_score = 10000000

            for j, r_list in j_dict.items():
                score_list = [i["score"] for i in r_list]
                j_score = sum(score_list) / len(score_list)
                if j_score > best_LC_j_score:
                    best_LC_j_score = j_score
                    best_LC_j = {"judgement": json.loads(j), "refinements": r_list, "avg_score": j_score}
                if j_score < worst_LC_j_score:
                    worst_LC_j_score = j_score
                    worst_LC_j = {"judgement": json.loads(j), "refinements": r_list, "avg_score": j_score}

            result = {"scoring_detail": detailed_result, "filtered_data": {"messages": messages, "source_score": source_score, "refinement": best_record, "best_judgement": best_j, "worst_judgement":worst_j, "LC_refinement": best_LC_record, "best_LC_judgement": best_LC_j, "worst_LC_judgement": worst_LC_j}}
        return result
    else:
        raise NotImplementedError("OUTPUT_STRUCTURE should be sft or dpo !")
        
def individual_data_sampling(args):
    data = get_json_data(args.input_file)
    max_points = min(len(data), args.sample_nums)
    data = data[:max_points]
    data = prepare_reject_sampling_input(data)
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
