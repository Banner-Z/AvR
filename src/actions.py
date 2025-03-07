import json
import numpy as np
import requests
import random
from servers import chat_model, bt_model
from utils import message_to_string, extract_score

def individual_judge(message, model, temperature, template_version):
    SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the last user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response at the end of your answer on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n"""

    USER_PROMPT = """<|The Start of Assistant's Conversation with User|>\n\n{content}\n\n<|The End of Assistant's Conversation with User|>"""
    if template_version == "v1":
        content = message_to_string(message, exchange=False)
        input_message = [{"role": "system", "content": SYSTEM_PROMPT}, 
                {"role": "user", "content": USER_PROMPT.format(content=content)}]
    elif template_version == "v2":
        input_message = message[:]
        input_message.append({"role": "user", "content": SYSTEM_PROMPT.replace("below", "above")})
    else:
        pass
    retry = 0
    score = None
    while retry < 5:
        retry += 1
        result = chat_model(input_message, model, temperature=temperature)
        if result == None:
            continue
        score = extract_score(result)
        if score != None and 1<= score <= 10:
            break
        else:
            # print(f"Format error: {result}, temperature: {temperature}")
            print(f"Format error!")
    if score == None or not (1<= score <= 10):
        # print(f"Retry 5 times, temperature: {temperature}, skip! \n\nresult: {result}")
        print(f"Retry 5 times, temperature: {temperature}, skip!")
        return -1, None
    else:
        return score, result
    
def individual_refine(message, judgement, model, temperature, template_version):
    JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Your evaluation should focus on the assistant's answer to the last user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response at the end of your answer on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n"""

    SYSTEM_PROMPT = """Please revise the AI assistant's response based on the evaluation provided below, addressing any shortcomings mentioned in the review. Your revision should focus solely on improving the assistant's answer to the last user question. Provide the revised response directly, without any additional commentary.\n\n"""

    USER_PROMPT = """<|The Start of Assistant's Conversation with User|>\n\n{content}\n\n<|The End of Assistant's Conversation with User|>\n\n<|The Start of Evaluation|>\n\n{judgement}\n\n<|The End of Evaluation|>"""

    if template_version == "v1":
        content = message_to_string(message, exchange=False)
        input_message = [{"role": "system", "content": SYSTEM_PROMPT}, 
               {"role": "user", "content": USER_PROMPT.format(content=content, judgement=judgement)}]
    elif template_version == "v2":
        input_message = message[:]
        input_message.append({"role": "user", "content": JUDGE_PROMPT.replace("below", "above")})
        input_message.append({"role": "assistant", "content": judgement})
        input_message.append({"role": "user", "content": SYSTEM_PROMPT.replace("below", "above")})
    else:
        pass
    result = chat_model(input_message, model, temperature=temperature)
    return result

def rm_scoring(messages, model_name):
    
    return bt_model(messages, model_name)

def generate_response(messages, model_name, temperature=0.7):
    response = chat_model(messages, model_name, temperature)
    return response

def pairwise_judge(message_1, message_2, model, temperature):
    SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. You should focus on who provides a better answer to the last user question. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict at the end of your answer by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.\n\n"""

    USER_PROMPT = """<|The Start of Conversation|>\n\n{content}\n\n<|The End of Conversation|>\n\n<|The Start of Assistant A's answer|>\n\n### Assistant A:\n{answer_a}\n\n<|The End of Assistant A's answer|>\n\n<|The Start of Assistant B's answer|>\n\n### Assistant B:\n{answer_b}\n\n<|The End of Assistant B's answer|>"""

    content = message_to_string(message_1[:-1], exchange=False)

    input_message = [{"role": "system", "content": SYSTEM_PROMPT}, 
               {"role": "user", "content": USER_PROMPT.format(content=content, answer_a=message_1[-1]["content"], answer_b=message_2[-1]["content"])}]
    
    retry = 0
    choice = None
    while retry < 5:
        result = chat_model(input_message, model, temperature=temperature)
        if "[[A]]" in result:
            choice = 'A'
        elif "[[B]]" in result:
            choice = 'B'
        if choice != None:
            break
        else:
            # print(f"Format error: {result}, temperature: {temperature}")
            print(f"Format error!")
        retry += 1
    if choice == None:
        # print(f"Retry 5 times, temperature: {temperature}, skip! \n\nresult: {result}")
        print(f"Retry 5 times, temperature: {temperature}, skip!")
        return None, None
    else:
        return choice, result
    
def pairwise_refine(message_1, message_2, judgement, model, temperature):
    SYSTEM_PROMPT = """Please create a new response based on the two provided AI assistant responses and their evaluation. Your goal is to combine the strengths of both responses while addressing the shortcomings mentioned in the evaluation. Focus on crafting a single, coherent, and high-quality response that effectively answers the last user question. Use the evaluation as a guide to refine and balance the tone, content, and accuracy of the response. Provide the revised response directly, without any additional commentary.\n\n"""

    USER_PROMPT = """<|The Start of Conversation|>\n\n{content}\n\n<|The End of Conversation|>\n\n<|The Start of Assistant A's answer|>\n\n### Assistant A:\n{answer_a}\n\n<|The End of Assistant A's answer|>\n\n<|The Start of Assistant B's answer|>\n\n### Assistant B:\n{answer_b}\n\n<|The End of Assistant B's answer|>\n\n<|The Start of Evaluation|>\n\n{judgement}\n\n<|The End of Evaluation|>"""

    content = message_to_string(message_1[:-1], exchange=False)

    input_message = [{"role": "system", "content": SYSTEM_PROMPT}, 
               {"role": "user", "content": USER_PROMPT.format(content=content, answer_a=message_1[-1]["content"], answer_b=message_2[-1]["content"], judgement=judgement)}]
    
    result = chat_model(input_message, model, temperature=temperature)
    if result is not None:
        result = result.replace("### Assistant:\n", "").replace("### Assistant: \n", "").replace("### Assistant:", "").replace("Assistant:\n", "")

    return result

def chat_cot_model(message, model_name, temperature=0.7):
    
    chat_response = chat_model(message, model_name, temperature=temperature)
    return chat_response