from openai import OpenAI
import random
import json
import numpy as np
import os
from transformers import AutoTokenizer
import requests

def chat_model(message, model_name, temperature=0.7):
    openai_api_key = "EMPTY"
    openai_api_base = "http://0.0.0.0:8055/v1"
    port = random.choice(["11101", "11102", "11103", "11104", "11105", "11106", "11107", "11108"])

    openai_api_base = openai_api_base.replace("8055", port)
    
    MODEl_NAME=model_name

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    try:
        chat_response = client.chat.completions.create(
            model=MODEl_NAME,
            messages=message,
            temperature=temperature,
            top_p=0.8,
            # max_completion_tokens=4096
        )
        response = json.loads(chat_response.model_dump_json())["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        return None
    return response

def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response

def bt_model(messages, model_name):
    
    api_url = f"http://0.0.0.0:8055/pooling"
    port = random.choice(["12321", "12322", "12323", "12324"])  
    api_url = api_url.replace("8055", port)

    prompt = {
        "model":
        model_name,
        "messages": messages
    }
    retry = 0
    score = None
    while retry < 3:
        retry += 1
        try:
            pooling_response = post_http_request(prompt=prompt, api_url=api_url)
            score = pooling_response.json()["data"][0]["data"][0]
            break
        except Exception as e:
            print(f"\nrm error: {e}, 'pooling_response': {str(pooling_response.json())}\n\nmessages: {str(messages)}\n\n")
    return score