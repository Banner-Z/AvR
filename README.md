# AvR: Alignment via Refinement
The official repository of paper [Unlocking Recursive Thinking of LLMs: Alignment via Refinement](https://aclanthology.org/2025.findings-acl.582/)

long CoT training data: [huggingface dataset](https://huggingface.co/datasets/zhk/ASCENT)

## Set up enviroment
```
pip install vllm>=0.6.6.post1
```

## Main steps

**Don't forget to set your own file and model paths in the scripts.**

### Stage 1
```
cd AvR

# use VLLM to deploy the model used to generate sft data.
bash scripts/run_generative_model_server.sh

# generate sft data for stage 1
bash scripts/stage1_generation.sh

# use VLLM to deploy Bradley-Terry reward model.
bash scripts/run_reward_model_server.sh

# # scoring (for reject sampling) (Set "output-structure" in the code to "sft")
bash scripts/stage1_scoring.sh

# prepare training data
python src/data/prepare_sft_data.py

# train your RSFT model with the frameworks like LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory.git)

# use VLLM to deploy the RSFT model.
bash scripts/run_generative_model_server.sh

# generate dpo data for stage 1 (Set "model-name" in the code to your RSFT model)
bash scripts/stage1_generation.sh

# scoring (for reject sampling) (Set "output-structure" in the code to "dpo")
bash scripts/stage1_scoring.sh

# prepare dpo training data
python src/data/prepare_dpo_data.py

# train your DPO model with the frameworks like LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory.git)
```

### Stage 2
```
# use VLLM to deploy stage 1 model.
bash scripts/run_generative_model_server.sh

# use VLLM to deploy Bradley-Terry reward model.
bash scripts/run_reward_model_server.sh

# generate sft data for stage 2 in a greedy search way.
bash scripts/cot_generation.sh

# prepare training data
python src/data/prepare_cot_sft_data_greedy.py

# train your RSFT model with the frameworks like LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory.git)

# use VLLM to deploy the RSFT model.
bash scripts/run_generative_model_server.sh

# generate dpo data for stage 2
bash scripts/cot_dpo_generation.sh

# scoring (for reject sampling)
bash scripts/cot_dpo_scoring.sh

# prepare dpo training data (for length control)
python src/data/prepare_cot_dpo_data.py

# train your DPO model with the frameworks like LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory.git)
```

## Evaluation

Refer to [README](https://github.com/Banner-Z/ASCENT/blob/main/src/eval/README.md)

## Citation
```
@inproceedings{zhang-etal-2025-unlocking,
    title = "Unlocking Recursive Thinking of {LLM}s: Alignment via Refinement",
    author = "Zhang, Haoke  and
      Liang, Xiaobo  and
      Wang, Cunxiang  and
      Li, Juntao  and
      Zhang, Min",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.582/",
    pages = "11169--11182",
    ISBN = "979-8-89176-256-5"
}
```
