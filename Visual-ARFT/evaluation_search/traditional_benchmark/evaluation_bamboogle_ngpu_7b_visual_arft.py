import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tools.web_search import SearchSession

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools
import itertools
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool

system_prompt = """# Role  
You are a step-by-step multimodal reasoning assistant.  
Given an image, a question, and optional partial reasoning chain, your task is to solve the problem **one substep at a time**.  

# Guiding Principles  
At each turn, you must **either**:  
1. Issue **one specific, text-only search** enclosed in <search> </search> tags,  
2. Or provide the **final answer** enclosed in <answer> </answer> tags.  

All outputs **must begin with a thought** enclosed in <think> </think> tags, explaining your current reasoning and what to do next.  

- Do not reference “the image” in your searches.  
- Do not repeat past queries.  
- Only output **one action per step**: either <search> or <answer>, never both.  
- When ready to conclude, summarize reasoning and give a final answer.

# Output Format (strict):  
Always start with <think>. Do not output the previous reasoning chain. Then, depending on the case, output one of the following:

## 1. If reasoning continues:  
<think> Your current reasoning and next plan </think>  
<search> One precise, retrievable textual query </search>

## 2. If ready to conclude:  
<think> Summarize all reasoning and derive the answer </think>  
<answer> Final answer, as briefly as possible </answer>

# Current reasoning chain:
"""

model_path = '/share_models/Qwen2.5-VL-7B-Instruct_GRPO_agent_search_data20_63_gpu8/checkpoint-200/'

def run(rank, world_size):
    ### 多卡时候，device_map需要设置为cpu，再分配到不同GPU上，不能设置为 auto
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='cpu',
    )
    processor = AutoProcessor.from_pretrained(model_path)

    model = model.to(torch.device(rank))
    model = model.eval()

    # 加载数据集数据
    wikimultihopqa = []
    with open('/bamboogle/test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                wikimultihopqa.append(json.loads(line))
    print(len(wikimultihopqa))

    print(wikimultihopqa[0]['question'])
    print(wikimultihopqa[0]['golden_answers'])
    print("Rank:" + str(rank))
    print("World Size:" + str(world_size))
    import math
    split_length = math.ceil(len(wikimultihopqa)/world_size)
    print("Split Chunk Length:" + str(split_length))
    split_wikimultihopqa = wikimultihopqa[int(rank*split_length) : int((rank+1)*split_length)]
    print(len(split_wikimultihopqa))
    wikimultihopqa = split_wikimultihopqa

    combine_results = []
    for i in tqdm(range(len(wikimultihopqa))):
        pred_answer = None
        search_session = SearchSession()
        query = wikimultihopqa[i]['question']
        answer = wikimultihopqa[i]['golden_answers']
        item_id = wikimultihopqa[i]['id']
        input_text = system_prompt + '\n' +f'<query> {query} </query>'
        # print("################################################################")
        # print(query)
        # print(answer)

        try:
            iterative_num = 0
            while iterative_num<5: # 推理轮次大于 5 次就退出
                iterative_num += 1
                messages = [
                    { "role": "user", 
                        "content": [{"type": "text", "text": input_text}]}
                ]
                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    # images=image_inputs,
                    # videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=2048)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                result = output_text[0]
                # print(result)

                ###### 进行一轮搜索或者终止 ######
                if '<answer>' in result:
                    match = re.search(r"<answer>\s*(.*?)\s*</answer>", result)
                    if match:
                        pred_answer = match.group(1).strip()
                    # print(pred_answer)
                    input_text = input_text +'\n'+ result
                    break
                elif '<search>' in result:
                    match = re.search(r"<search>\s*(.*?)\s*</search>", result)
                    if match:
                        search_content = match.group(1).strip()
                    # Web search excerpts feed the next reasoning turn.
                    search_results = search_session.search(search_content, 4)
                    # print(search_results)
                    format_search_results = '<information> '
                    for index,item in enumerate(search_results):
                        format_search_results += f'{index+1}.' + ' '+"Content:"+item['body']
                    format_search_results += ' </information> '
                    # print(format_search_results)
                
                    ###### prompt更新 ######
                    input_text = input_text +'\n'+ result +'\n'+ format_search_results
        except Exception as e:
            logger.info("ERROR OCCURES")
            logger.info({e})
        if pred_answer != None:
            combine_results.append(
                {'id': item_id, 'pred_answer': pred_answer, 'gt': answer, 'query': query}
            )
        else:
            combine_results.append(
                {'id': item_id, 'pred_answer': None, 'gt': answer, 'query': query}
            )
    with open(f"7b_step200_bamboogle_web_n4_{rank}.json", "w", encoding="utf-8") as f:
        json.dump(combine_results, f, ensure_ascii=False, indent=4)
    return combine_results

def main():
    multiprocess = torch.cuda.device_count() >= 2
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        global_results = []
        for i in range(world_size):
            global_results = global_results + result_lists[i]
            
        logger.info("Done")
        logger.info('finished running')
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()