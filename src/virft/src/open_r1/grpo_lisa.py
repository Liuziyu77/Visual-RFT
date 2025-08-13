# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#u
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import traceback
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
from transformers import Qwen2VLForConditionalGeneration
import PIL
import numpy as np
from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from sklearn.metrics import f1_score

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

import numpy as np

def compute_giou(gt_bbox, student_bbox):
    x1_gt, y1_gt = gt_bbox[0]
    x2_gt, y2_gt = gt_bbox[1]
    
    x1_st, y1_st = student_bbox[0]
    x2_st, y2_st = student_bbox[1]

    x1_inter = max(x1_gt, x1_st)
    y1_inter = max(y1_gt, y1_st)
    x2_inter = min(x2_gt, x2_st)
    y2_inter = min(y2_gt, y2_st)

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    student_area = (x2_st - x1_st) * (y2_st - y1_st)

    union_area = gt_area + student_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    x1_c = min(x1_gt, x1_st)
    y1_c = min(y1_gt, y1_st)
    x2_c = max(x2_gt, x2_st)
    y2_c = max(y2_gt, y2_st)

    c_area = (x2_c - x1_c) * (y2_c - y1_c)

    giou = iou - (c_area - union_area) / c_area if c_area > 0 else iou

    giou_scaled = giou + 1
    return giou_scaled



def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            # try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            
            try:
                pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
                student_matches = re.findall(pattern, student_answer)
                student_bbox = [(int(x1) / 1000, int(y1) / 1000) for x1, y1, x2, y2 in student_matches] + [(int(x2) / 1000, int(y2) / 1000) for x1, y1, x2, y2 in student_matches]
                gt_matches = re.findall(pattern, ground_truth)
                gt_bbox = [(int(x1) / 1000, int(y1) / 1000) for x1, y1, x2, y2 in gt_matches] + [(int(x2) / 1000, int(y2) / 1000) for x1, y1, x2, y2 in gt_matches]
                reward = compute_giou(gt_bbox, student_bbox)

            except Exception as e:
                # print(traceback.format_exc())
                reward = 0.0
            # Compare the extracted answers
            if student_answer == ground_truth:
                reward = 2.0

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = Dataset.from_json("/path/to/lisa_train_sft.json")

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and your grounding box. Following \"<think> thinking process </think>\n<answer>(x1,y1),(x2,y2)</answer>)\" format."

    def make_conversation(example):
        conversations = example["conversations"]
        formatted_conversation = []
        for message in conversations:
            if message["from"] == "user":
                formatted_conversation.append({"role": "user", "content": message["value"]})
            elif message["from"] == "assistant":
                formatted_conversation.append({"role": "assistant", "content": message["value"]})
        
        return {"prompt": formatted_conversation}

    def make_conversation_image(example):
        conversations = example["conversations"]
        formatted_conversation = []
        for message in conversations:
            if message["from"] == "user":
                # Check if the message contains an image
                if "<img>" in message["value"] and "</img>" in message["value"]:
                    image_path = message["value"].split("<img>")[1].split("</img>")[0].strip()
                    text_content = message["value"].replace(f"<img>{image_path}</img>", "").strip()
                    formatted_conversation.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=text_content)},
                            ],
                        }
                    )
                else:
                    formatted_conversation.append({"role": "user", "content": message["value"]})
            elif message["from"] == "assistant":
                res = '<answer> ' + message["value"] + ' </answer>'
                # formatted_conversation.append({"role": "assistant", "content": message["value"]})
        # print(image_path)
        return {"image": PIL.Image.open(image_path), "prompt": formatted_conversation, 'solution': res}

    dataset = dataset.map(make_conversation_image)

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
