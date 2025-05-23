import argparse
import ipdb
import pickle
from PIL import Image
import re
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser(description="Process episode data from a pickle file.")
parser.add_argument('--episode_id','-e', default=0, type=int, help="The episode_id (integer) to load the corresponding file.")

# 引数をパース
args = parser.parse_args()

# episode_idを使ってファイル名を作成
file_name = f"tuple_data_{args.episode_id}.pkl"

pkl_file = os.path.join("pickles", file_name)
with open(pkl_file, 'rb') as f:
    content = pickle.load(f)

print(f"load pickle file from {pkl_file}")


# Qwenの最新VLMモデルを設定
model_checkpoint = "Qwen/Qwen2.5-VL-32B-Instruct"  # 最新のQwenモデルを指定

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_checkpoint, torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-32B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_checkpoint)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

print(f"use model_checkpoint: {model_checkpoint}")


instruction, images, segment_count = content

segment_dict = {}
current_seg = None
img_idx = -1  # only_images の index カウンタ

only_images = []
for item in images:
    if isinstance(item, str) and item.startswith("Segment"):
        # "Segment N:" からキー用の "Segment N" を作成
        seg_num = item.split()[1].rstrip(':')
        current_seg = f"Segment {seg_num}"
        segment_dict[current_seg] = []
    elif isinstance(item, Image.Image):
        # 画像に遭遇するたびに only_images のインデックスをインクリメント
        img_idx += 1
        only_images.append(item)
        if current_seg is not None:
            segment_dict[current_seg].append(img_idx)

# only_images = [x for x in images if isinstance(x, Image.Image)]
print(segment_dict)


sorted_segments = sorted(
    segment_dict.keys(),
    key=lambda s: int(s.split()[1])
)

print(sorted_segments)

# 各セグメントについて画像を追加
seg_cnt = 0
for seg in sorted_segments:
    indices = segment_dict[seg]
    messages = []
    content = []
    # print(indices)
    for idx in indices:
        content.append({
            "type": "image",
            "image": only_images[idx]
        })
    seg_cnt += 1
    content.append({
        "type": "text",
        "text": (
            f"The robot successfully completed a task specified by the instruction: '{instruction}', hera are images contained in the {seg_cnt} segment of the entire task execution consisting of {segment_count} segments."
            "Focus on the robot's hand movements in the provided segment, identify the subtasks being performed in this segment, and explain why this subtask should be performed based on the environment."
            "You should output in dictionary format: {segment_number: [subtask, reason for justification], ...} format, segment_number must be an integer, the output dictionary key correspond to each segment."
        )
    })
    messages.append({
        "role": "user",
        "content": content
    })

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=80)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)


# print(messages)

