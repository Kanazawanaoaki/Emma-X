import argparse
import ipdb
import pickle
from PIL import Image
import re
import os

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

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch


torch_device = "cuda"
# model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
model_checkpoint = "OpenGVLab/InternVL3-14B-hf"
# model_checkpoint = "OpenGVLab/InternVL3-38B-hf"
# model_checkpoint = "OpenGVLab/InternVL3-78B-hf"

print(f"use model_checkpoint: {model_checkpoint}")

processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(
    model_checkpoint,
    torch_dtype=torch.bfloat16
).to(torch_device)


instruction, images, segment_count = content
# import ipdb
# ipdb.set_trace()

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
    messages.append([{
        "role": "user",
        "content": content
    }])

    # テンプレート適用＆テンソル化
    inputs = processor.apply_chat_template(
        messages,
        padding=True,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(torch_device, torch.bfloat16)

    # 生成
    # output_ids = model.generate(**inputs, max_new_tokens=25)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,               # 最大生成トークン数を増やす
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # デコード
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)

    def extract_after_assistant(text: str) -> str | None:
        """
        text: ipdb> text で表示された長い文字列
        戻り値: 'assistant\n' 以降のすべての文字列（先頭・末尾の空白を除去）。マッチしない場合は None。
        """
        # DOTALL で改行を含めてマッチ
        pattern = r'assistant\s*\n(.*)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    for text in decoded:
        # print(text)
        print(extract_after_assistant(text))
        # import ipdb
        # ipdb.set_trace()

