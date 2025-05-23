import argparse
import ipdb
import pickle
import os
from PIL import Image

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

# メッセージの定義（URL の代わりに PIL.Image を渡す）
messages = [
    # [
    #     {
    #         "role": "user",
    #         "content": [
    #             *[{"type": "image", "image": img} for img in only_images],
    #             {"type": "text",  "text": (
    #                 f"The robot successfully completed a task specified by the instruction: '{instruction}', here are all images for the robot hand to perform the task specified by instruction."
    #                 f"Each segment contains images whose indexes are represented by a dictionary {segment_dict}."
    #                 "Based on the sequence of segments provided in sequential order, pay attention to the robot hand and identify which subtask it is performing in each segment, and provide the justification for why should the subtask be done based on the environment. You can assign same subtask to multiple segments. "
    #                 "You should output in dictionary format: {segment_number: [subtask, reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment, and output dictionary length should be same as the number of segments "
    #                 f"{segment_count}"
    #                 )
    #              },
    #         ],
    #     },
    # ],


    # ## not use instruction info.
    # [
    #     {
    #         "role": "user",
    #         "content": [
    #             *[{"type": "image", "image": img} for img in only_images],
    #             {"type": "text",  "text": (
    #                 f"Here are all images for the robot hand to perform the task specified by instruction."
    #                 f"Each segment contains images whose indexes are represented by a dictionary {segment_dict}."
    #                 "Based on the sequence of segments provided in sequential order, pay attention to the robot hand and identify which subtask it is performing in each segment, and provide the justification for why should the subtask be done based on the environment. You can assign same subtask to multiple segments. "
    #                 "You should output in dictionary format: {segment_number: [subtask, reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment, and output dictionary length should be same as the number of segments "
    #                 f"{segment_count}"
    #                 )
    #              },
    #         ],
    #     },
    # ],

    ## not use segment dict info
    [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in only_images],
                {"type": "text",  "text": (
                    f"The robot successfully completed a task specified by the instruction: '{instruction}', here are all images for the robot hand to perform the task specified by instruction."
                    "Pay attention to the movements of the robot's hands in the images, divide the entire sequence of tasks into segments, identify the subtasks being performed, and explain why those subtasks should be performed based on the environment. You can not assign same subtask to multiple segments. "
                    "You should output in dictionary format: {segment_number: [subtask, [list of indexes of images included in the segment],  reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment, and output dictionary length should be same as the number of segments "
                    f"{segment_count}"
                    )
                 },
            ],
        },
    ],

    # ## not use segment dict info, make sure image index(not work well)
    # [
    #     {
    #         "role": "user",
    #         "content": [
    #             *[{"type": "image", "image": img} for img in only_images],
    #             {"type": "text",  "text": (
    #                 f"The robot successfully completed a task specified by the instruction: '{instruction}', here are all images for the robot hand to perform the task specified by instruction."
    #                 "Pay attention to the movements of the robot's hands in the images, divide the entire sequence of tasks into segments, identify the subtasks being performed, and explain why those subtasks should be performed based on the environment. You can not assign same subtask to multiple segments. "
    #                 "You should output in dictionary format: {segment_number: [subtask, [list of indexes of images included in the segment],  reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment, and output dictionary length should be same as the number of segments "
    #                 f"{segment_count}"
    #                 f"Make sure that all image indexes between 0 and {len(only_images) - 1} are included in one of the segments without overlapping between segments. Also, make sure that no other indexes are included."
    #                 )
    #              },
    #         ],
    #     },
    # ],



    # ## not use segment dict info and instruction
    # [
    #     {
    #         "role": "user",
    #         "content": [
    #             *[{"type": "image", "image": img} for img in only_images],
    #             {"type": "text",  "text": (
    #                 f"Here are all images for the robot hand to perform the task specified by instruction."
    #                 "Pay attention to the movements of the robot's hands in the images, divide the entire sequence of tasks into segments, identify the subtasks being performed, and explain why those subtasks should be performed based on the environment. You can not assign same subtask to multiple segments. "
    #                 "You should output in dictionary format: {segment_number: [subtask, [list of indexes of images included in the segment], reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment, and output dictionary length should be same as the number of segments "
    #                 f"{segment_count}"
    #                 )
    #              },
    #         ],
    #     },
    # ],

    # ## not use segment info and segment_count with image index
    # [
    #     {
    #         "role": "user",
    #         "content": [
    #             *[{"type": "image", "image": img} for img in only_images],
    #             {"type": "text",  "text": (
    #                 f"The robot successfully completed a task specified by the instruction: '{instruction}', here are all images for the robot hand to perform the task specified by instruction."
    #                 "Pay attention to the movements of the robot's hands in the images, divide the entire sequence of tasks into segments, identify the subtasks being performed, and explain why those subtasks should be performed based on the environment. You can not assign same subtask to multiple segments. "
    #                 "You should output in dictionary format: {segment_number: [subtask, [list of indexes of images included in the segment], reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment."
    #             )
    #              },
    #         ],
    #     },
    # ],


    # ## not use segment info and segment_count and instruction
    # [
    #     {
    #         "role": "user",
    #         "content": [
    #             *[{"type": "image", "image": img} for img in only_images],
    #             {"type": "text",  "text": (
    #                 f"Here are all images for the robot hand to perform the task specified by instruction."
    #                 "Pay attention to the movements of the robot's hands in the images, divide the entire sequence of tasks into segments, identify the subtasks being performed, and explain why those subtasks should be performed based on the environment. You can not assign same subtask to multiple segments. "
    #                 "You should output in dictionary format: {segment_number: [subtask, [list of indexes of images included in the segment], reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment."
    #             )
    #              },
    #         ],
    #     },
    # ],

    # ## not use segment info and segment_count with make sure image index output(not work well)
    # [
    #     {
    #         "role": "user",
    #         "content": [
    #             *[{"type": "image", "image": img} for img in only_images],
    #             {"type": "text",  "text": (
    #                 f"The robot successfully completed a task specified by the instruction: '{instruction}', here are all images for the robot hand to perform the task specified by instruction."
    #                 "Pay attention to the movements of the robot's hands in the images, divide the entire sequence of tasks into segments, identify the subtasks being performed, and explain why those subtasks should be performed based on the environment. You can not assign same subtask to multiple segments. "
    #                 "You should output in dictionary format: {segment_number: [subtask, [list of indexes of images included in the segment], reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment."
    #                 f"Make sure that all image indexes between 0 and {len(only_images) - 1} are included in one of the segments without overlapping between segments. Also, make sure that no other indexes are included."
    #             )
    #              },
    #         ],
    #     },
    # ],

    # ## not use segment info and segment_count and instruction  with make sure image index output(not work well)
    # [
    #     {
    #         "role": "user",
    #         "content": [
    #             *[{"type": "image", "image": img} for img in only_images],
    #             {"type": "text",  "text": (
    #                 f"Here are all images for the robot hand to perform the task specified by instruction."
    #                 "Pay attention to the movements of the robot's hands in the images, divide the entire sequence of tasks into segments, identify the subtasks being performed, and explain why those subtasks should be performed based on the environment. You can not assign same subtask to multiple segments. "
    #                 "You should output in dictionary format: {segment_number: [subtask, [list of indexes of images included in the segment], reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment."
    #                 f"Make sure that all image indexes between 0 and {len(only_images) - 1} are included in one of the segments without overlapping between segments. Also, make sure that no other indexes are included."
    #             )
    #              },
    #         ],
    #     },
    # ],


]


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
    max_new_tokens=1000,               # 最大生成トークン数を増やす
    early_stopping=True,             # EOS に達したら早めに止める
    eos_token_id=processor.tokenizer.eos_token_id,
    pad_token_id=processor.tokenizer.pad_token_id,
)

# デコード
decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
for text in decoded:
    print(text)

