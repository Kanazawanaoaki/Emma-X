import argparse
11;rgb:3030/0a0a/2424import pickle
import os
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
# from transformers import AutoProcessor, AutoModelForImageTextToText

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="Process episode data from a pickle file.")
parser.add_argument('--episode_id','-e', default=0, type=int, help="The episode_id (integer) to load the corresponding file.")
args = parser.parse_args()

# episode_idを使ってファイル名を作成
file_name = f"tuple_data_{args.episode_id}.pkl"
pkl_file = os.path.join("pickles", file_name)

# pickleファイルを読み込み
with open(pkl_file, 'rb') as f:
    content = pickle.load(f)

print(f"load pickle file from {pkl_file}")

# Qwenの最新VLMモデルを設定
# model_checkpoint = "Qwen/Qwen2.5-VL-32B-Instruct"
model_checkpoint = "Qwen/Qwen2.5-VL-72B-Instruct"

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



# pickleから読み込んだデータの処理
instruction, images, segment_count = content

segment_dict = {}
current_seg = None
img_idx = -1  # only_images のインデックスカウンタ
only_images = []

# セグメントの画像インデックスを作成
for item in images:
    if isinstance(item, str) and item.startswith("Segment"):
        seg_num = item.split()[1].rstrip(':')
        current_seg = f"Segment {seg_num}"
        segment_dict[current_seg] = []
    elif isinstance(item, Image.Image):
        img_idx += 1
        only_images.append(item)
        if current_seg is not None:
            segment_dict[current_seg].append(img_idx)

# メッセージの定義
messages = [
    {
        "role": "user",
        "content": [
            *[{"type": "image", "image": img} for img in only_images],
            {"type": "text", "text": (
                f"The robot successfully completed a task specified by the instruction: '{instruction}', here are all images for the robot hand to perform the task specified by instruction."
                f"Each segment contains images whose indexes are represented by a dictionary {segment_dict}."
                "Based on the sequence of segments provided in sequential order, pay attention to the robot hand and identify which subtask it is performing in each segment, and provide the justification for why should the subtask be done based on the environment. You can assign same subtask to multiple segments. "
                "You should output in dictionary format: {segment_number: [subtask, reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment, and output dictionary length should be same as the number of segments "
                f"{segment_count}"
            )}
        ]
    }

    # ## not use segment dict info
    # {
    #     "role": "user",
    #     "content": [
    #         *[{"type": "image", "image": img} for img in only_images],
    #         {"type": "text",  "text": (
    #             f"The robot successfully completed a task specified by the instruction: '{instruction}', here are all images for the robot hand to perform the task specified by instruction."
    #             "Pay attention to the movements of the robot's hands in the images, divide the entire sequence of tasks into segments, identify the subtasks being performed, and explain why those subtasks should be performed based on the environment. You can not assign same subtask to multiple segments. "
    #             "You should output in dictionary format: {segment_number: [subtask, [list of indexes of images included in the segment],  reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment, and output dictionary length should be same as the number of segments "
    #             f"{segment_count}"
    #         )}
    #     ]
    # }

]

# # プロセッサでメッセージをテンプレート化＆テンソル化
# inputs = processor.apply_chat_template(
#     messages,
#     padding=True,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt"
# ).to(torch_device, torch.bfloat16)

# # モデルで生成
# output_ids = model.generate(
#     **inputs,
#     max_new_tokens=1000,               # 最大生成トークン数を増やす
#     early_stopping=True,               # EOSトークンに達したら早めに止める
#     eos_token_id=processor.tokenizer.eos_token_id,
#     pad_token_id=processor.tokenizer.pad_token_id,
# )

# # デコードして結果を表示
# decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
# for text in decoded:
#     print(text)

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
generated_ids = model.generate(**inputs, max_new_tokens=1000)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
