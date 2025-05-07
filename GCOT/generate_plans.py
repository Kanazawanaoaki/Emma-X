import ast
import json
import multiprocessing
import os
import re
import time
from collections import Counter

import google.generativeai as genai
from tqdm import tqdm
import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  
from dataset import LeRobotEMMADataset


from utils import get_soft_plus_gripper_segment as segment_method
# from utils import get_gripper_segment as segment_method
# from utils import get_soft_segment as segment_method
# from utils import get_nstep_segment as segment_method


model_config = {
    "temperature": 1.5,
    "top_p": 0.99,
    "top_k": 0,
    "max_output_tokens": 4096,
}
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-001", generation_config=model_config
)


def check_valid(response, segment_count):
    try:
        text = response.text
    except Exception:
        return "no response"

    search_result = re.search(r"\{[\s\S]*\}", text)
    if search_result == None:
        return "no dict"
    try:
        match = ast.literal_eval(search_result.group(0))
    except Exception:
        return "no valid dict"

    # Check Format
    for k, v in match.items():
        if len(v) != 2:
            return "wrong format"

    if len(match) != segment_count:
        return "wrong segment number"

    return True


def segment_to_keyinfo(content):
    instruction, images, segment_count = content
    prompt = (
        f"The robot successfully completed a task specified by the instruction: '{instruction}', here are list of segments of images for the robot hand to perform the task specified by instruction."
        "Based on the sequence of segments provided in sequential order, pay attention to the robot hand and identify which subtask it is performing in each segment, and provide the justification for why should the subtask be done based on the environment. You can assign same subtask to multiple segments. "
        "You should output in dictionary format: {segment_number: [subtask, reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment, and output dictionary length should be same as the number of segments "
        f"{segment_count}"
    )
    for _ in range(3):
        while True:
            try:
                response = model.generate_content([prompt, *images])
                break
            except Exception:
                time.sleep(10)
        valid = check_valid(response, segment_count)
        if valid is True:
            return response.text
    return valid


def main(split, load_prev=None):
    repo_id = "IPEC-COMMUNITY/austin_buds_dataset_lerobot"
    dataset = LeRobotEMMADataset(repo_id)
    samples = []

    if load_prev is not None:
        with open(load_prev) as f:
            key_infos = json.load(f)
        print("loaded", len(key_infos))
    else:
        key_infos = dict()

    for i in range(len(dataset)):
        samples = dataset[i]
        outputs, overall_segments = segment_method(samples)
        file_path = str(dataset.meta.get_data_file_path(i))

        model_outputs = segment_to_keyinfo(outputs)


        instruction = outputs[0]
        episode_id = str(dataset.episodes[i])
        key_infos[file_path + "|" + episode_id] = (
            instruction,
            overall_segments.tolist(),
            model_outputs,
        )

    with open(f"plans/plans_{split}.json", "w") as f:
        json.dump(key_infos, f, indent=4)
    ## ends


def valid_stats(split):
    with open(f"plans/plans_{split}.json") as f:
        data = json.load(f)
    print(split, len(data))
    valid = []
    for k, v in data.items():
        _, _, model_output = v
        if model_output in [
            "no response",
            "no dict",
            "no valid dict",
            "wrong format",
            "wrong segment number",
        ]:
            valid.append(model_output)
        else:
            valid.append("valid")
    print(Counter(valid))


if __name__ == "__main__":
    # Below takes 5 hours
    main(split="train")
    main(split="val")

    valid_stats(split="train")
    valid_stats(split="val")

# Last run 16 Nov
# train 38660
# Counter({'valid': 35963, 'wrong segment number': 1930, 'no valid dict': 734, 'wrong format': 28, 'no dict': 5})
# val 5147
# Counter({'valid': 4814, 'wrong segment number': 226, 'no valid dict': 105, 'wrong format': 1, 'no dict': 1})

# Run @ 2 Dec for gripper only
# train 38660
# Counter({'valid': 33417, 'wrong segment number': 3850, 'no valid dict': 1331, 'wrong format': 52, 'no dict': 9, 'no response': 1})
# val 5147
# Counter({'valid': 4436, 'wrong segment number': 531, 'no valid dict': 170, 'wrong format': 9, 'no dict': 1})

# Run @ 2 Dec for soft only
# train 38660
# Counter({'valid': 37473, 'wrong segment number': 902, 'no valid dict': 272, 'wrong format': 8, 'no dict': 5})
# val 5147
# Counter({'valid': 4972, 'wrong segment number': 125, 'no valid dict': 46, 'wrong format': 3, 'no dict': 1})
