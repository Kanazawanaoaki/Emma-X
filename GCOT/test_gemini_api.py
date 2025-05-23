import os
import google.generativeai as genai
import pickle
import argparse

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

instruction, images, segment_count = content
prompt = (
    f"The robot successfully completed a task specified by the instruction: '{instruction}', here are list of segments of images for the robot hand to perform the task specified by instruction."
    "Based on the sequence of segments provided in sequential order, pay attention to the robot hand and identify which subtask it is performing in each segment, and provide the justification for why should the subtask be done based on the environment. You can assign same subtask to multiple segments. "
    "You should output in dictionary format: {segment_number: [subtask, reason for justification], ...} format, segment_number starts from 1 and must be an integer, the output dictionary key correspond to each segment, and output dictionary length should be same as the number of segments "
    f"{segment_count}"
)
print(prompt)
response = model.generate_content([prompt, *images])
print(response.text)
