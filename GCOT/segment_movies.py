import os
import argparse
import pickle
from PIL import ImageDraw, ImageFont, Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

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

output_dir = f"output_segments/output_sample_{args.episode_id}"
os.makedirs(output_dir, exist_ok=True)
video_path = os.path.join(output_dir, "segments_video.mp4")

# デフォルトフォントを取得（必要に応じて .ttf を指定してください）
font = ImageFont.load_default()

segment_idx = None
count = 0
annotated_files = []

for item in images:
    if isinstance(item, str) and item.startswith("Segment"):
        # "Segment N:" の文字列から N を抽出
        segment_idx = int(item.split()[1].rstrip(':'))
        count = 0
    elif isinstance(item, Image.Image):
        if segment_idx is None:
            # 番号が見つかっていない場合はスキップ
            continue

        img = item.copy()
        draw = ImageDraw.Draw(img)

        # 描き込むテキスト
        text = f"Segment {segment_idx}-{count+1}"
        # テキスト位置
        x, y = 5, 5
        # テキスト描画（fill=(R,G,B) で色指定。デフォルトは黒）
        draw.text((x, y), text, font=font, fill=(255, 0, 0))

        # ファイル名を決定（例: segment_1_0.png, segment_1_1.png, ...）
        filename = f"segment_{segment_idx}_{count}.png"
        path = os.path.join(output_dir, filename)
        img.save(path)
        print(f"Saved: {path}")
        annotated_files.append(path)

        count += 1

# フレームレート（1秒あたりの画像数）
fps = 1  # 必要に応じて変更してください

# クリップを作成して書き出し
clip = ImageSequenceClip(annotated_files, fps=fps)
clip.write_videofile(video_path, codec="libx264")

print(f"動画を作成しました: {video_path}")

# import ipdb
# ipdb.set_trace()



