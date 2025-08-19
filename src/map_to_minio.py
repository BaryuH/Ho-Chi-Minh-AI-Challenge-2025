import json
import os

esjson_path = r"C:\GIAHUY\AIC2025\data\esjs.json" # need to change
keyframes_minio_path = r"C:\GIAHUY\AIC2025\data\keyframes_minio.json" # need to change
keyframes_base_url = "http://127.0.0.1:9000/keyframes/keyframes_sieve"
videos_base_url = "http://127.0.0.1:9000/videos/videos"

with open(esjson_path, "r", encoding="utf-8") as f:
    data = json.load(f)

video_added = {}

for item in data:
    video_name = item["video_name"]
    frame_name = item["frame_name"]
    item["frame_url"] = f"{keyframes_base_url}/{video_name}/{frame_name}?"
    if video_name not in video_added:
        video_added[video_name] = f"{videos_base_url}/{video_name}.mp4?"
    item["video_url"] = video_added[video_name]

with open(keyframes_minio_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

