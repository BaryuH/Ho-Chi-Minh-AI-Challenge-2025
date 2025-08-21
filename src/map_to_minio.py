import json

esjson_path = r"C:\GIAHUY\AIC2025\data\esjs.json"  
keyframes_minio_path = r"C:\GIAHUY\AIC2025\data\keyframes_minio.json"  

keyframes_base_url = "http://127.0.0.1:9000/keyframes/keyframes_sieve"
videos_base_url = "http://127.0.0.1:9000/videos/videos"

with open(esjson_path, "r", encoding="utf-8") as f:
    data = json.load(f)   

for item in data:
    item.pop("objects", None)
    video_name = item["video_name"]
    frame_name = item["frame_name"]
    # thêm trường URL
    item["frame_url"] = f"{keyframes_base_url}/{video_name}/{frame_name}?"
    item["video_url"] = f"{videos_base_url}/{video_name}.mp4?"

with open(keyframes_minio_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Đã thêm frame_url, video_url cho {len(data)} bản ghi.")
