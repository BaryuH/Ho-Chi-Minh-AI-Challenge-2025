from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
import json
import os

# ====== 1. Kết nối Milvus ======
connections.connect(host="localhost", port="19530")
print("Connected to Milvus!")

# ====== 2. Hàm tạo schema cho collection ======
def create_schema(vector_field_name, dim):
    return CollectionSchema([
        FieldSchema(name="global_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="video_name", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="frame_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="frame_index", dtype=DataType.INT64),
        FieldSchema(name="timestamp_ms", dtype=DataType.INT64),
        FieldSchema(name=vector_field_name, dtype=DataType.FLOAT_VECTOR, dim=dim)
    ])

# ====== 3. Tạo collection nếu chưa có ======
def create_collection(name, vector_field_name, dim):
    if not utility.has_collection(name):
        schema = create_schema(vector_field_name, dim)
        collection = Collection(name, schema)
        print(f"Created collection: {name} with dim {dim}")
        return collection
    else:
        collection = Collection(name)
        print(f"Collection {name} already exists")
        return collection

# ====== 4. Load JSONL ======
def load_jsonl(file_path):
    frames = []
    if not os.path.exists(file_path):
        print(f"File {file_path} không tồn tại, bỏ qua")
        return frames

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in {file_path}: {e}")
    return frames

# ====== 5. Hàm index JSONL vào Milvus ======
def index_milvus(jsonl_path, collection_name, vector_field_name, batch_size=1000):
    frames = load_jsonl(jsonl_path)
    if not frames:
        print(f"{jsonl_path} rỗng, bỏ qua")
        return
    
    sample_vec = frames[0].get(vector_field_name)
    if not sample_vec:
        raise ValueError(f"Không tìm thấy {vector_field_name} trong {jsonl_path}")
    dim = len(sample_vec)
    for frame in frames:
        vec = frame.get(vector_field_name)
        if vec and len(vec) != dim:
            raise ValueError(f"Vector dimension mismatch in {jsonl_path}: expected {dim}, got {len(vec)}")
    collection = create_collection(collection_name, vector_field_name, dim)
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        global_ids, video_names, frame_names, frame_indices, timestamps_ms, vectors = [], [], [], [], [], []

        for frame in batch:
            vec = frame.get(vector_field_name)
            if vec:
                global_ids.append(frame["global_id"])
                video_names.append(frame["video_name"])
                frame_names.append(frame["frame_name"])
                frame_indices.append(frame["frame_index"])
                timestamps_ms.append(frame["timestamp_ms"])
                vectors.append(vec)
        
        if vectors:
            collection.insert([global_ids, video_names, frame_names, frame_indices, timestamps_ms, vectors])
    
    collection.create_index(vector_field_name, {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 1024}
    })
    collection.load()
    print(f"Created index and loaded collection {collection_name}")

# ====== 6. Index 3 file JSONL ======
index_milvus("beit3_milvus.jsonl", "beit3_vec", "beit3_vec")
index_milvus("vith14_milvus.jsonl", "vith14_vec", "vith14_vec")
index_milvus("vitg14_milvus.jsonl", "vitg14_vec", "vitg14_vec")

print("🎉 Hoàn tất index tất cả các collection!")