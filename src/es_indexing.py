import json
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=None,
    headers={"es-version-check": "false"}  
)

if not es.indices.exists(index="frame_info"):
    es.indices.create(index="frame_info")

file_path = "C:/GIAHUY/AIC2025/data/keyframes_minio.json" # need to change 
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

actions = [
    {
        "_index": "frame_info",
        "_id": frame["global_id"],  
        "_source": frame
    }
    for frame in data
]

helpers.bulk(es, actions)
print("Indexing completed!")
