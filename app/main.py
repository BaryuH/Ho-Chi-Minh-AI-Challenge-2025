from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import numpy as np
import torch
from sklearn.preprocessing import normalize
from io import BytesIO
from pymilvus import connections, Collection
from elasticsearch import Elasticsearch
import sys
import os
from PIL import Image
import requests

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'models')))

# ============================================================
from vith14_model import VITH14Model
from vitg14_model import VITG14Model
from beit3_model import BEiT3FeatureExtractor
CHECKPOINT_PATH = r"C:\D\02_AIC\PATH\beit3_large_itc_patch16_224.pth"  # ndeed to change
# need to change
WEIGHT_PATH = r"C:\D\02_AIC\PATH\beit3_large_patch16_384_f30k_retrieval.pth"
SENTENCEPIECE_PATH = r"C:\D\02_AIC\PATH\beit3.spm"  # need to change
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =======================MODEL=====================================
vith14 = None  # VITH14Model()
beit3 = BEiT3FeatureExtractor(
    model_path=WEIGHT_PATH,
    sentencepiece_model_path=SENTENCEPIECE_PATH,
    device=DEVICE,
    model_type='large',
    image_size=384,
    img_size=384,
    patch_size=16,
    drop_path_rate=0.1
)
vitg14 = None  # VITG14Model()
# ============================================================
# Kết nối Milvus và Elasticsearch
connections.connect(host="localhost", port="19530")
collection_name = ["vith14_vec", "beit3_vec", "vitg14_vec"]
collection1, collection2, collection3 = [
    Collection(name) for name in collection_name]
es = Elasticsearch("http://localhost:9200")

# ============================================================
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000", # Port của FE trong Docker
    "http://localhost:5173", # Port của FE khi chạy "npm run dev"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== SUB FUNCTIONS ===========================
def normalize_embedding(embedding):
    return normalize(embedding.reshape(1, -1))[0]


def prepare_image_for_model(image: Image.Image) -> BytesIO:
    buf = BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return buf


def get_fused_embedding(image, x=1/3, y=1/3, z=1/3):
    image_buf = prepare_image_for_model(image)
    embedding_vith14 = np.array(vith14.get_image_embedding(image_buf))
    image_buf.seek(0)
    embedding_beit3 = np.array(beit3.get_image_embedding(image_buf))
    image_buf.seek(0)
    embedding_vitg14 = np.array(vitg14.get_image_embedding(image_buf))
    embedding_vith14 = normalize_embedding(embedding_vith14)
    embedding_beit3 = normalize_embedding(embedding_beit3)
    embedding_vitg14 = normalize_embedding(embedding_vitg14)
    fused_embedding = (x * embedding_vith14 + y *
                       embedding_beit3 + z * embedding_vitg14)
    return fused_embedding


def load_image_from_bytes(file_content: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(file_content)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi đọc ảnh: {e}")


def load_image_from_url(url: str) -> Image.Image:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return load_image_from_bytes(resp.content)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Lỗi khi tải ảnh từ URL: {e}")


def filter_ids_by_es(
    ocr_keywords: Optional[List[str]],
    asr_keywords: Optional[List[str]]
) -> Optional[List[str]]:
    if not ocr_keywords and not asr_keywords:
        return None

    must_clauses = []

    # --- OCR keywords fuzzy ---
    if ocr_keywords:
        for kw in ocr_keywords:
            must_clauses.append({
                "fuzzy": {
                    "ocr_text": {
                        "value": kw,
                        "fuzziness": "AUTO"
                    }
                }
            })

    # --- ASR keywords fuzzy ---
    if asr_keywords:
        should_clauses = [{"fuzzy": {"asr_text": {"value": kw, "fuzziness": "AUTO"}}}
                          for kw in asr_keywords]
        must_clauses.append({
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        })

    es_query = {
        "query": {
            "bool": {
                "must": must_clauses
            }
        },
        "_source": ["global_id"],
        "size": 1000
    }

    res = es.search(index="frame_info", body=es_query)
    candidate_ids = [hit["_source"]["global_id"]
                     for hit in res["hits"]["hits"]]

    if not candidate_ids:
        return ["__none__"]  
    return candidate_ids



def fetch_meta_by_global_ids(global_ids: List[str]) -> dict:
    """
    Trả về dict: global_id -> {frame_name, video_name, timestamp_ms, frame_url, video_url}
    lấy từ Elasticsearch index 'frame_info'
    """
    if not global_ids:
        return {}

    # Ưu tiên match theo keyword để tránh tokenization
    es_query_kw = {
        "query": {"terms": {"global_id.keyword": global_ids}},
        "_source": ["global_id", "frame_name", "video_name", "timestamp_ms", "timestamps_ms", "frame_url", "video_url"],
        "size": len(global_ids)
    }
    try:
        res = es.search(index="frame_info", body=es_query_kw)
        hits = res.get("hits", {}).get("hits", [])

        # Fallback nếu index không có subfield .keyword
        if not hits:
            es_query_plain = {
                "query": {"terms": {"global_id": global_ids}},
                "_source": ["global_id", "frame_name", "video_name", "timestamp_ms", "timestamps_ms", "frame_url", "video_url"],
                "size": len(global_ids)
            }
            res = es.search(index="frame_info", body=es_query_plain)
            hits = res.get("hits", {}).get("hits", [])

        meta_map = {}
        for hit in hits:
            src = hit.get("_source", {})
            gid = src.get("global_id")
            if gid:
                # chấp nhận cả timestamp_ms và timestamps_ms
                ts = src.get("timestamp_ms", src.get("timestamps_ms"))
                meta_map[gid] = {
                    "frame_name": src.get("frame_name"),
                    "video_name": src.get("video_name"),
                    "timestamp_ms": ts,
                    "frame_url": src.get("frame_url"),
                    "video_url": src.get("video_url"),
                }
        return meta_map
    except Exception as e:
        print(f"[fetch_meta_by_global_ids] ES error: {e}")
        return {}

# ------------------ API SEARCH IMAGE ------------------


@app.post("/search_image")
async def search_image(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Query(None, description="URL ảnh"),
    model: str = Query("fused", enum=["vith14", "beit3", "vitg14", "fused"]),
    top_k: int = Query(25, ge=1, le=400),
    ocr_keywords: Optional[List[str]] = Query(
        None, description="Từ khóa OCR để filter"),
    asr_keywords: Optional[List[str]] = Query(
        None, description="Từ khóa ASR để filter")
):
    if image:
        content = await image.read()
        img = load_image_from_bytes(content)
    elif image_url:
        img = load_image_from_url(image_url)
    else:
        raise HTTPException(
            status_code=400, detail="Cần cung cấp ảnh qua file hoặc URL.")

    if model == "fused":
        embedding = get_fused_embedding(img)
        collection = collection2
        anns_field = "beit3_vec"
    elif model == "vith14":
        img_buf = prepare_image_for_model(img)
        embedding = vith14.get_image_embedding(img_buf)
        collection = collection1
        anns_field = "vith14_vec"
    elif model == "beit3":
        img_buf = prepare_image_for_model(img)
        embedding = beit3.get_image_embedding(img_buf)
        collection = collection2
        anns_field = "beit3_vec"
    elif model == "vitg14":
        img_buf = prepare_image_for_model(img)
        embedding = vitg14.get_image_embedding(img_buf)
        collection = collection3
        anns_field = "vitg14_vec"

    candidate_ids = filter_ids_by_es(ocr_keywords, asr_keywords)

    if candidate_ids is not None:
        results = collection.search(
            [embedding.tolist()],
            anns_field=anns_field,
            param={"metric_type": "IP", "params": {"nprobe": 1000}},
            limit=top_k,
            expr=f'global_id in {candidate_ids}',
            output_fields=["global_id"]
        )
    else:
        results = collection.search(
            [embedding.tolist()],
            anns_field=anns_field,
            param={"metric_type": "IP", "params": {"nprobe": 1000}},
            limit=top_k,
            output_fields=["global_id"]
        )

    # --- Chuẩn bị kết quả (enrich metadata từ ES) ---
    global_ids = [
        hit.entity.get("global_id")
        for hit in results[0]
        if getattr(hit, "entity", None) is not None and hit.entity.get("global_id") is not None
    ]
    meta_map = fetch_meta_by_global_ids(global_ids)
    result_data = []
    for rank, hit in enumerate(results[0], start=1):
        gid = hit.entity.get("global_id") if getattr(
            hit, "entity", None) is not None else None
        meta = meta_map.get(gid, {}) if gid is not None else {}
        result_data.append({
            "rank": rank,
            "id": gid,             
            "score": hit.distance,
            "frame_name": meta.get("frame_name"),
            "video_name": meta.get("video_name"),
            "timestamp_ms": meta.get("timestamp_ms"),
            "frame_url": meta.get("frame_url"),
            "video_url": meta.get("video_url"),
        })
    return {"results": result_data}


# ------------------ API SEARCH TEXT ------------------
@app.post("/search_text")
async def search_text(
    query: str = Query(..., description="Nội dung văn bản để tìm kiếm"),
    model: str = Query("vith14", enum=["vith14", "beit3", "vitg14"]),
    top_k: int = Query(25, ge=25, le=400),
    ocr_keywords: Optional[List[str]] = Query(
        None, description="Từ khóa OCR để filter"),
    asr_keywords: Optional[List[str]] = Query(
        None, description="Từ khóa ASR để filter")
):
    candidate_ids = filter_ids_by_es(ocr_keywords, asr_keywords)

    results = []
    if model == "vith14":
        embedding = vith14.get_text_embedding(query)
        if candidate_ids is not None:
            results = collection1.search(
                [embedding.tolist()],
                anns_field="vith14_vec",
                param={"metric_type": "IP", "params": {"nprobe": 500}},
                limit=top_k,
                expr=f'global_id in {candidate_ids}',
                output_fields=["global_id"]
            )
        else:
            results = collection1.search(
                [embedding.tolist()],
                anns_field="vith14_vec",
                param={"metric_type": "IP", "params": {"nprobe": 500}},
                limit=top_k,
                output_fields=["global_id"]
            )
    elif model == "beit3":
        embedding = beit3.get_text_embedding(query)
        if candidate_ids is not None:
            results = collection2.search(
                [embedding.tolist()],
                anns_field="beit3_vec",
                param={"metric_type": "IP", "params": {"nprobe": 1000}},
                limit=top_k,
                expr=f'global_id in {candidate_ids}',
                output_fields=["global_id"]
            )
        else:
            results = collection2.search(
                [embedding.tolist()],
                anns_field="beit3_vec",
                param={"metric_type": "IP", "params": {"nprobe": 1000}},
                limit=top_k,
                output_fields=["global_id"]
            )
    elif model == "vitg14":
        embedding = vitg14.get_text_embedding(query)
        if candidate_ids is not None:
            results = collection3.search(
                [embedding.tolist()],
                anns_field="vitg14_vec",
                param={"metric_type": "IP", "params": {"nprobe": 1000}},
                limit=top_k,
                expr=f'global_id in {candidate_ids}',
                output_fields=["global_id"]
            )
        else:
            results = collection3.search(
                [embedding.tolist()],
                anns_field="vitg14_vec",
                param={"metric_type": "IP", "params": {"nprobe": 1000}},
                limit=top_k,
                output_fields=["global_id"]
            )

    global_ids = [
        hit.entity.get("global_id")
        for hit in results[0]
        if getattr(hit, "entity", None) is not None and hit.entity.get("global_id") is not None
    ]
    meta_map = fetch_meta_by_global_ids(global_ids)
    result_data = []
    for rank, hit in enumerate(results[0], start=1):
        gid = hit.entity.get("global_id") if getattr(
            hit, "entity", None) is not None else None
        meta = meta_map.get(gid, {}) if gid is not None else {}
        result_data.append({
            "rank": rank,
            "id": gid,                  
            "score": hit.distance,
            "frame_name": meta.get("frame_name"),
            "video_name": meta.get("video_name"),
            "timestamp_ms": meta.get("timestamp_ms"),
            "frame_url": meta.get("frame_url"),
            "video_url": meta.get("video_url"),
        })
    return {"results": result_data}
