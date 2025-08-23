from fastapi import FastAPI, Query, UploadFile, File, HTTPException
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
from fastapi.middleware.cors import CORSMiddleware 
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'models')))

# ============================================================
from vith14_model import VITH14Model
from vitg14_model import VITG14Model
from beit3_model import BEiT3FeatureExtractor
CHECKPOINT_PATH = r"C:\GIAHUY\AIC2025\beit3_large_itc_patch16_224.pth"  # ndeed to change
# need to change
WEIGHT_PATH = r"C:\GIAHUY\AIC2025\beit3_large_patch16_384_f30k_retrieval.pth"
SENTENCEPIECE_PATH = r"C:\GIAHUY\AIC2025\beit3.spm"  # need to change
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
MINIMUM_SHOULD_MATCH = "30%"  
MAX_VIDEOS = 50               
MAX_SEGMENTS_PER_VIDEO = 30    
MAX_FRAMES_PER_SEGMENT = 50  
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


def filter_ids_by_es(
    ocr_query: Optional[str],
    asr_query: Optional[str]
) -> Optional[List[str]]:
    if not ocr_query and not asr_query:
        return None

    must_clauses = []

    if ocr_query:
        must_clauses.append({
            "match": {
                "ocr_text": {
                    "query": ocr_query,
                    "operator": "or",
                    "minimum_should_match": MINIMUM_SHOULD_MATCH
                }
            }
        })

    if asr_query:
        must_clauses.append({
            "match": {
                "asr_text": {
                    "query": asr_query,
                    "operator": "or",
                    "minimum_should_match": MINIMUM_SHOULD_MATCH
                }
            }
        })

    es_query = {
        "query": {"bool": {"must": must_clauses}},
        "_source": ["global_id"],
        "size": 1000
    }

    res = es.search(index="frame_info", body=es_query)
    candidate_ids = [hit["_source"]["global_id"] for hit in res["hits"]["hits"]]

    if not candidate_ids:
        return ["__none__"]
    return candidate_ids


def fetch_meta_by_global_ids(global_ids: List[str]) -> dict:
    if not global_ids:
        return {}

    es_query_kw = {
        "query": {"terms": {"global_id.keyword": global_ids}},
        "_source": ["global_id", "frame_name", "video_name", "timestamp_ms", "timestamps_ms", "frame_url", "video_url"],
        "size": len(global_ids)
    }
    try:
        res = es.search(index="frame_info", body=es_query_kw)
        hits = res.get("hits", {}).get("hits", [])


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

# ------------------ SEARCH IMAGE ------------------
@app.post("/search_image")
async def search_image(
    image: Optional[UploadFile] = File(None),
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


# ------------------ SEARCH TEXT ------------------
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

# ================== OCR SEARCH ==================
@app.post("/search_ocr")
async def search_ocr(
    query: str = Query(..., description="Câu OCR để match"),
    top_k: int = Query(25, ge=1, le=400)
):
    es_body = {
        "query": {
            "match": {
                "ocr_text": {
                    "query": query,
                    "operator": "or",
                    "minimum_should_match": MINIMUM_SHOULD_MATCH
                }
            }
        },
        "_source": ["global_id", "frame_name", "video_name", "timestamp_ms", "timestamps_ms", "frame_url", "video_url"],
        "size": top_k
    }
    try:
        res = es.search(index="frame_info", body=es_body)
        hits = res.get("hits", {}).get("hits", [])
        global_ids = [h["_source"]["global_id"] for h in hits if "_source" in h]
        meta_map = fetch_meta_by_global_ids(global_ids)

        results = []
        for rank, h in enumerate(hits, start=1):
            src = h["_source"]
            gid = src.get("global_id")
            meta = meta_map.get(gid, {})
            results.append({
                "rank": rank,
                "id": gid,
                "score": h.get("_score"),
                "frame_name": meta.get("frame_name", src.get("frame_name")),
                "video_name": meta.get("video_name", src.get("video_name")),
                "timestamp_ms": meta.get("timestamp_ms", src.get("timestamp_ms", src.get("timestamps_ms"))),
                "frame_url": meta.get("frame_url", src.get("frame_url")),
                "video_url": meta.get("video_url", src.get("video_url")),
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi OCR search: {e}")
    
# ================== ASR SEARCH ==================
@app.post("/search_asr")
async def search_asr(
    query: str = Query(..., description="ASR text to match"),
    top_videos: int = 50,
    top_segments_per_video: int = 10,
    top_frames_per_segment: int = 50
):
    base_query = {
        "match": {
            "asr_text": {
                "query": query,
                "operator": "or",
                "minimum_should_match": MINIMUM_SHOULD_MATCH  
            }
        }
    }

    es_body = {
        "size": 0,               
        "query": base_query,
        "aggs": {
            "by_video": {
                "terms": {
                    "field": "video_name.keyword",
                    "size": top_videos,
                    "order": {"_count": "desc"}  
                },
                "aggs": {
                    "by_asr": {
                        "terms": {
                            "field": "asr_text.keyword",
                            "size": top_segments_per_video,
                            "order": {"min_ts": "asc"}  
                        },
                        "aggs": {
                            "min_ts": { "min": { "field": "timestamp_ms" } },
                            "frames": {
                                "top_hits": {
                                    "size": top_frames_per_segment,
                                    "sort": [{ "timestamp_ms": { "order": "asc" } }],
                                    "_source": {
                                        "includes": [
                                            "global_id",
                                            "frame_name",
                                            "video_name",
                                            "timestamp_ms",
                                            "timestamps_ms",
                                            "frame_url",
                                            "video_url",
                                            "asr_text"
                                        ]
                                    }
                                }
                            }
                        }
                    },
                    "first_ts": { "min": { "field": "timestamp_ms" } }  
                }
            }
        }
    }

    try:
        res = es.search(index="frame_info", body=es_body)
        buckets_video = res.get("aggregations", {}).get("by_video", {}).get("buckets", [])

        out = {"videos": []}
        for vb in buckets_video:
            video_name = vb.get("key")
            segment_buckets = vb.get("by_asr", {}).get("buckets", [])

            segments_out = []
            for sb in segment_buckets:
                asr_text = sb.get("key")
                hits = sb.get("frames", {}).get("hits", {}).get("hits", [])
                frames = []
                for h in hits:
                    src = h.get("_source", {}) or {}
                    frames.append({
                        "id": src.get("global_id"),
                        "frame_name": src.get("frame_name"),
                        "timestamp_ms": src.get("timestamp_ms", src.get("timestamps_ms")),
                        "frame_url": src.get("frame_url"),
                        "video_url": src.get("video_url")
                    })
                segments_out.append({
                    "asr_text": asr_text,
                    "min_timestamp_ms": sb.get("min_ts", {}).get("value"),
                    "frames": frames
                })

            out["videos"].append({
                "video_name": video_name,
                "first_timestamp_ms": vb.get("first_ts", {}).get("value"),
                "segments": segments_out
            })

        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi ASR grouped search: {e}")
    


# ================== TEMPORAL SEARCH==================

GAP_MS = 500  
def _search_event_candidates_text(
    event_text: str,
    model: str,
    top_k: int,
    ocr_query: Optional[str],
    asr_query: Optional[str]
):
    candidate_ids = filter_ids_by_es(ocr_query, asr_query)
    if model == "vith14":
        emb = vith14.get_text_embedding(event_text)
        coll = collection1
        anns_field = "vith14_vec"
        params = {"metric_type": "IP", "params": {"nprobe": 500}}
    elif model == "beit3":
        emb = beit3.get_text_embedding(event_text)
        coll = collection2
        anns_field = "beit3_vec"
        params = {"metric_type": "IP", "params": {"nprobe": 1000}}
    else:  # "vitg14"
        emb = vitg14.get_text_embedding(event_text)
        coll = collection3
        anns_field = "vitg14_vec"
        params = {"metric_type": "IP", "params": {"nprobe": 1000}}

    if candidate_ids is not None:
        results = coll.search(
            [emb.tolist()],
            anns_field=anns_field,
            param=params,
            limit=top_k,
            expr=f'global_id in {candidate_ids}',
            output_fields=["global_id"]
        )
    else:
        results = coll.search(
            [emb.tolist()],
            anns_field=anns_field,
            param=params,
            limit=top_k,
            output_fields=["global_id"]
        )

    # Enrich meta từ ES
    gids = [
        hit.entity.get("global_id")
        for hit in results[0]
        if getattr(hit, "entity", None) is not None and hit.entity.get("global_id") is not None
    ]
    meta_map = fetch_meta_by_global_ids(gids)

    out = []
    for hit in results[0]:
        gid = hit.entity.get("global_id") if getattr(hit, "entity", None) is not None else None
        if not gid:
            continue
        m = meta_map.get(gid, {})
        ts = m.get("timestamp_ms")
        vid = m.get("video_name")
        if ts is None or vid is None:
            continue
        out.append({
            "id": gid,
            "score": hit.distance,
            "video": vid,
            "ts": ts,
            "frame_name": m.get("frame_name"),
            "frame_url": m.get("frame_url"),
            "video_url": m.get("video_url"),
        })
    return out


def _segmentize(frames):
    if not frames:
        return []
    frames = sorted(frames, key=lambda x: x["timestamp_ms"])
    segs, cur = [], [frames[0]]
    for f in frames[1:]:
        if f["timestamp_ms"] - cur[-1]["timestamp_ms"] <= GAP_MS:
            cur.append(f)
        else:
            segs.append({"start_ms": cur[0]["timestamp_ms"], "end_ms": cur[-1]["timestamp_ms"], "frames": cur})
            cur = [f]
    segs.append({"start_ms": cur[0]["timestamp_ms"], "end_ms": cur[-1]["timestamp_ms"], "frames": cur})
    return segs


def _dp_monotone_chain(stages_by_video):
    out = {}
    for video, stages in stages_by_video.items():
        m = len(stages)
        if m == 0:
            continue
        if any(len(st) == 0 for st in stages):
            continue
        cands = [sorted(st, key=lambda x: x["ts"]) for st in stages]
        idx_back = []  
        dp_prev = [c["score"] for c in cands[0]]
        idx_prev = list(range(len(cands[0])))
        idx_back.append([-1 for _ in cands[0]])  
        for i in range(1, m):
            a = cands[i-1]  
            b = cands[i]    
            dp_curr = [float("-inf")] * len(b)
            back_curr = [-1] * len(b)
            k = 0
            best_val = float("-inf")
            best_k = -1
            for j in range(len(b)):
                ts_j = b[j]["ts"]
                while k < len(a) and a[k]["ts"] <= ts_j:
                    if dp_prev[k] > best_val:
                        best_val = dp_prev[k]
                        best_k = k
                    k += 1
                if best_k != -1:
                    dp_curr[j] = b[j]["score"] + best_val
                    back_curr[j] = best_k

            dp_prev = dp_curr
            idx_back.append(back_curr)
        last_stage = len(cands) - 1
        if all(v == float("-inf") for v in dp_prev):
            continue
        j_best = max(range(len(dp_prev)), key=lambda j: dp_prev[j])
        total = dp_prev[j_best]
        path = []
        cur_j = j_best
        for i in range(last_stage, -1, -1):
            f = cands[i][cur_j]
            path.append({
                "event_index": i,
                "id": f["id"],
                "timestamp_ms": f["ts"],
                "frame_name": f["frame_name"],
                "frame_url": f["frame_url"],
                "video_url": f["video_url"]
            })
            cur_j = idx_back[i][cur_j] if i > 0 else -1
            if cur_j == -1 and i > 0:
                break
        path.reverse()

        out[video] = {
            "total_score": total,
            "events": path,
            "segments": _segmentize(path)
        }
    return out


@app.post("/search_temporal")
async def search_temporal(
    q1: str = Query(..., description="Mô tả sự kiện 1 (text search)"),
    q2: Optional[str] = Query(None, description="Mô tả sự kiện 2 (text search)"),
    q3: Optional[str] = Query(None, description="Mô tả sự kiện 3 (text search)"),
    ocr1: Optional[str] = Query(None), asr1: Optional[str] = Query(None),
    ocr2: Optional[str] = Query(None), asr2: Optional[str] = Query(None),
    ocr3: Optional[str] = Query(None), asr3: Optional[str] = Query(None),
    model: str = Query("beit3", enum=["vith14", "beit3", "vitg14"]),
    topk_per_event: int = Query(200, ge=10, le=1000)
):
    try:
        stages_raw = []

        c1 = _search_event_candidates_text(q1, model, topk_per_event, ocr1, asr1)
        stages_raw.append(c1)

        if q2:
            c2 = _search_event_candidates_text(q2, model, topk_per_event, ocr2, asr2)
            stages_raw.append(c2)

        if q3:
            c3 = _search_event_candidates_text(q3, model, topk_per_event, ocr3, asr3)
            stages_raw.append(c3)

        m = len(stages_raw)
        if m == 0:
            raise HTTPException(status_code=400, detail="Không có sự kiện nào hợp lệ.")

        stages_by_video = {}
        for i, cand_list in enumerate(stages_raw):
            for c in cand_list:
                v = c["video"]
                stages_by_video.setdefault(v, [ [] for _ in range(m) ])
                stages_by_video[v][i].append(c)

        if m == 1:
            out = {"videos": []}
            for v, stage_lists in stages_by_video.items():
                s0 = sorted(stage_lists[0], key=lambda x: x["score"], reverse=True)
                if not s0:
                    continue
                best = s0[0]
                events = [{
                    "event_index": 0,
                    "id": best["id"],
                    "timestamp_ms": best["ts"],
                    "frame_name": best["frame_name"],
                    "frame_url": best["frame_url"],
                    "video_url": best["video_url"]
                }]
                out["videos"].append({
                    "video_name": v,
                    "best_sequence": {
                        "total_score": best["score"],
                        "events": events,
                        "segments": _segmentize(events)
                    }
                })
            return out

        best_by_video = _dp_monotone_chain(stages_by_video)

        resp = {"videos": []}
        for v, seq in best_by_video.items():
            resp["videos"].append({
                "video_name": v,
                "best_sequence": seq
            })
        return resp

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi temporal search: {e}")