from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .chunker import chunk_video
from .extractor import extract_frame
from .fuser import fuse_embeddings, FUSED_DIM, VISUAL_DIM
from .store import save_index, load_index, cosine_search
from .text_embedder import embed_text
from .transcriber import transcribe_segment
from .visual_embedder import embed_frame

app = FastAPI(title="Video Embedding API")

DEFAULT_INDEX_DIR = "index"


class EmbedRequest(BaseModel):
    video_path: str
    chunk_duration: float = 10.0
    overlap: float = 2.0
    frames_dir: str = "frames"
    index_dir: str = DEFAULT_INDEX_DIR


@app.post("/embed")
def embed_video(req: EmbedRequest):
    if not Path(req.video_path).exists():
        raise HTTPException(status_code=404, detail="Video not found")

    segments = chunk_video(req.video_path, req.chunk_duration, req.overlap)
    segments_meta = []
    embeddings = []

    for seg in segments:
        frame_path = extract_frame(seg, req.frames_dir)
        transcript = transcribe_segment(seg)
        visual_vec = embed_frame(frame_path)
        text_vec = embed_text(transcript if transcript else "")
        fused = fuse_embeddings(visual_vec, text_vec)
        segments_meta.append({
            "segment_id": seg.segment_id,
            "start_sec": seg.start_sec,
            "end_sec": seg.end_sec,
            "frame_path": frame_path,
            "transcript": transcript,
            "video_path": seg.video_path,
        })
        embeddings.append(fused)

    save_index(segments_meta, embeddings, req.index_dir)
    return {"status": "ok", "segments": len(segments)}


@app.get("/search")
def search(q: str, top_k: int = 5, index_dir: str = DEFAULT_INDEX_DIR):
    index_path = Path(index_dir)
    if not (index_path / "metadata.parquet").exists():
        raise HTTPException(status_code=404, detail="No index found. Run /embed first.")

    meta_df, emb_df = load_index(index_dir)

    text_vec = embed_text(q)
    query_vec = np.zeros(FUSED_DIM, dtype=np.float32)
    query_vec[VISUAL_DIM:] = np.array(text_vec, dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    segment_ids = cosine_search(query_vec.tolist(), emb_df, top_k)
    results = meta_df[meta_df["segment_id"].isin(segment_ids)].to_dict(orient="records")
    return {"results": results}


@app.get("/segment/{segment_id}")
def get_segment(segment_id: int, index_dir: str = DEFAULT_INDEX_DIR):
    index_path = Path(index_dir)
    if not (index_path / "metadata.parquet").exists():
        raise HTTPException(status_code=404, detail="No index found.")

    meta_df, _ = load_index(index_dir)
    row = meta_df[meta_df["segment_id"] == segment_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found.")
    return row.iloc[0].to_dict()
