from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def save_index(segments_meta: List[Dict[str, Any]], embeddings: List[List[float]], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    meta_df = pd.DataFrame(segments_meta)
    emb_df = pd.DataFrame({
        "segment_id": [m["segment_id"] for m in segments_meta],
        "embedding": embeddings,
    })
    meta_df.to_parquet(Path(output_dir) / "metadata.parquet", index=False)
    emb_df.to_parquet(Path(output_dir) / "embeddings.parquet", index=False)


def load_index(index_dir: str):
    meta_df = pd.read_parquet(Path(index_dir) / "metadata.parquet")
    emb_df = pd.read_parquet(Path(index_dir) / "embeddings.parquet")
    return meta_df, emb_df


def cosine_search(query_vec: List[float], emb_df: pd.DataFrame, top_k: int = 5) -> List[int]:
    q = np.array(query_vec, dtype=np.float32)
    q = q / np.linalg.norm(q)
    embeddings = np.stack(emb_df["embedding"].values).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = embeddings @ q
    top_indices = np.argsort(scores)[::-1][:top_k]
    return emb_df.iloc[top_indices]["segment_id"].tolist()
