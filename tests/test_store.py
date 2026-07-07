import numpy as np
from vedio_token.store import save_index, load_index, cosine_search

FUSED_DIM = 896


def _make_meta_and_embeddings(n=3):
    meta = [
        {"segment_id": i, "start_sec": float(i * 10), "end_sec": float(i * 10 + 10),
         "frame_path": f"frame_{i}.png", "transcript": f"text {i}", "video_path": "v.mp4"}
        for i in range(n)
    ]
    vecs = [np.random.rand(FUSED_DIM).astype(np.float32).tolist() for _ in range(n)]
    return meta, vecs


def test_save_and_load_index(tmp_path):
    meta, vecs = _make_meta_and_embeddings()
    save_index(meta, vecs, str(tmp_path))
    meta_df, emb_df = load_index(str(tmp_path))
    assert len(meta_df) == 3
    assert len(emb_df) == 3
    assert list(meta_df.columns) == ["segment_id", "start_sec", "end_sec", "frame_path", "transcript", "video_path"]


def test_cosine_search_returns_top_k(tmp_path):
    meta, vecs = _make_meta_and_embeddings(5)
    save_index(meta, vecs, str(tmp_path))
    _, emb_df = load_index(str(tmp_path))
    query = np.random.rand(FUSED_DIM).astype(np.float32).tolist()
    results = cosine_search(query, emb_df, top_k=3)
    assert len(results) == 3


def test_cosine_search_exact_match(tmp_path):
    meta, vecs = _make_meta_and_embeddings(3)
    target_vec = np.ones(FUSED_DIM, dtype=np.float32)
    target_vec /= np.linalg.norm(target_vec)
    vecs[1] = target_vec.tolist()
    save_index(meta, vecs, str(tmp_path))
    _, emb_df = load_index(str(tmp_path))
    results = cosine_search(target_vec.tolist(), emb_df, top_k=1)
    assert results[0] == 1
