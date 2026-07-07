# Video Embedding Pipeline — Design Spec

**Date:** 2026-07-06  
**Status:** Approved

## Goal

Convert a single long video into multimodal embeddings (visual + speech) and expose semantic search via FastAPI. Query: text → top-K matching video segments.

---

## Architecture

```
video.mp4
    │
    ├─ VideoChunker      → segments (start_sec, end_sec)
    ├─ FrameExtractor    → keyframe PNG per segment (ffmpeg)
    ├─ Transcriber       → Whisper → transcript per segment
    ├─ VisualEmbedder    → CLIP (openai/clip-vit-base-patch32) → (N, 512)
    ├─ TextEmbedder      → sentence-transformers (all-MiniLM-L6-v2) → (N, 384)
    ├─ EmbeddingFuser    → concat + L2-normalize → (N, 896)
    ├─ IndexStore        → embeddings.parquet + metadata.parquet
    │
    └─ FastAPI app
            POST /embed        → run full pipeline on video path
            GET  /search?q=    → top-K segments
            GET  /segment/{id} → metadata + frame path
```

---

## Data Model

**metadata.parquet**
| segment_id | start_sec | end_sec | frame_path | transcript | video_path |
|---|---|---|---|---|---|

**embeddings.parquet**
| segment_id | embedding (896-dim float32) |
|---|---|

---

## Pipeline Parameters

| Param | Default | Notes |
|---|---|---|
| `chunk_duration` | 10s | Fixed-interval segment length |
| `overlap` | 2s | Sliding window overlap |
| `frame_sample` | midpoint | 1 frame per segment |
| `whisper_model` | `base` | Upgradeable to `small`/`medium` |
| `top_k` | 5 | Search results returned |

---

## Search Flow

```
query string
  → sentence-transformers → (384,)
  → zero-pad to (896,)
  → cosine similarity vs all embeddings rows
  → top-K segment_ids
  → join metadata
  → return JSON
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| ffmpeg not installed | `RuntimeError` at startup |
| Whisper fails on segment | Log warning, store empty transcript, continue |
| CLIP fails on frame | Skip segment, log warning |
| Search on empty index | HTTP 404 with message |

---

## Stack

```
python        3.11+
ffmpeg        system binary
openai-whisper
transformers  (CLIP)
sentence-transformers
pandas + pyarrow
fastapi + uvicorn
pytest
```

---

## 12-Commit Plan

| # | Commit |
|---|---|
| 1 | `init: project scaffold + pyproject.toml` |
| 2 | `feat: video chunker (fixed-interval segments)` |
| 3 | `feat: frame extractor (ffmpeg)` |
| 4 | `feat: transcriber (Whisper)` |
| 5 | `feat: visual embedder (CLIP)` |
| 6 | `feat: text embedder (sentence-transformers)` |
| 7 | `feat: embedding fuser` |
| 8 | `feat: parquet index store` |
| 9 | `feat: FastAPI app skeleton + /embed endpoint` |
| 10 | `feat: /search endpoint with cosine similarity` |
| 11 | `feat: /segment endpoint` |
| 12 | `chore: CLI runner + README` |

---

## Testing Strategy

- Each module: one test file, minimal fixture (3s synthetic video or mock frames)
- Integration tests hit real ffmpeg + Whisper (no mocking)
- FastAPI: `TestClient` for all endpoints
- Runner: `pytest`
