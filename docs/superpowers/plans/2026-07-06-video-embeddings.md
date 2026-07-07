# Video Embedding Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that converts a long video into multimodal (CLIP + Whisper) embeddings stored as parquet, with a FastAPI search interface.

**Architecture:** Sequential pipeline — chunk video → extract frames → transcribe audio → embed visually (CLIP 512-dim) + textually (sentence-transformers 384-dim) → fuse to 896-dim vectors → store as parquet. FastAPI serves `/embed`, `/search`, `/segment` endpoints.

**Tech Stack:** Python 3.11+, ffmpeg (system), openai-whisper, transformers (CLIP), sentence-transformers, pandas/pyarrow, FastAPI, uvicorn, pytest

---

## File Map

```
vedio_token/
├── pyproject.toml
├── cli.py
├── src/
│   └── vedio_token/
│       ├── __init__.py
│       ├── chunker.py          # chunk_video() → List[Segment]
│       ├── extractor.py        # extract_frame() → frame PNG path
│       ├── transcriber.py      # transcribe_segment() → str
│       ├── visual_embedder.py  # embed_frame() → List[float] (512,)
│       ├── text_embedder.py    # embed_text() → List[float] (384,)
│       ├── fuser.py            # fuse_embeddings() → List[float] (896,)
│       ├── store.py            # save_index(), load_index(), cosine_search()
│       └── api.py              # FastAPI app
└── tests/
    ├── conftest.py             # test_video fixture (synthetic 15s video)
    ├── test_chunker.py
    ├── test_extractor.py
    ├── test_transcriber.py
    ├── test_visual_embedder.py
    ├── test_text_embedder.py
    ├── test_fuser.py
    ├── test_store.py
    └── test_api.py
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/vedio_token/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "vedio-token"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai-whisper",
    "transformers",
    "sentence-transformers",
    "pandas",
    "pyarrow",
    "fastapi",
    "uvicorn",
    "httpx",
    "pillow",
    "torch",
    "ftfy",
    "regex",
    "tqdm",
]

[project.scripts]
vedio-token = "cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/vedio_token"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create package init**

```python
# src/vedio_token/__init__.py
```

- [ ] **Step 3: Create tests/conftest.py with synthetic video fixture**

```python
import pytest
import subprocess


@pytest.fixture(scope="session")
def test_video(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("videos")
    video_path = str(out_dir / "test.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", "15",
        "-shortest",
        video_path,
    ], capture_output=True, check=True)
    return video_path
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -e ".[dev]" 2>/dev/null || pip install -e .
pip install pytest
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ tests/conftest.py
git commit -m "init: project scaffold + pyproject.toml"
```

---

## Task 2: Video Chunker

**Files:**
- Create: `src/vedio_token/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
from vedio_token.chunker import chunk_video, Segment


def test_chunk_video_returns_segments(test_video):
    segments = chunk_video(test_video, chunk_duration=5.0, overlap=1.0)
    assert len(segments) > 0
    assert all(isinstance(s, Segment) for s in segments)


def test_chunk_video_segment_fields(test_video):
    segments = chunk_video(test_video, chunk_duration=5.0, overlap=1.0)
    s = segments[0]
    assert s.segment_id == 0
    assert s.start_sec == 0.0
    assert s.end_sec == 5.0
    assert s.video_path == test_video


def test_chunk_video_ids_sequential(test_video):
    segments = chunk_video(test_video, chunk_duration=5.0, overlap=1.0)
    ids = [s.segment_id for s in segments]
    assert ids == list(range(len(ids)))


def test_chunk_video_no_overlap_beyond_duration(test_video):
    segments = chunk_video(test_video, chunk_duration=5.0, overlap=1.0)
    assert all(s.end_sec <= 15.5 for s in segments)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_chunker.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Write chunker implementation**

```python
# src/vedio_token/chunker.py
import json
import subprocess
from dataclasses import dataclass
from typing import List


@dataclass
class Segment:
    segment_id: int
    start_sec: float
    end_sec: float
    video_path: str


def chunk_video(video_path: str, chunk_duration: float = 10.0, overlap: float = 2.0) -> List[Segment]:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True, check=True,
    )
    duration = float(json.loads(result.stdout)["format"]["duration"])

    segments = []
    step = chunk_duration - overlap
    start = 0.0
    seg_id = 0
    while start < duration:
        end = min(start + chunk_duration, duration)
        segments.append(Segment(segment_id=seg_id, start_sec=start, end_sec=end, video_path=video_path))
        seg_id += 1
        if end >= duration:
            break
        start += step
    return segments
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_chunker.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/vedio_token/chunker.py tests/test_chunker.py
git commit -m "feat: video chunker (fixed-interval segments)"
```

---

## Task 3: Frame Extractor

**Files:**
- Create: `src/vedio_token/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extractor.py
import os
from vedio_token.chunker import chunk_video
from vedio_token.extractor import extract_frame


def test_extract_frame_creates_png(test_video, tmp_path):
    segments = chunk_video(test_video, chunk_duration=5.0, overlap=0.0)
    frame_path = extract_frame(segments[0], str(tmp_path / "frames"))
    assert os.path.exists(frame_path)
    assert frame_path.endswith(".png")


def test_extract_frame_returns_path_per_segment(test_video, tmp_path):
    segments = chunk_video(test_video, chunk_duration=5.0, overlap=0.0)
    frames_dir = str(tmp_path / "frames")
    paths = [extract_frame(s, frames_dir) for s in segments]
    assert len(set(paths)) == len(paths)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_extractor.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write extractor implementation**

```python
# src/vedio_token/extractor.py
import subprocess
from pathlib import Path

from .chunker import Segment


def extract_frame(segment: Segment, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    midpoint = (segment.start_sec + segment.end_sec) / 2
    out_path = str(Path(output_dir) / f"frame_{segment.segment_id:04d}.png")
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(midpoint),
        "-i", segment.video_path,
        "-frames:v", "1",
        "-q:v", "2",
        out_path,
    ], capture_output=True, check=True)
    return out_path
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_extractor.py -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/vedio_token/extractor.py tests/test_extractor.py
git commit -m "feat: frame extractor (ffmpeg)"
```

---

## Task 4: Transcriber

**Files:**
- Create: `src/vedio_token/transcriber.py`
- Create: `tests/test_transcriber.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transcriber.py
from vedio_token.chunker import chunk_video
from vedio_token.transcriber import transcribe_segment


def test_transcribe_returns_string(test_video):
    segments = chunk_video(test_video, chunk_duration=5.0, overlap=0.0)
    result = transcribe_segment(segments[0])
    assert isinstance(result, str)


def test_transcribe_does_not_raise_on_silent_segment(test_video):
    segments = chunk_video(test_video, chunk_duration=5.0, overlap=0.0)
    result = transcribe_segment(segments[0])
    assert result is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_transcriber.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write transcriber implementation**

```python
# src/vedio_token/transcriber.py
import logging
import subprocess
import tempfile

import whisper

from .chunker import Segment

_model = None


def _get_model(model_name: str = "base"):
    global _model
    if _model is None:
        _model = whisper.load_model(model_name)
    return _model


def transcribe_segment(segment: Segment, model_name: str = "base") -> str:
    try:
        model = _get_model(model_name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(segment.start_sec),
                "-to", str(segment.end_sec),
                "-i", segment.video_path,
                "-ar", "16000", "-ac", "1",
                tmp.name,
            ], capture_output=True, check=True)
            result = model.transcribe(tmp.name)
            return result["text"].strip()
    except Exception as e:
        logging.warning(f"Whisper failed for segment {segment.segment_id}: {e}")
        return ""
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_transcriber.py -v
```
Expected: 2 PASSED (Whisper downloads `base` model on first run — may take ~30s)

- [ ] **Step 5: Commit**

```bash
git add src/vedio_token/transcriber.py tests/test_transcriber.py
git commit -m "feat: transcriber (Whisper)"
```

---

## Task 5: Visual Embedder (CLIP)

**Files:**
- Create: `src/vedio_token/visual_embedder.py`
- Create: `tests/test_visual_embedder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_visual_embedder.py
from vedio_token.chunker import chunk_video
from vedio_token.extractor import extract_frame
from vedio_token.visual_embedder import embed_frame, VISUAL_DIM


def test_embed_frame_returns_list(test_video, tmp_path):
    segments = chunk_video(test_video, chunk_duration=5.0, overlap=0.0)
    frame_path = extract_frame(segments[0], str(tmp_path / "frames"))
    result = embed_frame(frame_path)
    assert isinstance(result, list)
    assert len(result) == VISUAL_DIM


def test_embed_frame_normalized(test_video, tmp_path):
    import math
    segments = chunk_video(test_video, chunk_duration=5.0, overlap=0.0)
    frame_path = extract_frame(segments[0], str(tmp_path / "frames"))
    result = embed_frame(frame_path)
    norm = math.sqrt(sum(x ** 2 for x in result))
    assert abs(norm - 1.0) < 1e-3


def test_embed_frame_returns_none_on_bad_path():
    result = embed_frame("/nonexistent/frame.png")
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_visual_embedder.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write visual embedder implementation**

```python
# src/vedio_token/visual_embedder.py
import logging
from typing import List, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

VISUAL_DIM = 512
_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _model, _processor


def embed_frame(frame_path: str) -> Optional[List[float]]:
    try:
        model, processor = _get_model()
        image = Image.open(frame_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(_device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()
    except Exception as e:
        logging.warning(f"CLIP failed for {frame_path}: {e}")
        return None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_visual_embedder.py -v
```
Expected: 3 PASSED (CLIP downloads model on first run — may take ~1 min)

- [ ] **Step 5: Commit**

```bash
git add src/vedio_token/visual_embedder.py tests/test_visual_embedder.py
git commit -m "feat: visual embedder (CLIP)"
```

---

## Task 6: Text Embedder

**Files:**
- Create: `src/vedio_token/text_embedder.py`
- Create: `tests/test_text_embedder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_text_embedder.py
import math
from vedio_token.text_embedder import embed_text, TEXT_DIM


def test_embed_text_returns_list():
    result = embed_text("a dog running in a park")
    assert isinstance(result, list)
    assert len(result) == TEXT_DIM


def test_embed_text_normalized():
    result = embed_text("a dog running in a park")
    norm = math.sqrt(sum(x ** 2 for x in result))
    assert abs(norm - 1.0) < 1e-3


def test_embed_text_empty_string():
    result = embed_text("")
    assert isinstance(result, list)
    assert len(result) == TEXT_DIM
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_text_embedder.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write text embedder implementation**

```python
# src/vedio_token/text_embedder.py
from typing import List

from sentence_transformers import SentenceTransformer

TEXT_DIM = 384
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> List[float]:
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_text_embedder.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/vedio_token/text_embedder.py tests/test_text_embedder.py
git commit -m "feat: text embedder (sentence-transformers)"
```

---

## Task 7: Embedding Fuser

**Files:**
- Create: `src/vedio_token/fuser.py`
- Create: `tests/test_fuser.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fuser.py
import math
from vedio_token.fuser import fuse_embeddings, FUSED_DIM, VISUAL_DIM, TEXT_DIM


def test_fuse_returns_correct_dim():
    visual = [0.1] * VISUAL_DIM
    text = [0.2] * TEXT_DIM
    result = fuse_embeddings(visual, text)
    assert len(result) == FUSED_DIM


def test_fuse_is_normalized():
    visual = [1.0] + [0.0] * (VISUAL_DIM - 1)
    text = [1.0] + [0.0] * (TEXT_DIM - 1)
    result = fuse_embeddings(visual, text)
    norm = math.sqrt(sum(x ** 2 for x in result))
    assert abs(norm - 1.0) < 1e-5


def test_fuse_none_visual_uses_zeros():
    text = [0.5] * TEXT_DIM
    result = fuse_embeddings(None, text)
    assert len(result) == FUSED_DIM
    assert result[:VISUAL_DIM] == [0.0] * VISUAL_DIM
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_fuser.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write fuser implementation**

```python
# src/vedio_token/fuser.py
from typing import List, Optional

import numpy as np

from .visual_embedder import VISUAL_DIM
from .text_embedder import TEXT_DIM

FUSED_DIM = VISUAL_DIM + TEXT_DIM  # 896


def fuse_embeddings(visual: Optional[List[float]], text: List[float]) -> List[float]:
    if visual is None:
        visual = [0.0] * VISUAL_DIM
    v = np.array(visual, dtype=np.float32)
    t = np.array(text, dtype=np.float32)
    fused = np.concatenate([v, t])
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused.tolist()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_fuser.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/vedio_token/fuser.py tests/test_fuser.py
git commit -m "feat: embedding fuser"
```

---

## Task 8: Parquet Index Store

**Files:**
- Create: `src/vedio_token/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
import numpy as np
from vedio_token.store import save_index, load_index, cosine_search
from vedio_token.fuser import FUSED_DIM


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_store.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write store implementation**

```python
# src/vedio_token/store.py
from pathlib import Path
from typing import Dict, List, Any

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
    embeddings = np.stack(emb_df["embedding"].values).astype(np.float32)
    scores = embeddings @ q
    top_indices = np.argsort(scores)[::-1][:top_k]
    return emb_df.iloc[top_indices]["segment_id"].tolist()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_store.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/vedio_token/store.py tests/test_store.py
git commit -m "feat: parquet index store"
```

---

## Task 9: FastAPI App + /embed Endpoint

**Files:**
- Create: `src/vedio_token/api.py`
- Create: `tests/test_api.py` (partial — /embed test only)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from vedio_token.api import app

client = TestClient(app)


def test_embed_returns_404_for_missing_video():
    resp = client.post("/embed", json={"video_path": "/nonexistent/video.mp4"})
    assert resp.status_code == 404


def test_embed_processes_real_video(test_video, tmp_path):
    resp = client.post("/embed", json={
        "video_path": test_video,
        "chunk_duration": 5.0,
        "overlap": 1.0,
        "frames_dir": str(tmp_path / "frames"),
        "index_dir": str(tmp_path / "index"),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["segments"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api.py::test_embed_returns_404_for_missing_video -v
```
Expected: `ImportError`

- [ ] **Step 3: Write API app with /embed**

```python
# src/vedio_token/api.py
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


class SearchResponse(BaseModel):
    results: list


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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_api.py::test_embed_returns_404_for_missing_video tests/test_api.py::test_embed_processes_real_video -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/vedio_token/api.py tests/test_api.py
git commit -m "feat: FastAPI app skeleton + /embed endpoint"
```

---

## Task 10: /search Endpoint

**Files:**
- Modify: `tests/test_api.py` (add search tests)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_api.py`

```python
def test_search_returns_404_when_no_index(tmp_path, monkeypatch):
    monkeypatch.setattr("vedio_token.api.DEFAULT_INDEX_DIR", str(tmp_path / "empty_index"))
    resp = client.get("/search", params={"q": "dog", "index_dir": str(tmp_path / "empty_index")})
    assert resp.status_code == 404


def test_search_returns_results(test_video, tmp_path):
    index_dir = str(tmp_path / "idx")
    client.post("/embed", json={
        "video_path": test_video,
        "chunk_duration": 5.0,
        "overlap": 1.0,
        "frames_dir": str(tmp_path / "frames"),
        "index_dir": index_dir,
    })
    resp = client.get("/search", params={"q": "blue screen", "top_k": 2, "index_dir": index_dir})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) <= 2
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_api.py::test_search_returns_404_when_no_index tests/test_api.py::test_search_returns_results -v
```
Expected: FAIL — `/search` not defined

- [ ] **Step 3: Add /search endpoint to api.py**

Add to `src/vedio_token/api.py` after the `/embed` route:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_api.py -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add src/vedio_token/api.py tests/test_api.py
git commit -m "feat: /search endpoint with cosine similarity"
```

---

## Task 11: /segment Endpoint

**Files:**
- Modify: `src/vedio_token/api.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_api.py`

```python
def test_get_segment_returns_metadata(test_video, tmp_path):
    index_dir = str(tmp_path / "idx2")
    client.post("/embed", json={
        "video_path": test_video,
        "chunk_duration": 5.0,
        "overlap": 0.0,
        "frames_dir": str(tmp_path / "frames2"),
        "index_dir": index_dir,
    })
    resp = client.get("/segment/0", params={"index_dir": index_dir})
    assert resp.status_code == 200
    data = resp.json()
    assert data["segment_id"] == 0
    assert "start_sec" in data
    assert "transcript" in data


def test_get_segment_returns_404_for_missing(test_video, tmp_path):
    index_dir = str(tmp_path / "idx3")
    client.post("/embed", json={
        "video_path": test_video,
        "chunk_duration": 5.0,
        "overlap": 0.0,
        "frames_dir": str(tmp_path / "frames3"),
        "index_dir": index_dir,
    })
    resp = client.get("/segment/9999", params={"index_dir": index_dir})
    assert resp.status_code == 404
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_api.py::test_get_segment_returns_metadata tests/test_api.py::test_get_segment_returns_404_for_missing -v
```
Expected: FAIL

- [ ] **Step 3: Add /segment endpoint to api.py**

Add to `src/vedio_token/api.py` after the `/search` route:

```python
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
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add src/vedio_token/api.py tests/test_api.py
git commit -m "feat: /segment endpoint"
```

---

## Task 12: CLI Runner + README

**Files:**
- Create: `cli.py`
- Create: `README.md`

- [ ] **Step 1: Write cli.py**

```python
# cli.py
import argparse

import httpx


def main():
    parser = argparse.ArgumentParser(description="Video embedding CLI")
    parser.add_argument("--host", default="http://localhost:8000")
    sub = parser.add_subparsers(dest="command")

    embed_p = sub.add_parser("embed", help="Embed a video file")
    embed_p.add_argument("video_path")
    embed_p.add_argument("--chunk-duration", type=float, default=10.0)
    embed_p.add_argument("--overlap", type=float, default=2.0)
    embed_p.add_argument("--index-dir", default="index")

    search_p = sub.add_parser("search", help="Search embedded video")
    search_p.add_argument("query")
    search_p.add_argument("--top-k", type=int, default=5)
    search_p.add_argument("--index-dir", default="index")

    args = parser.parse_args()

    if args.command == "embed":
        resp = httpx.post(f"{args.host}/embed", json={
            "video_path": args.video_path,
            "chunk_duration": args.chunk_duration,
            "overlap": args.overlap,
            "index_dir": args.index_dir,
        }, timeout=600)
        print(resp.json())

    elif args.command == "search":
        resp = httpx.get(f"{args.host}/search", params={
            "q": args.query,
            "top_k": args.top_k,
            "index_dir": args.index_dir,
        }, timeout=30)
        for r in resp.json()["results"]:
            print(f"[{r['start_sec']:.1f}s – {r['end_sec']:.1f}s] {r['transcript'][:100]}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write README.md**

```markdown
# vedio-token

Convert a long video into multimodal embeddings (CLIP + Whisper) and search it by text query.

## Requirements

- Python 3.11+
- ffmpeg installed (`brew install ffmpeg` / `apt install ffmpeg`)

## Install

```bash
pip install -e .
```

## Usage

Start the API server:

```bash
uvicorn vedio_token.api:app --reload
```

Embed a video:

```bash
python cli.py embed /path/to/video.mp4 --chunk-duration 10 --overlap 2
```

Search:

```bash
python cli.py search "someone playing guitar"
```

## API

| Method | Path | Description |
|---|---|---|
| POST | `/embed` | Run full pipeline on a video |
| GET | `/search?q=...&top_k=5` | Semantic search |
| GET | `/segment/{id}` | Get segment metadata |
```

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest tests/ -v
```
Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
git add cli.py README.md
git commit -m "chore: CLI runner + README"
```

---

## Final Verification

```bash
# Confirm 12 commits exist
git log --oneline

# Confirm all tests pass
pytest tests/ -v --tb=short
```

Expected git log output:
```
chore: CLI runner + README
feat: /segment endpoint
feat: /search endpoint with cosine similarity
feat: FastAPI app skeleton + /embed endpoint
feat: parquet index store
feat: embedding fuser
feat: text embedder (sentence-transformers)
feat: visual embedder (CLIP)
feat: transcriber (Whisper)
feat: frame extractor (ffmpeg)
feat: video chunker (fixed-interval segments)
init: project scaffold + pyproject.toml
docs: add video embedding pipeline design spec
```
