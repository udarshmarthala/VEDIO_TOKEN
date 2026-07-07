# vedio-token

Convert a long video into multimodal embeddings (CLIP + Whisper) and search it by text query.

## Requirements

- Python 3.11+
- ffmpeg (`brew install ffmpeg` / `apt install ffmpeg`)

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

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/embed` | Run full pipeline on a video |
| GET | `/search?q=...&top_k=5` | Semantic search |
| GET | `/segment/{id}` | Get segment metadata |
