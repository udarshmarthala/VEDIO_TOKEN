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

## VS Code Extension

`vscode-extension/` provides IDE integration with the video embedding API.

### Features

| Feature | How |
|---------|-----|
| Embed video | Right-click `.mp4`/`.mov`/`.mkv`/`.avi`/`.webm` in Explorer → **Vedio Token: Embed Video** |
| Search index | Command Palette → **Vedio Token: Search Video** |
| Browse segments | Activity Bar → **Vedio Token** sidebar |
| Segment detail | Click any segment in the sidebar |
| Start/stop server | Command Palette → **Vedio Token: Start API Server** / **Stop API Server** |
| API health | Status bar item (bottom-right) — green check when server is up, warning when down |

### Install (development)

```bash
cd vscode-extension
npm install
npm run build
```

Open the project folder in VS Code and press **F5** to launch the Extension Development Host.

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `vedio-token.apiUrl` | `http://localhost:8000` | FastAPI server URL |
| `vedio-token.indexDir` | `index` | Embedding index directory |
| `vedio-token.chunkDuration` | `10.0` | Chunk size in seconds |
| `vedio-token.overlap` | `2.0` | Overlap between chunks in seconds |
| `vedio-token.pythonPath` | `python` | Python executable for server auto-start |
