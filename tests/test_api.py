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
