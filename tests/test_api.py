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


def test_search_returns_404_when_no_index(tmp_path):
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
