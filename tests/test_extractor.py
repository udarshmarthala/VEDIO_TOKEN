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
