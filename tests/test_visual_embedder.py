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
