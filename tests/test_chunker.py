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
