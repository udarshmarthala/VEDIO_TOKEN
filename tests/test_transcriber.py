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
