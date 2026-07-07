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
    import os
    tmp_path = None
    try:
        model = _get_model(model_name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(segment.start_sec),
            "-to", str(segment.end_sec),
            "-i", segment.video_path,
            "-ar", "16000", "-ac", "1",
            tmp_path,
        ], capture_output=True, check=True)
        result = model.transcribe(tmp_path)
        return result["text"].strip()
    except Exception as e:
        logging.warning(f"Whisper failed for segment {segment.segment_id}: {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
