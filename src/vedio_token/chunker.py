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
