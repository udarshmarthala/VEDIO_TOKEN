import subprocess
from pathlib import Path

from .chunker import Segment


def extract_frame(segment: Segment, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    midpoint = (segment.start_sec + segment.end_sec) / 2
    out_path = str(Path(output_dir) / f"frame_{segment.segment_id:04d}.png")
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(midpoint),
        "-i", segment.video_path,
        "-frames:v", "1",
        "-q:v", "2",
        out_path,
    ], capture_output=True, check=True)
    return out_path
