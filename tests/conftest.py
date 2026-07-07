import pytest
import subprocess


@pytest.fixture(scope="session")
def test_video(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("videos")
    video_path = str(out_dir / "test.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", "15",
        "-shortest",
        video_path,
    ], capture_output=True, check=True)
    return video_path
