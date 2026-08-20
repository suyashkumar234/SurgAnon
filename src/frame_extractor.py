import subprocess
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, fps: int = 1) -> list[str]:
    """Extract frames from video at given fps using ffmpeg. Returns sorted list of frame paths."""
    video_path = Path(video_path)
    out_dir = Path(output_dir) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_pattern = str(out_dir / "frame_%06d.jpg")

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        "-y",
        frame_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path.name}:\n{result.stderr}")

    frames = sorted(out_dir.glob("frame_*.jpg"))
    return [str(f) for f in frames]


def get_frame_timestamps(frames: list[str], fps: int = 1) -> list[float]:
    return [i / fps for i in range(len(frames))]
