from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# 匹配 ffmpeg showinfo 输出中的 pts_time
_SHOWINFO_RE = re.compile(r"pts_time:([0-9.]+)")


def _run_ffmpeg_extract(
    source: Path, output_pattern: Path, vf_expr: str
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ffmpeg",
        "-i",
        str(source),
        "-vf",
        vf_expr,
        "-vsync",
        "vfr",
        "-q:v",
        "2",
        str(output_pattern),
        "-y",
    ]
    logger.debug("FFmpeg command: %s", " ".join(cmd))
    process = subprocess.run(cmd, capture_output=True, text=True, check=False)
    logger.debug("FFmpeg return code: %s", process.returncode)
    logger.debug("FFmpeg stdout: %s", process.stdout)
    logger.debug("FFmpeg stderr: %s", process.stderr)

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg keyframe extraction failed: {process.stderr}")
    return process


def _parse_pts(stderr: str) -> list[float]:
    timestamps: list[float] = []
    for line in stderr.splitlines():
        match = _SHOWINFO_RE.search(line)
        if match:
            try:
                timestamps.append(float(match.group(1)))
            except ValueError:
                logger.debug("Failed to parse pts_time from line: %s", line)
    return timestamps


def _cleanup_frames(tmp_dir: Path) -> None:
    for frame_file in tmp_dir.glob("frame_*.jpg"):
        frame_file.unlink(missing_ok=True)


def extract_keyframes_with_timestamps(
    path: Union[str, Path],
    *,
    max_frames: int = 16,
    scene_threshold: float = 0.3,
) -> list[tuple[Path, float]]:
    """使用 ffmpeg 抽取关键帧，并返回 (帧路径, 时间戳秒) 列表。

    - max_frames: 限制最大帧数，避免成本过高
    - scene_threshold: 场景变化阈值，越大越“严格”
    """
    source = Path(path)
    logger.info("Extracting keyframes with timestamps for file: %s", source)

    if not source.exists():
        logger.error("Video file not found: %s", source)
        raise FileNotFoundError(source)

    tmp_dir = Path(tempfile.mkdtemp(prefix="keyframes_"))
    output_pattern = tmp_dir / "frame_%04d.jpg"

    scene_vf = f"select='gt(scene,{scene_threshold})+eq(n,0)',showinfo"
    try:
        proc = _run_ffmpeg_extract(source, output_pattern, scene_vf)
    except RuntimeError as exc:
        logger.warning(
            "Scene-based keyframe extraction failed (%s); falling back to uniform sampling.",
            exc,
        )
        _cleanup_frames(tmp_dir)
        proc = _run_ffmpeg_extract(source, output_pattern, "fps=1,showinfo")
    timestamps = _parse_pts(proc.stderr)

    frames = sorted(tmp_dir.glob("frame_*.jpg"))

    if not frames:
        logger.warning("Scene detection produced no frames; using uniform sampling fallback.")
        _cleanup_frames(tmp_dir)
        proc = _run_ffmpeg_extract(source, output_pattern, "fps=1,showinfo")
        timestamps = _parse_pts(proc.stderr)
        frames = sorted(tmp_dir.glob("frame_*.jpg"))

    if not frames:
        logger.info("Uniform sampling fallback also produced no frames for %s", source)
        return []

    # 在正常情况下，showinfo 的顺序与输出帧顺序一致，直接 zip 对齐
    pairs = list(zip(frames, timestamps))

    # 可能出现帧数和时间戳数不一致的情况，这里做一次保护
    if len(frames) != len(timestamps):
        logger.warning(
            "Number of frames (%d) and timestamps (%d) does not match, truncating to min.",
            len(frames),
            len(timestamps),
        )
        n = min(len(frames), len(timestamps))
        pairs = list(zip(frames[:n], timestamps[:n]))

    if len(pairs) > max_frames:
        # 简单下采样，保持顺序（KISS）
        step = max(1, len(pairs) // max_frames)
        pairs = pairs[::step][:max_frames]

    logger.info("Extracted %d keyframes", len(pairs))
    return pairs
