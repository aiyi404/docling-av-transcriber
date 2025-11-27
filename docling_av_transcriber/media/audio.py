from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Union

# 设置日志记录器
logger = logging.getLogger(__name__)


class NoAudioStreamError(RuntimeError):
    """Raised when the input media does not contain any audio stream."""


_NO_AUDIO_ERROR_MARKERS = (
    "Output file does not contain any stream",
    "Stream specifier ':a'",
    "matches no streams",
)


def ensure_wav_audio(path: Union[str, Path]) -> Path:
    """Ensure the given file is a wav audio; convert via ffmpeg when necessary."""
    source = Path(path)
    logger.info(f"Ensuring WAV audio for file: {source}")
    logger.debug(f"File extension: {source.suffix.lower()}")

    if source.suffix.lower() == ".wav":
        logger.info("File is already WAV format, no conversion needed")
        return source

    logger.info("Converting file to WAV format using ffmpeg")
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    logger.debug(f"Temporary WAV file: {tmp}")

    cmd = [
        "ffmpeg",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(tmp),
        "-y",
    ]
    logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    process = subprocess.run(cmd, capture_output=True, text=True, check=False)
    logger.debug(f"FFmpeg return code: {process.returncode}")
    logger.debug(f"FFmpeg stdout: {process.stdout}")
    logger.debug(f"FFmpeg stderr: {process.stderr}")

    if process.returncode != 0:
        logger.error(f"ffmpeg conversion failed with return code {process.returncode}")
        logger.error(f"ffmpeg stderr: {process.stderr}")
        tmp.unlink(missing_ok=True)

        stderr_lower = process.stderr.lower()
        if any(marker.lower() in stderr_lower for marker in _NO_AUDIO_ERROR_MARKERS):
            raise NoAudioStreamError(f"No audio stream found in file: {source}")

        raise RuntimeError(f"ffmpeg conversion failed: {process.stderr}")

    logger.info("Successfully converted file to WAV format")
    return tmp
