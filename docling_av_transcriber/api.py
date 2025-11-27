from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Union

from docling_core.types.doc import DoclingDocument

from docling_av_transcriber.pipelines.asr_pipeline import AsrPipelineLite, TranscriptionResult

# 设置日志记录器
logger = logging.getLogger(__name__)


_pipeline_singleton: AsrPipelineLite | None = None


def _get_pipeline() -> AsrPipelineLite:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = AsrPipelineLite()
    return _pipeline_singleton


def transcribe_file(path: Union[str, Path], *, language: str = "zh") -> DoclingDocument:
    logger.info(f"Transcribing file: {path}")
    pipeline = _get_pipeline()
    result = pipeline.transcribe(path, language=language)
    logger.info("File transcription completed")
    return result


def transcribe_bytes(data: bytes, *, filename: str = "audio.wav", language: str = "zh") -> DoclingDocument:
    pipeline = _get_pipeline()
    return pipeline.transcribe(BytesIO(data), filename=filename, language=language)


def transcribe_file_with_artifacts(
    path: Union[str, Path], *, language: str = "zh"
) -> TranscriptionResult:
    """Return both the Docling document and extracted WAV audio path."""
    pipeline = _get_pipeline()
    return pipeline.transcribe_with_artifacts(path, language=language)


def transcribe_bytes_with_artifacts(
    data: bytes, *, filename: str = "audio.wav", language: str = "zh"
) -> TranscriptionResult:
    """Byte-stream variant that also provides the extracted WAV path."""
    pipeline = _get_pipeline()
    return pipeline.transcribe_with_artifacts(BytesIO(data), filename=filename, language=language)
