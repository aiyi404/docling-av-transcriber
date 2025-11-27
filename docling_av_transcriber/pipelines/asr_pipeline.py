from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from docling_core.types.doc import DoclingDocument

from docling_av_transcriber.media import (
    ensure_wav_audio,
    extract_keyframes_with_timestamps,
    validate_input,
    NoAudioStreamError,
)
from docling_av_transcriber.media.backend import BackendResult
from docling_av_transcriber.models import (
    AliyunBailianAsrClient,
    AliyunVisionClient,
    ImageCaptionProvider,
    SpeechToTextProvider,
)
from docling_av_transcriber.pipelines.document_builder import build_docling_document
from docling_av_transcriber.types import ConversationItem, VisionServiceError

# 设置日志记录器
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


@dataclass
class TranscriptionResult:
    """Transcription artifacts returned by ``transcribe_with_artifacts``."""

    document: DoclingDocument
    audio_path: Path | None


class AsrPipelineLite:
    """Lightweight pipeline orchestrating ASR provider and Docling document building."""

    def __init__(
        self,
        provider: Optional[SpeechToTextProvider] = None,
        vision_provider: Optional[ImageCaptionProvider] = None,
    ) -> None:
        self.provider = provider or AliyunBailianAsrClient()
        if vision_provider is not None:
            self.vision_provider = vision_provider
        else:
            self.vision_provider = self._init_default_vision_provider()

    def transcribe(
        self,
        path_or_stream: Union[str, Path, BytesIO],
        *,
        filename: Optional[str] = None,
        language: str = "zh",
        summary: Optional[str] = None,
    ) -> DoclingDocument:
        """Backward compatible API returning only the Docling document."""
        result = self.transcribe_with_artifacts(
            path_or_stream, filename=filename, language=language, summary=summary
        )
        return result.document

    def transcribe_with_artifacts(
        self,
        path_or_stream: Union[str, Path, BytesIO],
        *,
        filename: Optional[str] = None,
        language: str = "zh",
        summary: Optional[str] = None,
    ) -> TranscriptionResult:
        logger.info("Starting transcription pipeline")
        backend_result = validate_input(path_or_stream, filename)
        logger.info(f"Input validated: {backend_result.filename}")
        binary_hash = self._compute_binary_hash(backend_result)

        wav_path = self._prepare_wav_artifact(backend_result)

        conversation: list[ConversationItem]
        if wav_path is None:
            logger.info(
                "No audio stream detected for %s; skipping speech-to-text stage.",
                backend_result.filename,
            )
            conversation = []
        elif isinstance(backend_result.path_or_stream, Path):
            logger.info(f"Processing file: {backend_result.path_or_stream}")
            conversation = self.provider.transcribe_path(wav_path, language=language)
        else:
            with wav_path.open("rb") as handle:
                wav_bytes = handle.read()
            logger.debug(f"Stream data size after conversion: {len(wav_bytes)} bytes")
            conversation = self.provider.transcribe_bytes(
                wav_bytes, filename=backend_result.filename, language=language
            )

        logger.info(f"Transcription completed with {len(conversation)} items")
        is_video_file = self._is_video_file(backend_result.filename)
        if self._should_use_visual_fallback(is_video_file, conversation):
            logger.info(
                "Audio transcription is empty for video input; attempting keyframe-based analysis."
            )
            visual_items = self._describe_video_frames(backend_result)
            if visual_items:
                conversation.extend(visual_items)
                logger.info(
                    "Added %d visual items generated from keyframes", len(visual_items)
                )
        mimetype = "video/mp4" if is_video_file else "audio/wav"

        logger.info("Building Docling document")
        doc = build_docling_document(
            filename=backend_result.filename,
            mimetype=mimetype,
            binary_hash=binary_hash,
            conversation=conversation,
            summary=summary,
        )
        logger.info("Document building completed")
        return TranscriptionResult(document=doc, audio_path=wav_path)

    def _compute_binary_hash(self, backend_result: BackendResult) -> str:
        hasher = hashlib.sha256()
        try:
            if isinstance(backend_result.path_or_stream, Path):
                with backend_result.path_or_stream.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        if not chunk:
                            break
                        hasher.update(chunk)
            else:
                buffer = backend_result.path_or_stream.getbuffer()
                hasher.update(buffer)
            return hasher.hexdigest()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "Failed to compute binary hash for %s: %s",
                backend_result.filename,
                exc,
            )
            return hashlib.sha256(
                backend_result.filename.encode("utf-8", "ignore")
            ).hexdigest()

    def _init_default_vision_provider(self) -> ImageCaptionProvider | None:
        try:
            return AliyunVisionClient()
        except VisionServiceError as exc:
            logger.info("Vision provider disabled: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Unexpected error initializing vision provider: %s", exc)
        return None

    def _should_use_visual_fallback(
        self, is_video_file: bool, conversation: list[ConversationItem]
    ) -> bool:
        return is_video_file and not conversation and self.vision_provider is not None

    def _describe_video_frames(self, backend_result: BackendResult) -> list[ConversationItem]:
        if self.vision_provider is None:
            logger.debug("Vision provider is not configured; skipping keyframe analysis")
            return []

        temp_source: Path | None = None
        try:
            if isinstance(backend_result.path_or_stream, Path):
                video_path = backend_result.path_or_stream
            else:
                data = backend_result.path_or_stream.getvalue()
                temp_source = self._write_temp_file(data, backend_result.filename)
                video_path = temp_source

            frames = extract_keyframes_with_timestamps(video_path)
            if not frames:
                logger.info("No keyframes extracted for video input: %s", backend_result.filename)
                return []

            logger.info("Describing %d keyframes via vision provider", len(frames))
            return self.vision_provider.describe_frames(frames)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Keyframe-based analysis failed for %s: %s", backend_result.filename, exc)
            return []
        finally:
            if temp_source is not None:
                temp_source.unlink(missing_ok=True)

    @staticmethod
    def _is_video_file(filename: str | None) -> bool:
        if not filename:
            return False
        return filename.lower().endswith(VIDEO_EXTENSIONS)

    @staticmethod
    def _write_temp_file(data: bytes, filename: str) -> Path:
        suffix = Path(filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            temp_path = Path(tmp.name)
        logger.debug(f"Wrote temporary file for BytesIO input: {temp_path}")
        return temp_path

    def _prepare_wav_artifact(self, backend_result: BackendResult) -> Path | None:
        """Normalize the input into a WAV file and return its path when available."""
        source_path: Path
        temp_source: Path | None = None
        wav_path: Path | None = None

        if isinstance(backend_result.path_or_stream, Path):
            source_path = backend_result.path_or_stream
            logger.info(f"Processing file: {source_path}")
        else:
            data = backend_result.path_or_stream.getvalue()
            temp_source = self._write_temp_file(data, backend_result.filename)
            source_path = temp_source

        try:
            wav_path = ensure_wav_audio(source_path)
            logger.info(f"Audio converted to WAV: {wav_path}")
            return wav_path
        except NoAudioStreamError:
            logger.info("File %s has no audio stream", backend_result.filename)
            return None
        finally:
            if temp_source is not None and (wav_path is None or wav_path != temp_source):
                logger.debug(f"Cleaning up intermediate temp file: {temp_source}")
                temp_source.unlink(missing_ok=True)
