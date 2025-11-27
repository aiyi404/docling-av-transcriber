from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Sequence

from docling_av_transcriber.types import ConversationItem


class SpeechToTextProvider(ABC):
    """Abstract speech-to-text provider."""

    @abstractmethod
    def transcribe_path(self, audio_path: Path, *, language: str) -> list[ConversationItem]:
        raise NotImplementedError

    @abstractmethod
    def transcribe_bytes(self, data: bytes, *, language: str, filename: str) -> list[ConversationItem]:
        raise NotImplementedError

    def transcribe_stream(self, stream: BinaryIO, *, language: str, filename: str) -> list[ConversationItem]:
        return self.transcribe_bytes(stream.read(), language=language, filename=filename)


class ImageCaptionProvider(ABC):
    """Abstract provider describing visual keyframes."""

    @abstractmethod
    def describe_frames(self, frames: Sequence[tuple[Path, float]]) -> list[ConversationItem]:
        """Return textual descriptions for ``(Path, timestamp_sec)`` frames."""
        raise NotImplementedError
