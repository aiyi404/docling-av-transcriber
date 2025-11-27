from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

KEYFRAME_PREFIX = "[[KEYFRAME_START]]"
KEYFRAME_SUFFIX = "[[KEYFRAME_END]]"


def _format_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "--:--:--.---"
    ms_int = int(ms)
    seconds, millis = divmod(ms_int, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


@dataclass(order=True)
class ConversationWord:
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass(order=True)
class ConversationItem:
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speaker_id: Optional[int] = None
    speaker: Optional[str] = None
    words: list[ConversationWord] = field(default_factory=list)

    def to_string(self) -> str:
        time_tag = None
        if self.start_time is not None and self.end_time is not None:
            time_tag = f"[time: {_format_ms(self.start_time)}-{_format_ms(self.end_time)}]"

        if self.text.startswith(KEYFRAME_PREFIX):
            header, sep, rest = self.text.partition("\n")
            header_body = header[len(KEYFRAME_PREFIX) :].lstrip()
            new_header_parts = [KEYFRAME_PREFIX]
            if time_tag:
                new_header_parts.append(time_tag)
            if header_body:
                new_header_parts.append(header_body)
            new_header = " ".join(new_header_parts)
            rebuilt = new_header
            if sep:
                rebuilt = f"{rebuilt}\n{rest}"
            return rebuilt

        chunks: list[str] = []
        if time_tag:
            chunks.append(time_tag)
        if self.speaker is not None:
            chunks.append(f"[speaker:{self.speaker}]")
        chunks.append(self.text)
        return " ".join(chunks)


class AsrServiceError(RuntimeError):
    pass


class VisionServiceError(RuntimeError):
    pass
