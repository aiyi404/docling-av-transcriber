from io import BytesIO
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from docling_av_transcriber.models.base import SpeechToTextProvider
from docling_av_transcriber.pipelines.asr_pipeline import AsrPipelineLite
from docling_av_transcriber.types import ConversationItem


class FakeProvider(SpeechToTextProvider):
    def transcribe_path(self, audio_path: Path, *, language: str):
        return [ConversationItem(text=f"file:{audio_path.name}")]

    def transcribe_bytes(self, data: bytes, *, language: str, filename: str):
        return [ConversationItem(text=f"bytes:{filename}")]


def test_transcribe_file(tmp_path: Path):
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"fake")
    pipeline = AsrPipelineLite(provider=FakeProvider())
    doc = pipeline.transcribe(wav, language="zh")
    assert isinstance(doc, DoclingDocument)
    assert "file:sample.wav" in doc.export_to_markdown()


def test_transcribe_bytes():
    pipeline = AsrPipelineLite(provider=FakeProvider())
    doc = pipeline.transcribe(BytesIO(b"abc"), filename="sample.wav", language="zh")
    assert "bytes:sample.wav" in doc.export_to_markdown()
