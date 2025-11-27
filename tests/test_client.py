import json
from pathlib import Path

import pytest
import responses

from docling_av_transcriber.config import AliyunBailianSettings
from docling_av_transcriber.models.aliyun_bailian import AliyunBailianAsrClient
from docling_av_transcriber.types import ConversationItem


@pytest.fixture()
def aliyun_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AliyunBailianAsrClient:
    settings = AliyunBailianSettings(api_key="test-key")
    return AliyunBailianAsrClient(settings)


def sample_response() -> dict:
    return {
        "code": "Success",
        "data": {
            "segments": [
                {
                    "text": "你好，世界",
                    "start": 0.0,
                    "end": 1.2,
                    "words": [
                        {"word": "你好", "start": 0.0, "end": 0.6},
                        {"word": "世界", "start": 0.6, "end": 1.2},
                    ],
                }
            ]
        },
    }


@responses.activate
def test_transcribe_path(tmp_path: Path, aliyun_client: AliyunBailianAsrClient) -> None:
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"fake-audio")
    responses.post(
        aliyun_client.settings.endpoint,
        json=sample_response(),
        status=200,
    )

    items = aliyun_client.transcribe_path(audio_file, language="zh")
    assert isinstance(items, list)
    assert isinstance(items[0], ConversationItem)
    assert items[0].text == "你好，世界"


@responses.activate
def test_transcribe_bytes_error(aliyun_client: AliyunBailianAsrClient) -> None:
    responses.post(
        aliyun_client.settings.endpoint,
        json={"code": "Failure", "request_id": "abc"},
        status=200,
    )
    with pytest.raises(Exception):
        aliyun_client.transcribe_bytes(b"", language="zh", filename="test.wav")
