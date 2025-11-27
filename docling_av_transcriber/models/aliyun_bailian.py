from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import requests
from http import HTTPStatus

from docling_av_transcriber.config import AliyunBailianSettings
from docling_av_transcriber.models.base import SpeechToTextProvider
from docling_av_transcriber.types import AsrServiceError, ConversationItem, ConversationWord

# 设置日志记录器
logger = logging.getLogger(__name__)


class AliyunBailianAsrClient(SpeechToTextProvider):
    """Aliyun Bailian ASR client wrapping DashScope compatible endpoints."""

    def __init__(self, settings: AliyunBailianSettings | None = None) -> None:
        self.settings = settings or AliyunBailianSettings.from_env()
        if not self.settings.api_key:
            raise AsrServiceError(
                "Aliyun Bailian API key is missing. Set ALIYUN_BAILIAN_API_KEY or pass settings explicitly."
            )

    # Public API -----------------------------------------------------------
    def transcribe_path(self, audio_path: Path, *, language: str | None = None) -> list[ConversationItem]:
        with audio_path.open("rb") as handle:
            return self._transcribe(handle.read(), filename=audio_path.name, language=language)

    def transcribe_bytes(
        self,
        data: bytes,
        *,
        language: str | None = None,
        filename: str = "audio.wav",
        file_urls: Sequence[str] | None = None,
    ) -> list[ConversationItem]:
        return self._transcribe(data, filename=filename, language=language, file_urls=file_urls)

    def transcribe_remote_urls(
        self,
        file_urls: Sequence[str],
        *,
        language: str | None = None,
    ) -> list[ConversationItem]:
        if not file_urls:
            raise AsrServiceError("file_urls must not be empty")
        # 尽量提取一个可读文件名，方便日志追踪
        first_name = Path(file_urls[0]).name or file_urls[0]
        return self._transcribe(data=None, filename=first_name, language=language, file_urls=file_urls)

    # Internal helpers -----------------------------------------------------
    def _transcribe(
        self,
        data: bytes | None,
        *,
        filename: str,
        language: str | None,
        file_urls: Sequence[str] | None = None,
    ) -> list[ConversationItem]:
        logger.info(f"Starting transcription for file: {filename}")
        if data is not None:
            logger.debug(f"Data size: {len(data)} bytes")
        if file_urls:
            logger.debug(f"Remote file URLs: {file_urls}")
            try:
                return self._transcribe_with_sdk(file_urls=file_urls, language=language)
            except ImportError as exc:
                msg = "DashScope SDK is required for remote URL transcription."
                logger.error(msg)
                raise AsrServiceError(msg) from exc
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("DashScope SDK transcription failed")
                raise AsrServiceError(f"DashScope SDK transcription failed: {exc}") from exc

        if data is None:
            raise AsrServiceError(
                "Audio bytes must be provided when file_urls are not supplied. "
                "If you already uploaded the file, call transcribe_remote_urls(file_urls=[...])."
            )

        # 优先尝试 SDK 上传+异步调用，失败才回退直连
        try:
            return self._transcribe_with_sdk_upload(data=data, filename=filename, language=language)
        except ImportError:
            logger.info("DashScope SDK not available; falling back to direct API transcription.")
        except AsrServiceError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(f"DashScope SDK upload path failed, falling back to direct API: {exc}")

        logger.info("Using direct API transcription fallback")
        return self._transcribe_direct(data, filename, language)

    def _transcribe_with_sdk(self, *, file_urls: Sequence[str], language: str | None) -> list[ConversationItem]:
        """Use DashScope SDK async workflow for already uploaded audio."""
        from dashscope.audio.asr import Transcription  # type: ignore[import-not-found]
        import dashscope  # type: ignore[import-not-found]

        dashscope.base_http_api_url = self.settings.base_http_api_url
        dashscope.api_key = self.settings.api_key

        parameters = {
            "language": language or self.settings.language,
            "enable_words": self.settings.enable_words,
            "enable_diarization": self.settings.diarization,
        }

        logger.info(f"Submitting DashScope async transcription request for {len(file_urls)} file(s)")
        task_response = Transcription.async_call(
            model=self.settings.model,
            file_urls=list(file_urls),
            parameters=parameters,
        )

                # 详细记录响应信息以便调试
        logger.debug(f"DashScope task_response type: {type(task_response)}")
        logger.debug(f"DashScope task_response: {task_response}")
        logger.debug(f"DashScope task_response.status_code: {getattr(task_response, 'status_code', 'N/A')}")
        logger.debug(f"DashScope task_response.message: {getattr(task_response, 'message', 'N/A')}")
        logger.debug(f"DashScope task_response.output: {getattr(task_response, 'output', 'N/A')}")
        
        # 检查响应状态
        if hasattr(task_response, 'status_code') and task_response.status_code != HTTPStatus.OK:
            error_msg = (
                f"DashScope async_call failed with status {task_response.status_code}: "
                f"{getattr(task_response, 'message', getattr(task_response, 'output', 'Unknown error'))}"
            )
            logger.error(error_msg)
            raise AsrServiceError(error_msg)

        task_output = self._as_dict(getattr(task_response, "output", None))
        logger.debug(f"DashScope task_output: {task_output}")
        task_id = task_output.get("task_id")
        if not task_id:
            raise AsrServiceError(
                f"DashScope SDK response did not include task_id. "
                f"Response status: {getattr(task_response, 'status_code', 'N/A')}, "
                f"Output: {task_output}"
            )

        logger.info(f"DashScope task submitted: {task_id}")
        transcribe_response = Transcription.wait(task=task_id)
        if transcribe_response.status_code != HTTPStatus.OK:
            raise AsrServiceError(
                f"DashScope transcription failed with status {transcribe_response.status_code}: "
                f"{getattr(transcribe_response, 'message', transcribe_response.output)}"
            )

        result_output = self._as_dict(getattr(transcribe_response, "output", None))
        logger.debug(f"DashScope transcription output: {result_output}")
        return self._parse_dashscope_result(result_output)

    def _transcribe_with_sdk_upload(self, *, data: bytes, filename: str, language: str | None) -> list[ConversationItem]:
        """Upload local bytes to DashScope then start async transcription."""
        file_urls = self._upload_file_to_dashscope(data=data, filename=filename)
        logger.info(f"Uploaded audio to DashScope, obtained file_urls={file_urls}")
        return self._transcribe_with_sdk(file_urls=file_urls, language=language)

    def _transcribe_direct(self, data: bytes, filename: str, language: str | None) -> list[ConversationItem]:
        """Fallback method using direct API call"""
        logger.info("Using fallback direct API method")

        payload = self._build_payload(language=language)
        logger.debug(f"Payload: {payload}")

        files = {
            "file": (filename, data),
            "payload": (None, json.dumps(payload), "application/json"),
        }

        logger.info(f"Sending request to Aliyun Bailian endpoint: {self.settings.endpoint}")
        logger.debug(f"Headers: {self._headers()}")
        logger.debug(f"Timeout: {self.settings.timeout}")
        logger.debug(f"Retries: {self.settings.retries}")

        for attempt in range(1, self.settings.retries + 1):
            logger.info(f"Attempt {attempt}/{self.settings.retries}")
            try:
                response = requests.post(
                    self.settings.endpoint,
                    headers=self._headers(),
                    files=files,
                    timeout=self.settings.timeout,
                )
                logger.info(f"Response status code: {response.status_code}")
                logger.debug(f"Response headers: {response.headers}")
                logger.debug(f"Response text: {response.text}")

                if response.status_code >= 500 and attempt < self.settings.retries:
                    logger.warning(f"Server error (status {response.status_code}), retrying in {2**attempt} seconds...")
                    time.sleep(2**attempt)
                    continue
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt}: {e}")
                if attempt < self.settings.retries:
                    logger.info(f"Retrying in {2**attempt} seconds...")
                    time.sleep(2**attempt)
                    continue
                raise AsrServiceError(f"Network error during transcription: {e}")

        if response.status_code != 200:
            error_msg = f"Aliyun Bailian request failed with status {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise AsrServiceError(error_msg)

        try:
            content = response.json()
            logger.debug(f"Response JSON: {content}")
        except json.JSONDecodeError as e:
            error_msg = f"Failed to decode JSON response: {e}, response text: {response.text}"
            logger.error(error_msg)
            raise AsrServiceError(error_msg)

        if (code := content.get("code")) not in {None, "Success"}:
            error_msg = f"Aliyun Bailian returned error code={code}, request_id={content.get('request_id')}"
            logger.error(error_msg)
            raise AsrServiceError(error_msg)

        asr_result = content.get("data") or content.get("output") or {}
        logger.debug(f"ASR result: {asr_result}")

        items = self._parse_items(asr_result)
        logger.info(f"Successfully parsed {len(items)} conversation items")
        return items

    def _parse_dashscope_result(self, result: dict[str, Any]) -> list[ConversationItem]:
        """Parse DashScope ASR result format"""
        logger.debug(f"Parsing DashScope result: {result}")

        # DashScope 可能直接返回 segments/sentences
        if result.get("segments") or result.get("sentences"):
            return self._parse_items(result)

        # 从结果中获取转录URL并下载转录内容
        results = result.get("results", [])
        if not results:
            logger.warning("No results found in DashScope response")
            return []

        # 获取第一个结果的转录URL
        transcription_url = results[0].get("transcription_url")
        if not transcription_url:
            logger.warning("No transcription URL found in results")
            return []

        logger.info(f"Downloading transcription from: {transcription_url}")
        try:
            response = requests.get(transcription_url, timeout=self.settings.timeout)
        except requests.exceptions.RequestException as exc:
            logger.error(f"Failed to download DashScope transcription: {exc}")
            raise AsrServiceError(f"Failed to download DashScope transcription: {exc}") from exc

        if response.status_code != HTTPStatus.OK:
            error_msg = f"Failed to download transcription: {response.status_code}"
            logger.error(error_msg)
            raise AsrServiceError(error_msg)

        try:
            transcription_data = response.json()
            logger.debug(f"Transcription data: {transcription_data}")
            if transcription_data.get("transcripts"):
                logger.debug("Detected DashScope transcripts format")
                transcription_data = {
                    "segments": self._segments_from_transcripts(transcription_data["transcripts"])
                }
            return self._parse_items(transcription_data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode transcription JSON: {e}")
            return []

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
        }

    def _build_payload(self, *, language: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "input": {
                "parameters": {
                    "language": language or self.settings.language,
                    "enable_words": self.settings.enable_words,
                    "enable_diarization": self.settings.diarization,
                }
            },
        }
        return payload

    def _parse_items(self, result: dict[str, Any]) -> list[ConversationItem]:
        segments = result.get("segments") or result.get("sentences") or []
        items: list[ConversationItem] = []
        for segment in segments:
            words = [
                ConversationWord(
                    text=word.get("text") or word.get("word") or "",
                    start_time=word.get("start") or word.get("start_time"),
                    end_time=word.get("end") or word.get("end_time"),
                )
                for word in segment.get("words", [])
            ]
            items.append(
                ConversationItem(
                    text=(segment.get("text") or segment.get("sentence") or "").strip(),
                    start_time=segment.get("start") or segment.get("begin_time"),
                    end_time=segment.get("end") or segment.get("end_time"),
                    speaker_id=segment.get("speaker_id") or segment.get("speaker"),
                    speaker=segment.get("speaker_label"),
                    words=words,
                )
            )
        return items

    @staticmethod
    def _segments_from_transcripts(transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for transcript in transcripts:
            for sentence in transcript.get("sentences", []):
                words = [
                    {
                        "text": word.get("text"),
                        "start": word.get("begin_time"),
                        "end": word.get("end_time"),
                    }
                    for word in sentence.get("words", [])
                ]
                segments.append(
                    {
                        "text": sentence.get("text", ""),
                        "start": sentence.get("begin_time"),
                        "end": sentence.get("end_time"),
                        "words": words,
                        "speaker": transcript.get("speaker"),
                        "speaker_label": transcript.get("speaker_label"),
                        "speaker_id": transcript.get("speaker_id"),
                    }
                )
        return segments

    def _upload_file_to_dashscope(self, *, data: bytes, filename: str) -> list[str]:
        """Upload local audio to DashScope files endpoint and return URLs."""
        endpoint = self.settings.file_upload_endpoint or f"{self.settings.base_http_api_url.rstrip('/')}/files"
        logger.info(f"Uploading audio to DashScope endpoint: {endpoint}")

        try:
            response = requests.post(
                endpoint,
                headers=self._headers(),
                files={"file": (filename, data)},
                timeout=self.settings.timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise AsrServiceError(f"Failed to upload audio to DashScope: {exc}") from exc

        if response.status_code != HTTPStatus.OK:
            raise AsrServiceError(
                f"DashScope file upload failed with status {response.status_code}: {response.text}"
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise AsrServiceError(f"DashScope file upload returned invalid JSON: {exc}") from exc

        logger.debug(f"DashScope upload response: {payload}")
        file_urls = (
            payload.get("output", {}).get("file_urls")
            or payload.get("file_urls")
            or payload.get("data", {}).get("file_urls")
            or payload.get("data", {}).get("urls")
        )
        if not file_urls:
            raise AsrServiceError("DashScope file upload did not return any file_urls")
        if isinstance(file_urls, str):
            file_urls = [file_urls]
        return list(file_urls)

    @staticmethod
    def _as_dict(value: Any | None) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        return {}
