from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from http import HTTPStatus
from pathlib import Path
from typing import Sequence

import requests

from docling_av_transcriber.config import AliyunVisionSettings
from docling_av_transcriber.models.base import ImageCaptionProvider
from docling_av_transcriber.types import (
    ConversationItem,
    VisionServiceError,
    KEYFRAME_PREFIX,
    KEYFRAME_SUFFIX,
)

logger = logging.getLogger(__name__)


class AliyunVisionClient(ImageCaptionProvider):
    """Aliyun DashScope multimodal client describing keyframes."""

    def __init__(self, settings: AliyunVisionSettings | None = None) -> None:
        self.settings = settings or AliyunVisionSettings.from_env()
        if not self.settings.api_key:
            raise VisionServiceError(
                "Aliyun vision API key is missing. Set ALIYUN_BAILIAN_API_KEY/DASHSCOPE_API_KEY or pass settings."
            )

    def describe_frames(self, frames: Sequence[tuple[Path, float]]) -> list[ConversationItem]:
        items: list[ConversationItem] = []
        for idx, (frame_path, timestamp_sec) in enumerate(frames, start=1):
            try:
                description = self._describe_single_frame(frame_path, timestamp_sec)
            except VisionServiceError as exc:
                logger.warning(
                    "Vision model failed on frame %d (%s): %s", idx, frame_path, exc
                )
                continue

            description = self._wrap_keyframe_text(description, timestamp_sec, idx)
            start_ms = int(timestamp_sec * 1000)
            items.append(
                ConversationItem(
                    text=description,
                    start_time=start_ms,
                    end_time=start_ms + 1,
                )
            )
        return items

    def _describe_single_frame(self, frame_path: Path, timestamp_sec: float) -> str:
        payload = self._build_payload(frame_path, timestamp_sec)
        response_data = self._post_with_retry(payload)
        text = self._extract_text(response_data).strip()
        if not text:
            raise VisionServiceError("Aliyun vision response did not include text output")
        return text

    def _build_payload(self, frame_path: Path, timestamp_sec: float) -> dict[str, object]:
        mime, _ = mimetypes.guess_type(str(frame_path))
        mime = mime or "image/jpeg"
        with frame_path.open("rb") as handle:
            image_bytes = handle.read()
        data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        prompt_text = f"时间戳 {timestamp_sec:.3f} 秒。{self.settings.prompt}".strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": data_uri},
                    {"text": prompt_text},
                ],
            }
        ]
        payload: dict[str, object] = {
            "model": self.settings.model,
            "input": {"messages": messages},
        }
        if self.settings.parameters:
            payload["parameters"] = self.settings.parameters
        return payload

    @staticmethod
    def _wrap_keyframe_text(description: str, timestamp_sec: float, index: int) -> str:
        meta = f"frame={index};time={timestamp_sec:.3f}s"
        return f"{KEYFRAME_PREFIX}{meta}\n{description}\n{KEYFRAME_SUFFIX}"

    def _post_with_retry(self, payload: dict[str, object]) -> dict[str, object]:
        for attempt in range(1, self.settings.retries + 1):
            try:
                response = requests.post(
                    self.settings.endpoint,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.settings.timeout,
                )
            except requests.exceptions.RequestException as exc:
                if attempt >= self.settings.retries:
                    raise VisionServiceError(
                        f"Network error during vision request: {exc}"
                    ) from exc
                logger.warning(
                    "Vision request failed on attempt %d/%d: %s",
                    attempt,
                    self.settings.retries,
                    exc,
                )
                time.sleep(2**attempt)
                continue

            if response.status_code != HTTPStatus.OK:
                if response.status_code >= 500 and attempt < self.settings.retries:
                    logger.warning(
                        "Vision API server error (%s) attempt %d/%d, retrying",
                        response.status_code,
                        attempt,
                        self.settings.retries,
                    )
                    time.sleep(2**attempt)
                    continue
                raise VisionServiceError(
                    f"Aliyun vision request failed with status {response.status_code}: {response.text}"
                )

            try:
                return response.json()
            except json.JSONDecodeError as exc:
                raise VisionServiceError(
                    f"Failed to decode vision response JSON: {exc}"
                ) from exc

        raise VisionServiceError("Aliyun vision request exhausted retries")

    def _extract_text(self, payload: dict[str, object]) -> str:
        output = payload.get("output")
        if isinstance(output, dict):
            choices = output.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message")
                    if isinstance(message, dict):
                        content_blocks = message.get("content")
                        if isinstance(content_blocks, list):
                            for block in content_blocks:
                                if isinstance(block, dict):
                                    text = block.get("text")
                                    if isinstance(text, str) and text.strip():
                                        return text
                        text = message.get("text")
                        if isinstance(text, str) and text.strip():
                            return text
                    text = choice.get("text")
                    if isinstance(text, str) and text.strip():
                        return text
            text = output.get("text")
            if isinstance(text, str) and text.strip():
                return text
        if isinstance(output, str) and output.strip():
            return output
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        return ""

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
