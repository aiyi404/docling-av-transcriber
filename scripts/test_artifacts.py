#!/usr/bin/env python
"""Quick smoke test for transcribe_with_artifacts."""

from __future__ import annotations

import argparse
import logging
import tempfile
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

from docling_av_transcriber import transcribe_file_with_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe local audio/video and verify returned audio artifact."
    )
    parser.add_argument("path", type=str, help="本地文件路径或 HTTP(S) URL。")
    parser.add_argument(
        "--language",
        default="zh",
        help="ASR 语言提示，默认 zh。",
    )
    parser.add_argument(
        "--save-markdown",
        type=Path,
        help="可选，将生成的 Docling 文档导出为 Markdown 文件。",
    )
    parser.add_argument(
        "--copy-audio-to",
        type=Path,
        help="可选，将抽取后的 WAV 音频复制到指定路径，方便后续上传。",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="在验证完成后删除临时 WAV 文件。",
    )
    return parser.parse_args()


def _prepare_input(path_str: str) -> tuple[Path, bool]:
    """Return a local Path; download remote files to a temp location."""
    parsed = urlparse(path_str)
    if parsed.scheme in {"http", "https"}:
        logging.info("Downloading remote media: %s", path_str)
        try:
            response = requests.get(path_str, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SystemExit(f"下载远程媒体失败: {exc}") from exc
        suffix = Path(parsed.path).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(response.content)
            temp_path = Path(tmp.name)
        logging.info("已下载到临时文件: %s", temp_path)
        return temp_path, True

    local_path = Path(path_str)
    if not local_path.exists():
        raise SystemExit(f"文件不存在: {local_path}")
    return local_path, False


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    local_path, is_temp_source = _prepare_input(args.path)

    logging.info("Running artifact smoke test: %s", local_path)
    result = transcribe_file_with_artifacts(local_path, language=args.language)
    audio_path = result.audio_path

    logging.info("Docling 文档名称: %s", result.document.name)

    if audio_path is None:
        logging.info("输入文件没有音频流，未生成 WAV 音轨。")
        if args.copy_audio_to:
            logging.warning("无音轨可复制，已忽略 --copy-audio-to 参数。")
        if args.cleanup:
            logging.info("无 WAV 可清理，忽略 --cleanup 选项。")
    else:
        if not audio_path.exists():
            raise SystemExit(f"提取的音轨不存在: {audio_path}")

        logging.info(
            "WAV 文件: %s (%.2f MB)", audio_path, audio_path.stat().st_size / (1024 * 1024)
        )

        if args.copy_audio_to:
            args.copy_audio_to.write_bytes(audio_path.read_bytes())
            logging.info("已复制 WAV 到: %s", args.copy_audio_to)

        if args.cleanup:
            audio_path.unlink(missing_ok=True)
            logging.info("已清理临时 WAV: %s", audio_path)
        else:
            logging.info("临时 WAV 未删除，可用于后续上传。")

    if args.save_markdown:
        args.save_markdown.write_text(result.document.export_to_markdown(), encoding="utf-8")
        logging.info("已将 Markdown 写入: %s", args.save_markdown)

    if is_temp_source:
        local_path.unlink(missing_ok=True)
        logging.info("已清理下载的源文件: %s", local_path)


if __name__ == "__main__":
    main()
