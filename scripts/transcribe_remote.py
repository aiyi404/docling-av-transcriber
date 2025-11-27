#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from docling_core.types.doc.labels import DocItemLabel

from docling_av_transcriber import transcribe_file_with_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a remote audio/video file, run the Docling pipeline, and export the result."
    )
    parser.add_argument("file_url", help="HTTP(S) URL that can be downloaded directly.")
    parser.add_argument(
        "--language",
        default="zh",
        help="Language hint passed to the ASR provider (default: zh).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("transcription.json"),
        help="Path to write the Docling document (default: transcription.json).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "md"],
        default="json",
        help="Output format. json -> DoclingDocument JSON, md -> markdown export. Default: json.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional summary text to prepend to the document.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the extracted WAV artifact instead of deleting it.",
    )
    return parser.parse_args()


def _download_remote_media(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise SystemExit(f"仅支持 http/https URL，收到: {url}")

    logging.info("Downloading remote media: %s", url)
    try:
        response = requests.get(url, stream=True, timeout=180)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"下载远程媒体失败: {exc}") from exc

    suffix = Path(parsed.path).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                tmp.write(chunk)
        temp_path = Path(tmp.name)

    logging.info("Remote media saved to %s", temp_path)
    return temp_path


def _apply_summary(document, summary: str | None) -> None:
    if not summary:
        return
    text = summary.strip()
    if not text:
        return
    document.add_text(label=DocItemLabel.TEXT, text=f"[summary] {text}")


def _export_document(document, output: Path, fmt: str) -> None:
    if fmt == "md":
        output.write_text(document.export_to_markdown(), encoding="utf-8")
    else:
        payload = document.model_dump()
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    temp_source = _download_remote_media(args.file_url)
    try:
        logging.info("Starting pipeline transcription")
        result = transcribe_file_with_artifacts(temp_source, language=args.language)
    finally:
        temp_source.unlink(missing_ok=True)
        logging.info("Removed downloaded source file: %s", temp_source)

    _apply_summary(result.document, args.summary)
    _export_document(result.document, args.output, args.format)
    logging.info("Document written to %s", args.output.resolve())

    audio_path = result.audio_path
    if audio_path is None:
        logging.info("No audio stream detected; no WAV artifact produced.")
    else:
        logging.info("Extracted WAV artifact: %s", audio_path)
        if not args.keep_audio:
            audio_path.unlink(missing_ok=True)
            logging.info("Removed temporary WAV artifact.")
        else:
            logging.info("Keeping WAV artifact as requested.")


if __name__ == "__main__":
    main()
