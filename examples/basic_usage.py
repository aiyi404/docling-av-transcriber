"""Example CLI for docling-av-transcriber."""

import argparse
import logging
import sys
from pathlib import Path

from docling_av_transcriber import transcribe_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 为特定模块设置更详细的日志级别
logging.getLogger("docling_av_transcriber").setLevel(logging.DEBUG)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio/video via Aliyun Bailian")
    parser.add_argument("path", type=Path, help="音频或视频文件路径")
    parser.add_argument("--language", default="zh")
    args = parser.parse_args()

    doc = transcribe_file(args.path, language=args.language)
    print(doc.export_to_markdown())


if __name__ == "__main__":
    main()
