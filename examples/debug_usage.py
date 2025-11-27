"""Debug CLI for docling-av-transcriber with enhanced logging."""

import argparse
import logging
import sys
from pathlib import Path

from docling_av_transcriber import transcribe_file
from docling_av_transcriber.config import AliyunBailianSettings

# 配置详细的日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("debug_transcription.log", mode='w', encoding='utf-8')
    ]
)

# 为特定模块设置详细的日志级别
logging.getLogger("docling_av_transcriber").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)
logging.getLogger("requests").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def check_environment():
    """检查环境配置"""
    logger.info("Checking environment configuration...")

    api_key = AliyunBailianSettings.from_env().api_key
    if not api_key:
        logger.error("ALIYUN_BAILIAN_API_KEY environment variable is not set")
        return False

    logger.info("API key found (first 8 characters): %s...", api_key[:8] if api_key else "None")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug transcription via Aliyun Bailian")
    parser.add_argument("path", type=Path, help="音频或视频文件路径")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()

    if args.debug:
        logger.info("Debug mode enabled")

    # 检查环境
    if not check_environment():
        logger.error("Environment check failed")
        return

    logger.info(f"Starting transcription for file: {args.path}")
    logger.info(f"Language setting: {args.language}")

    try:
        doc = transcribe_file(args.path, language=args.language)
        logger.info("Transcription completed successfully")
        print(doc.export_to_markdown())
    except Exception as e:
        logger.error(f"Transcription failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()