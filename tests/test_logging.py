#!/usr/bin/env python3
"""Simple test script to verify logging works correctly."""

import logging
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 为项目模块设置日志级别
logging.getLogger("docling_av_transcriber").setLevel(logging.DEBUG)

def test_imports():
    """Test that all modules can be imported correctly."""
    try:
        from docling_av_transcriber import transcribe_file
        from docling_av_transcriber.models.aliyun_bailian import AliyunBailianAsrClient
        from docling_av_transcriber.pipelines.asr_pipeline import AsrPipelineLite
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_logger():
    """Test that logging works."""
    logger = logging.getLogger("test_script")
    logger.info("Testing logger - this should appear in output")
    print("✓ Logger test completed")
    return True

if __name__ == "__main__":
    print("Running basic tests...")

    if test_logger() and test_imports():
        print("✓ All tests passed")
    else:
        print("✗ Some tests failed")
        sys.exit(1)