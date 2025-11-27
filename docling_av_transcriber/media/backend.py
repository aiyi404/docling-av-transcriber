from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Union

# 设置日志记录器
logger = logging.getLogger(__name__)


@dataclass
class BackendResult:
    path_or_stream: Union[Path, BytesIO]
    filename: str


def validate_input(path_or_stream: Union[str, Path, BytesIO], filename: str | None = None) -> BackendResult:
    logger.info(f"Validating input of type: {type(path_or_stream)}")

    if isinstance(path_or_stream, (str, Path)):
        path = Path(path_or_stream)
        logger.info(f"Processing file path: {path}")
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(path)
        logger.info(f"File exists: {path}")
        result = BackendResult(path_or_stream=path, filename=path.name)
        logger.debug(f"BackendResult created: path={result.path_or_stream}, filename={result.filename}")
        return result

    if isinstance(path_or_stream, BytesIO):
        logger.info("Processing BytesIO stream")
        if not filename:
            logger.error("Filename is required when using BytesIO")
            raise ValueError("filename is required when using BytesIO")
        if path_or_stream.getbuffer().nbytes == 0:
            logger.error("Empty BytesIO provided")
            raise ValueError("empty BytesIO provided")
        logger.debug(f"BytesIO size: {path_or_stream.getbuffer().nbytes} bytes")
        result = BackendResult(path_or_stream=path_or_stream, filename=filename)
        logger.debug(f"BackendResult created: filename={result.filename}")
        return result

    error_msg = f"Unsupported input type: {type(path_or_stream)}"
    logger.error(error_msg)
    raise TypeError(error_msg)
