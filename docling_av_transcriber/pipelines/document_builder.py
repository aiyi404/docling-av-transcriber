from __future__ import annotations

import logging

from docling_core.types.doc import DoclingDocument, DocumentOrigin
from docling_core.types.doc.labels import DocItemLabel

from docling_av_transcriber.types import ConversationItem

# 设置日志记录器
logger = logging.getLogger(__name__)


def build_docling_document(
    *,
    filename: str,
    mimetype: str,
    binary_hash: str,
    conversation: list[ConversationItem],
    summary: str | None = None,
) -> DoclingDocument:
    logger.info(f"Building Docling document for file: {filename}")
    logger.debug(f"Mimetype: {mimetype}, Binary hash: {binary_hash}")
    logger.info(f"Processing {len(conversation)} conversation items")

    origin = DocumentOrigin(filename=filename, mimetype=mimetype, binary_hash=binary_hash)
    document = DoclingDocument(name=filename, origin=origin)

    if summary:
        summary_text = f"[summary] {summary.strip()}"
        logger.debug(f"Adding summary: {summary_text[:100]}...")
        document.add_text(label=DocItemLabel.TEXT, text=summary_text)

    logger.debug("Adding conversation items to document")
    for i, item in enumerate(sorted(conversation)):
        text = item.to_string()
        logger.debug(f"Adding item {i}: {text[:100]}...")
        document.add_text(label=DocItemLabel.TEXT, text=text)

    logger.info(f"Document built successfully with {len(document.texts)} text items")
    return document
