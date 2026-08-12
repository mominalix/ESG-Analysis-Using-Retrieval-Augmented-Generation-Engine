from __future__ import annotations

import pytest

from src.core.exceptions import DocumentProcessingError
from src.services.document_service import DocumentService


@pytest.mark.asyncio
async def test_text_document_uses_stable_sha256_and_chunk_ids():
    service = DocumentService()
    content = b"Climate emissions and renewable energy policy. " * 80

    first = await service.process_uploaded_document(content, "../climate-policy.txt", "CSRD")
    second = await service.process_uploaded_document(content, "climate-policy.txt", "CSRD")

    assert first[0].metadata["filename"] == "climate-policy.txt"
    assert len(first[0].metadata["document_hash"]) == 64
    assert [item.metadata["chunk_id"] for item in first] == [
        item.metadata["chunk_id"] for item in second
    ]
    assert first[0].metadata["esg_category"] == "Environmental"


@pytest.mark.asyncio
async def test_document_rejects_unsupported_extension():
    with pytest.raises(DocumentProcessingError, match="Unsupported file extension"):
        await DocumentService().process_uploaded_document(b"content", "payload.exe")
