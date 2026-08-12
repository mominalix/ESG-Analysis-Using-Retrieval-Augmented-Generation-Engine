from __future__ import annotations

import pytest

from src.services.document_service import DocumentService
from src.services.vector_store_service import MemoryVectorStore, VectorStoreService


@pytest.mark.asyncio
async def test_memory_store_search_list_stats_and_delete():
    service = VectorStoreService()
    store = MemoryVectorStore()
    service.register_store("test", lambda: store)
    chunks = await DocumentService().process_uploaded_document(
        b"The climate risk policy tracks carbon emissions and renewable energy.",
        "climate.txt",
        esg_framework="CSRD",
        document_type="policy",
    )

    await service.add_documents(chunks, "test")
    results = await service.search("carbon emissions", store_name="test")
    documents, total = await service.list_documents(store_name="test")
    stats = await service.stats("test")

    assert results[0].score > 0
    assert total == 1
    assert documents[0].metadata["filename"] == "climate.txt"
    assert stats["documents_by_framework"] == {"CSRD": 1}
    assert await service.delete_document(chunks[0].metadata["document_hash"], "test")
    assert (await service.stats("test"))["total_documents"] == 0
