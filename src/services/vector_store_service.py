"""Pluggable vector/document stores with a dependency-free local provider."""

from __future__ import annotations

import asyncio
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langsmith import traceable

from ..core.config import reveal, settings
from ..core.exceptions import ConfigurationError, VectorStoreError
from ..core.logging import LoggingMixin


@dataclass(slots=True)
class SearchResult:
    document: Document
    score: float
    rank: int


def _matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    return all(
        metadata.get(key) == getattr(value, "value", value) for key, value in filters.items()
    )


class BaseVectorStore(ABC, LoggingMixin):
    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def similarity_search(
        self, query: str, k: int = 5, filter_dict: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        raise NotImplementedError

    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        # Providers can override this when native sparse+dense search is enabled.
        return await self.similarity_search(query, k, filter_dict)

    @abstractmethod
    async def all_documents(self, filter_dict: dict[str, Any] | None = None) -> list[Document]:
        raise NotImplementedError

    @abstractmethod
    async def delete_chunks(self, chunk_ids: list[str]) -> None:
        raise NotImplementedError

    async def health(self) -> str:
        return "healthy"


class MemoryVectorStore(BaseVectorStore):
    """Process-local lexical store for development, tests, and small datasets."""

    _token_pattern = re.compile(r"[\w-]+", re.UNICODE)

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _chunk_id(document: Document) -> str:
        chunk_id = document.metadata.get("chunk_id")
        if not chunk_id:
            raise VectorStoreError("Every document chunk must include chunk_id metadata")
        return str(chunk_id)

    @classmethod
    def _tokens(cls, text: str) -> Counter[str]:
        return Counter(cls._token_pattern.findall(text.casefold()))

    @classmethod
    def _score(cls, query: str, content: str) -> float:
        query_tokens = cls._tokens(query)
        document_tokens = cls._tokens(content)
        if not query_tokens or not document_tokens:
            return 0.0
        numerator = sum(count * document_tokens[token] for token, count in query_tokens.items())
        query_norm = math.sqrt(sum(count * count for count in query_tokens.values()))
        document_norm = math.sqrt(sum(count * count for count in document_tokens.values()))
        return numerator / (query_norm * document_norm) if query_norm and document_norm else 0.0

    async def add_documents(self, documents: list[Document]) -> list[str]:
        ids = [self._chunk_id(document) for document in documents]
        async with self._lock:
            self._documents.update(zip(ids, documents, strict=True))
        return ids

    async def similarity_search(
        self, query: str, k: int = 5, filter_dict: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        documents = await self.all_documents(filter_dict)
        scored = sorted(
            ((document, self._score(query, document.page_content)) for document in documents),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        return [
            SearchResult(document=document, score=score, rank=index)
            for index, (document, score) in enumerate(scored, start=1)
        ]

    async def all_documents(self, filter_dict: dict[str, Any] | None = None) -> list[Document]:
        async with self._lock:
            documents = list(self._documents.values())
        return [document for document in documents if _matches(document.metadata, filter_dict)]

    async def delete_chunks(self, chunk_ids: list[str]) -> None:
        async with self._lock:
            for chunk_id in chunk_ids:
                self._documents.pop(chunk_id, None)


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str):
        try:
            from langchain_chroma import Chroma
            from langchain_openai import OpenAIEmbeddings

            api_key = reveal(settings.openai_api_key)
            if not api_key:
                raise ConfigurationError("OPENAI_API_KEY is required for Chroma embeddings")
            embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=api_key)
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(settings.chroma_persist_directory),
            )
        except (ImportError, ConfigurationError):
            raise
        except Exception as exc:
            raise VectorStoreError("Chroma initialization failed") from exc

    async def add_documents(self, documents: list[Document]) -> list[str]:
        ids = [str(document.metadata["chunk_id"]) for document in documents]
        try:
            return await asyncio.to_thread(self.vector_store.add_documents, documents, ids=ids)
        except Exception as exc:
            raise VectorStoreError("Chroma document indexing failed") from exc

    @traceable(name="chroma_similarity_search")
    async def similarity_search(
        self, query: str, k: int = 5, filter_dict: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        try:
            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_relevance_scores,
                query,
                k=k,
                filter=filter_dict,
            )
            return [
                SearchResult(document=document, score=max(0.0, min(1.0, score)), rank=index)
                for index, (document, score) in enumerate(results, start=1)
            ]
        except Exception as exc:
            raise VectorStoreError("Chroma search failed") from exc

    async def all_documents(self, filter_dict: dict[str, Any] | None = None) -> list[Document]:
        try:
            result = await asyncio.to_thread(
                self.vector_store.get,
                where=filter_dict,
                include=["documents", "metadatas"],
            )
            return [
                Document(page_content=content or "", metadata=metadata or {})
                for content, metadata in zip(
                    result.get("documents", []), result.get("metadatas", []), strict=True
                )
            ]
        except Exception as exc:
            raise VectorStoreError("Unable to enumerate Chroma documents") from exc

    async def delete_chunks(self, chunk_ids: list[str]) -> None:
        try:
            await asyncio.to_thread(self.vector_store.delete, ids=chunk_ids)
        except Exception as exc:
            raise VectorStoreError("Chroma document deletion failed") from exc


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str):
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_qdrant import QdrantVectorStore as LangChainQdrant
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            api_key = reveal(settings.openai_api_key)
            if not api_key:
                raise ConfigurationError("OPENAI_API_KEY is required for Qdrant embeddings")

            self.collection_name = collection_name
            self.client = QdrantClient(
                url=settings.qdrant_url, api_key=reveal(settings.qdrant_api_key)
            )
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
            self.vector_store = LangChainQdrant(
                client=self.client,
                collection_name=collection_name,
                embedding=OpenAIEmbeddings(
                    model=settings.openai_embedding_model,
                    api_key=api_key,
                ),
            )
        except (ImportError, ConfigurationError):
            raise
        except Exception as exc:
            raise VectorStoreError("Qdrant initialization failed") from exc

    async def add_documents(self, documents: list[Document]) -> list[str]:
        ids = [str(document.metadata["chunk_id"]) for document in documents]
        try:
            return await asyncio.to_thread(self.vector_store.add_documents, documents, ids=ids)
        except Exception as exc:
            raise VectorStoreError("Qdrant document indexing failed") from exc

    @traceable(name="qdrant_similarity_search")
    async def similarity_search(
        self, query: str, k: int = 5, filter_dict: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        try:
            qdrant_filter = self._filter(filter_dict)
            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_relevance_scores,
                query,
                k=k,
                filter=qdrant_filter,
            )
            return [
                SearchResult(document=document, score=max(0.0, min(1.0, score)), rank=index)
                for index, (document, score) in enumerate(results, start=1)
            ]
        except Exception as exc:
            raise VectorStoreError("Qdrant search failed") from exc

    @staticmethod
    def _filter(filter_dict: dict[str, Any] | None):
        if not filter_dict:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=getattr(value, "value", value)),
                )
                for key, value in filter_dict.items()
            ]
        )

    async def all_documents(self, filter_dict: dict[str, Any] | None = None) -> list[Document]:
        def scroll_all() -> list[Document]:
            records: list[Document] = []
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=self._filter(filter_dict),
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    records.append(
                        Document(
                            page_content=str(payload.get("page_content", "")),
                            metadata=dict(payload.get("metadata", {})),
                        )
                    )
                if offset is None:
                    break
            return records

        try:
            return await asyncio.to_thread(scroll_all)
        except Exception as exc:
            raise VectorStoreError("Unable to enumerate Qdrant documents") from exc

    async def delete_chunks(self, chunk_ids: list[str]) -> None:
        try:
            await asyncio.to_thread(self.vector_store.delete, ids=chunk_ids)
        except Exception as exc:
            raise VectorStoreError("Qdrant document deletion failed") from exc

    async def health(self) -> str:
        try:
            await asyncio.to_thread(self.client.get_collections)
            return "healthy"
        except Exception:
            return "unhealthy"


StoreFactory = Callable[[], BaseVectorStore]


class VectorStoreService(LoggingMixin):
    """Lazy provider registry plus provider-independent document operations."""

    def __init__(self) -> None:
        self._stores: dict[str, BaseVectorStore] = {}
        self._factories: dict[str, StoreFactory] = {
            "memory": MemoryVectorStore,
            "chroma": lambda: ChromaVectorStore(settings.chroma_collection_name),
            "qdrant": lambda: QdrantVectorStore(settings.qdrant_collection_name),
        }

    def register_store(self, name: str, factory: StoreFactory) -> None:
        self._factories[name.lower()] = factory

    def get_store(self, store_name: str | None = None) -> BaseVectorStore:
        name = (store_name or settings.vector_store_type).lower()
        if name not in self._factories:
            raise ConfigurationError(f"Unknown vector store provider: {name}")
        if name not in self._stores:
            self._stores[name] = self._factories[name]()
        return self._stores[name]

    async def health(self) -> str:
        try:
            return await self.get_store().health()
        except Exception as exc:
            self.logger.warning("Vector store health check failed", error=str(exc))
            return "unhealthy"

    async def add_documents(
        self, documents: list[Document], store_name: str | None = None
    ) -> list[str]:
        if not documents:
            raise VectorStoreError("No document chunks were supplied for indexing")
        return await self.get_store(store_name).add_documents(documents)

    async def search(
        self,
        query: str,
        k: int = 5,
        search_type: str = "similarity",
        alpha: float = 0.5,
        filter_dict: dict[str, Any] | None = None,
        store_name: str | None = None,
    ) -> list[SearchResult]:
        store = self.get_store(store_name)
        if search_type == "hybrid":
            return await store.hybrid_search(query, k, alpha, filter_dict)
        return await store.similarity_search(query, k, filter_dict)

    async def list_documents(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        filter_dict: dict[str, Any] | None = None,
        store_name: str | None = None,
    ) -> tuple[list[Document], int]:
        chunks = await self.get_store(store_name).all_documents(filter_dict)
        unique: dict[str, Document] = {}
        for chunk in chunks:
            document_hash = str(chunk.metadata.get("document_hash", ""))
            if document_hash and document_hash not in unique:
                unique[document_hash] = chunk
        documents = sorted(
            unique.values(),
            key=lambda item: str(item.metadata.get("processed_at", "")),
            reverse=True,
        )
        return documents[offset : offset + limit], len(documents)

    async def get_document(
        self, document_hash: str, store_name: str | None = None
    ) -> list[Document]:
        chunks = await self.get_store(store_name).all_documents({"document_hash": document_hash})
        return sorted(chunks, key=lambda chunk: int(chunk.metadata.get("chunk_index", 0)))

    async def delete_document(self, document_hash: str, store_name: str | None = None) -> bool:
        store = self.get_store(store_name)
        chunks = await store.all_documents({"document_hash": document_hash})
        chunk_ids = [
            str(chunk.metadata["chunk_id"]) for chunk in chunks if chunk.metadata.get("chunk_id")
        ]
        if not chunk_ids:
            return False
        await store.delete_chunks(chunk_ids)
        return True

    async def stats(self, store_name: str | None = None) -> dict[str, Any]:
        chunks = await self.get_store(store_name).all_documents()
        documents: dict[str, dict[str, Any]] = {}
        total_content_length = 0
        for chunk in chunks:
            metadata = chunk.metadata
            document_hash = str(metadata.get("document_hash", ""))
            if not document_hash:
                continue
            total_content_length += len(chunk.page_content)
            record = documents.setdefault(
                document_hash,
                {
                    "document_hash": document_hash,
                    "filename": metadata.get("filename", "unknown"),
                    "framework": metadata.get("esg_framework"),
                    "category": metadata.get("esg_category"),
                    "doc_type": metadata.get("document_type"),
                    "chunks": 0,
                },
            )
            record["chunks"] += 1

        def counts(key: str) -> dict[str, int]:
            output: dict[str, int] = {}
            for document in documents.values():
                value = document.get(key)
                if value:
                    output[str(value)] = output.get(str(value), 0) + 1
            return output

        return {
            "total_documents": len(documents),
            "documents_by_framework": counts("framework"),
            "documents_by_category": counts("category"),
            "documents_by_type": counts("doc_type"),
            "total_chunks": len(chunks),
            "average_chunk_size": round(total_content_length / len(chunks)) if chunks else 0,
            "documents": list(documents.values()),
        }


vector_store_service = VectorStoreService()
