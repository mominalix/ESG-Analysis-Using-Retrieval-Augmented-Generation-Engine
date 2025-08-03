"""
Vector Store Service for managing document embeddings and similarity search
"""
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Pinecone, Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import pinecone
import chromadb
from langsmith import traceable

from ..core.config import settings
from ..core.logging import LoggingMixin
from ..core.exceptions import VectorStoreError, ConfigurationError


@dataclass
class SearchResult:
    """Document search result with metadata"""
    document: Document
    score: float
    rank: int


class BaseVectorStore(ABC, LoggingMixin):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store"""
        pass
    
    @abstractmethod
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search"""
        pass
    
    @abstractmethod
    async def hybrid_search(
        self, 
        query: str, 
        k: int = 5,
        alpha: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform hybrid (semantic + keyword) search"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from vector store"""
        pass


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, collection_name: str = "esg_documents"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key
        )
        
        try:
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_persist_directory,
            )
            self.logger.info("ChromaDB vector store initialized", collection=collection_name)
        except Exception as e:
            self.logger.error("Failed to initialize ChromaDB", error=str(e))
            raise VectorStoreError(f"ChromaDB initialization failed: {str(e)}")
    
    @traceable(name="chroma_add_documents")
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to ChromaDB"""
        try:
            self.logger.info("Adding documents to ChromaDB", count=len(documents))
            
            # ChromaDB add_documents is synchronous, so we run it in executor
            loop = asyncio.get_event_loop()
            ids = await loop.run_in_executor(
                None, 
                self.vector_store.add_documents, 
                documents
            )
            
            self.logger.info("Documents added successfully", ids_count=len(ids))
            return ids
        except Exception as e:
            self.logger.error("Failed to add documents to ChromaDB", error=str(e))
            raise VectorStoreError(f"ChromaDB add documents failed: {str(e)}")
    
    @traceable(name="chroma_similarity_search")
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search with ChromaDB"""
        try:
            self.logger.info("Performing similarity search", query_length=len(query), k=k)
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.vector_store.similarity_search_with_score,
                query,
                k,
                filter_dict
            )
            
            search_results = [
                SearchResult(
                    document=doc,
                    score=score,
                    rank=idx
                )
                for idx, (doc, score) in enumerate(results)
            ]
            
            self.logger.info("Similarity search completed", results_count=len(search_results))
            return search_results
        except Exception as e:
            self.logger.error("Similarity search failed", error=str(e))
            raise VectorStoreError(f"ChromaDB similarity search failed: {str(e)}")
    
    async def hybrid_search(
        self, 
        query: str, 
        k: int = 5,
        alpha: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """ChromaDB doesn't natively support hybrid search, fall back to similarity search"""
        self.logger.warning("ChromaDB doesn't support hybrid search, using similarity search")
        return await self.similarity_search(query, k, filter_dict)
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.vector_store.delete,
                document_ids
            )
            self.logger.info("Documents deleted", count=len(document_ids))
            return True
        except Exception as e:
            self.logger.error("Failed to delete documents", error=str(e))
            raise VectorStoreError(f"ChromaDB delete failed: {str(e)}")


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation with hybrid search support"""
    
    def __init__(self, collection_name: str = "esg_documents"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key
        )
        
        try:
            from qdrant_client import QdrantClient
            
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
            
            # Initialize collection if it doesn't exist
            self._initialize_collection()
            
            self.vector_store = Qdrant(
                client=self.client,
                collection_name=collection_name,
                embeddings=self.embeddings,
            )
            
            self.logger.info("Qdrant vector store initialized", collection=collection_name)
        except Exception as e:
            self.logger.error("Failed to initialize Qdrant", error=str(e))
            raise VectorStoreError(f"Qdrant initialization failed: {str(e)}")
    
    def _initialize_collection(self):
        """Initialize Qdrant collection with proper configuration"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info("Created new Qdrant collection", collection=self.collection_name)
        except Exception as e:
            self.logger.error("Failed to initialize Qdrant collection", error=str(e))
            raise VectorStoreError(f"Qdrant collection initialization failed: {str(e)}")
    
    @traceable(name="qdrant_add_documents")
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to Qdrant"""
        try:
            self.logger.info("Adding documents to Qdrant", count=len(documents))
            
            loop = asyncio.get_event_loop()
            ids = await loop.run_in_executor(
                None,
                self.vector_store.add_documents,
                documents
            )
            
            self.logger.info("Documents added successfully", ids_count=len(ids))
            return ids
        except Exception as e:
            self.logger.error("Failed to add documents to Qdrant", error=str(e))
            raise VectorStoreError(f"Qdrant add documents failed: {str(e)}")
    
    @traceable(name="qdrant_similarity_search")
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search with Qdrant"""
        try:
            self.logger.info("Performing similarity search", query_length=len(query), k=k)
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.vector_store.similarity_search_with_score,
                query,
                k,
                filter_dict
            )
            
            search_results = [
                SearchResult(
                    document=doc,
                    score=score,
                    rank=idx
                )
                for idx, (doc, score) in enumerate(results)
            ]
            
            self.logger.info("Similarity search completed", results_count=len(search_results))
            return search_results
        except Exception as e:
            self.logger.error("Similarity search failed", error=str(e))
            raise VectorStoreError(f"Qdrant similarity search failed: {str(e)}")
    
    async def hybrid_search(
        self, 
        query: str, 
        k: int = 5,
        alpha: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform hybrid search with Qdrant"""
        try:
            self.logger.info("Performing hybrid search", query_length=len(query), k=k, alpha=alpha)
            
            # For now, fall back to similarity search
            # TODO: Implement true hybrid search with BM25 + semantic
            return await self.similarity_search(query, k, filter_dict)
        except Exception as e:
            self.logger.error("Hybrid search failed", error=str(e))
            raise VectorStoreError(f"Qdrant hybrid search failed: {str(e)}")
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Qdrant"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.vector_store.delete,
                document_ids
            )
            self.logger.info("Documents deleted", count=len(document_ids))
            return True
        except Exception as e:
            self.logger.error("Failed to delete documents", error=str(e))
            raise VectorStoreError(f"Qdrant delete failed: {str(e)}")


class VectorStoreService(LoggingMixin):
    """Main vector store service that manages different providers"""
    
    def __init__(self):
        self.stores: Dict[str, BaseVectorStore] = {}
        self._initialize_stores()
    
    def _initialize_stores(self) -> None:
        """Initialize available vector stores"""
        try:
            if settings.vector_store_type.lower() == "chroma":
                self.stores["chroma"] = ChromaVectorStore(settings.chroma_collection_name)
                self.logger.info("ChromaDB store initialized")
            elif settings.vector_store_type.lower() == "qdrant":
                self.stores["qdrant"] = QdrantVectorStore(settings.qdrant_collection_name)
                self.logger.info("Qdrant store initialized")
            else:
                # Default to ChromaDB
                self.stores["chroma"] = ChromaVectorStore(settings.chroma_collection_name)
                self.logger.info("Default ChromaDB store initialized")
        except Exception as e:
            self.logger.error("Failed to initialize vector stores", error=str(e))
            raise VectorStoreError(f"Vector store initialization failed: {str(e)}")
    
    def get_store(self, store_name: Optional[str] = None) -> BaseVectorStore:
        """Get vector store by name"""
        store_name = store_name or settings.vector_store_type.lower()
        
        if store_name not in self.stores:
            raise VectorStoreError(f"Vector store '{store_name}' not available")
        
        return self.stores[store_name]
    
    async def add_documents(
        self, 
        documents: List[Document], 
        store_name: Optional[str] = None
    ) -> List[str]:
        """Add documents to vector store"""
        store = self.get_store(store_name)
        return await store.add_documents(documents)
    
    async def search(
        self,
        query: str,
        k: int = 5,
        search_type: str = "similarity",
        alpha: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None,
        store_name: Optional[str] = None
    ) -> List[SearchResult]:
        """Search documents in vector store"""
        store = self.get_store(store_name)
        
        if search_type == "hybrid":
            return await store.hybrid_search(query, k, alpha, filter_dict)
        else:
            return await store.similarity_search(query, k, filter_dict)
    
    async def delete_documents(
        self, 
        document_ids: List[str], 
        store_name: Optional[str] = None
    ) -> bool:
        """Delete documents from vector store"""
        store = self.get_store(store_name)
        return await store.delete_documents(document_ids)


# Global vector store service instance
vector_store_service = VectorStoreService()