"""
Advanced RAG Service implementing modern retrieval and generation techniques
"""

from typing import List, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
import asyncio
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from .llm_service import llm_service, LLMResponse
from .vector_store_service import vector_store_service, SearchResult
from ..core.config import settings
from ..core.logging import LoggingMixin
from ..core.exceptions import ESGAnalysisException
from ..core.taxonomy import get_taxonomy


@dataclass
class RAGResponse:
    """Response from RAG system with metadata"""

    answer: str
    source_documents: List[Document]
    confidence_score: float
    retrieval_time_ms: int
    generation_time_ms: int
    total_time_ms: int
    esg_framework: Optional[str] = None
    esg_categories: List[str] = field(default_factory=list)


@dataclass
class QueryDecomposition:
    """Decomposed query for complex question answering"""

    main_query: str
    sub_queries: List[str]
    strategy: str  # "sequential", "parallel", "hierarchical"


class AdvancedRAGService(LoggingMixin):
    """Advanced RAG service with modern techniques"""

    def __init__(self):
        self.esg_system_prompt = """You are an expert ESG (Environmental, Social, Governance) document analyst.

Your role is to:
1. Analyze ESG policies, reports, and frameworks
2. Identify gaps and compliance issues
3. Provide recommendations only when supported by evidence
4. Ensure accuracy by citing specific source documents
5. Consider regulatory requirements and best practices

When answering:
- Always ground your responses in the provided context
- Cite specific sources and page numbers when available
- Describe missing repository evidence as an evidence gap, not proof of non-compliance
- Provide practical recommendations
- If information is insufficient, clearly state limitations
- Use ESG terminology accurately and professionally

Context will be provided from relevant documents. Use only this context to answer questions."""

        self.query_decomposition_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at breaking down complex ESG-related questions into simpler sub-questions.

Given a complex question about ESG (Environmental, Social, Governance) topics, decompose it into 2-4 specific sub-questions that, when answered together, will provide a comprehensive response to the original question.

Return the decomposition as a JSON object with:
- "main_query": the original question
- "sub_queries": list of specific sub-questions
- "strategy": either "sequential" (answers build on each other) or "parallel" (can be answered independently)

Focus on ESG frameworks, compliance, reporting standards, and sustainability practices.""",
                ),
                ("human", "{query}"),
            ]
        )

    @traceable(name="rag_query")
    async def query(
        self,
        question: str,
        esg_framework: Optional[str] = None,
        search_strategy: str = "hybrid",
        k: int = 5,
        use_query_decomposition: bool = False,
        **kwargs: Any,
    ) -> RAGResponse:
        """Main RAG query method with advanced techniques"""
        start_time = datetime.now()

        self.logger.info(
            "Starting RAG query",
            question_length=len(question),
            esg_framework=esg_framework,
            search_strategy=search_strategy,
            k=k,
        )

        try:
            if use_query_decomposition and len(question) > 100:
                return await self._query_with_decomposition(
                    question, esg_framework, search_strategy, k, **kwargs
                )
            else:
                return await self._simple_query(
                    question, esg_framework, search_strategy, k, **kwargs
                )
        except ESGAnalysisException:
            raise
        except Exception as e:
            self.logger.error("RAG query failed", error=str(e))
            raise ESGAnalysisException("RAG query failed", error_code="RAG_QUERY_FAILED") from e
        finally:
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info("RAG query completed", total_time_ms=total_time)

    async def _simple_query(
        self,
        question: str,
        esg_framework: Optional[str] = None,
        search_strategy: str = "hybrid",
        k: int = 5,
        **kwargs: Any,
    ) -> RAGResponse:
        """Simple RAG query without decomposition"""

        # 1. Enhanced Query Processing
        enhanced_query = await self._enhance_query(question, esg_framework)

        # 2. Multi-stage Retrieval
        retrieval_start = datetime.now()
        search_results = await self._multi_stage_retrieval(
            enhanced_query, search_strategy, k, esg_framework
        )
        if not search_results:
            raise ESGAnalysisException(
                "No indexed documents matched this query",
                error_code="NO_RELEVANT_DOCUMENTS",
            )
        retrieval_time = (datetime.now() - retrieval_start).total_seconds() * 1000

        # 3. Context Preparation and Re-ranking
        context_documents = await self._prepare_context(search_results, question)

        # 4. Generation
        generation_start = datetime.now()
        llm_response = await self._generate_response(question, context_documents, esg_framework)
        generation_time = (datetime.now() - generation_start).total_seconds() * 1000

        # 5. Post-processing and Analysis
        confidence_score = self._calculate_confidence(search_results, llm_response)
        esg_categories = self._extract_esg_categories(context_documents)

        return RAGResponse(
            answer=llm_response.content,
            source_documents=context_documents,
            confidence_score=confidence_score,
            retrieval_time_ms=int(retrieval_time),
            generation_time_ms=int(generation_time),
            total_time_ms=int(retrieval_time + generation_time),
            esg_framework=esg_framework,
            esg_categories=esg_categories,
        )

    async def _query_with_decomposition(
        self,
        question: str,
        esg_framework: Optional[str] = None,
        search_strategy: str = "hybrid",
        k: int = 5,
        **kwargs: Any,
    ) -> RAGResponse:
        """RAG query with query decomposition for complex questions"""

        self.logger.info("Using query decomposition for complex question")

        # 1. Decompose the query
        decomposition = await self._decompose_query(question)

        # 2. Process sub-queries
        if decomposition.strategy == "sequential":
            sub_responses = await self._process_sequential_queries(
                decomposition, esg_framework, search_strategy, k
            )
        else:
            sub_responses = await self._process_parallel_queries(
                decomposition, esg_framework, search_strategy, k
            )

        # 3. Synthesize final response
        return await self._synthesize_responses(question, sub_responses, esg_framework)

    async def _enhance_query(self, question: str, esg_framework: Optional[str]) -> str:
        """Enhance query with ESG context and framework-specific terms"""

        enhanced_parts = [question]

        if esg_framework:
            enhanced_parts.append(f"Framework: {esg_framework}")

        # Add ESG context keywords
        esg_context = "ESG sustainability reporting compliance governance environmental social"
        enhanced_parts.append(esg_context)

        return " ".join(enhanced_parts)

    @traceable(name="multi_stage_retrieval")
    async def _multi_stage_retrieval(
        self, query: str, search_strategy: str, k: int, esg_framework: Optional[str]
    ) -> List[SearchResult]:
        """Multi-stage retrieval with filtering and re-ranking"""

        # Stage 1: Initial broad retrieval
        initial_k = min(k * 3, 20)  # Retrieve more initially

        filter_dict = {}
        if esg_framework:
            filter_dict["esg_framework"] = esg_framework

        initial_results = await vector_store_service.search(
            query=query,
            k=initial_k,
            search_type=search_strategy,
            filter_dict=filter_dict if filter_dict else None,
        )

        # Stage 2: Re-ranking based on query relevance
        reranked_results = await self._rerank_results(query, initial_results, k)

        self.logger.info(
            "Multi-stage retrieval completed",
            initial_results=len(initial_results),
            final_results=len(reranked_results),
        )

        return reranked_results

    async def _rerank_results(
        self, query: str, results: List[SearchResult], k: int
    ) -> List[SearchResult]:
        """Re-rank search results based on relevance"""

        if not settings.enable_reranking:
            for index, result in enumerate(results[:k], start=1):
                result.rank = index
            return results[:k]

        taxonomy_terms = {
            keyword.casefold()
            for keywords in get_taxonomy().categories.values()
            for keyword in keywords
        }

        def calculate_relevance_score(result: SearchResult) -> float:
            content = result.document.page_content.lower()
            query_lower = query.lower()

            # Base score from vector similarity
            base_score = result.score

            # Small taxonomy-aware lexical boost. This is configurable through
            # config/esg_taxonomy.json rather than embedded domain data.
            esg_boost = min(0.1, sum(0.01 for keyword in taxonomy_terms if keyword in content))

            # Boost for query term presence
            query_terms = query_lower.split()
            query_boost = min(0.15, sum(0.02 for term in query_terms if term in content))
            return min(1.0, base_score + esg_boost + query_boost)

        # Re-calculate scores and sort
        for result in results:
            result.score = calculate_relevance_score(result)

        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(results[:k], start=1):
            result.rank = i

        return results[:k]

    async def _prepare_context(
        self, search_results: List[SearchResult], question: str
    ) -> List[Document]:
        """Prepare and optimize context from search results"""

        context_docs = []
        total_length = 0
        max_context_length = settings.max_context_length

        for result in search_results:
            doc_length = len(result.document.page_content)

            if total_length + doc_length > max_context_length:
                break

            # Add metadata for citation
            enhanced_doc = Document(
                page_content=result.document.page_content,
                metadata={
                    **result.document.metadata,
                    "retrieval_score": result.score,
                    "retrieval_rank": result.rank,
                },
            )

            context_docs.append(enhanced_doc)
            total_length += doc_length

        self.logger.info(
            "Context prepared", num_documents=len(context_docs), total_length=total_length
        )

        return context_docs

    @traceable(name="generate_response")
    async def _generate_response(
        self, question: str, context_documents: List[Document], esg_framework: Optional[str]
    ) -> LLMResponse:
        """Generate response using LLM with prepared context"""

        # Prepare context string with citations
        context_parts = []
        for i, doc in enumerate(context_documents, 1):
            source_info = f"Source {i}"
            if doc.metadata.get("filename"):
                source_info += f" ({doc.metadata['filename']})"
            if doc.metadata.get("page"):
                source_info += f", Page {doc.metadata['page']}"

            context_parts.append(f"[{source_info}]\n{doc.page_content}")

        context_string = "\n\n".join(context_parts)

        # Build framework-specific prompt
        framework_context = ""
        if esg_framework:
            framework_context = (
                f"\nSpecific Focus: {esg_framework} framework compliance and requirements."
            )

        prompt = f"""{self.esg_system_prompt}{framework_context}

Context Documents:
{context_string}

Question: {question}

Please provide a comprehensive answer based on the context documents. Include specific citations (e.g., [Source 1]) for all claims and recommendations."""

        return await llm_service.generate(prompt)

    async def _decompose_query(self, question: str) -> QueryDecomposition:
        """Decompose complex query into sub-queries"""

        response = await llm_service.generate(
            prompt=self.query_decomposition_prompt.format(query=question)
        )

        try:
            import json

            decomposition_data = json.loads(response.content)

            return QueryDecomposition(
                main_query=decomposition_data["main_query"],
                sub_queries=decomposition_data["sub_queries"],
                strategy=decomposition_data["strategy"],
            )
        except Exception as e:
            self.logger.error("Failed to parse query decomposition", error=str(e))
            # Fallback to simple decomposition
            return QueryDecomposition(
                main_query=question, sub_queries=[question], strategy="parallel"
            )

    async def _process_parallel_queries(
        self,
        decomposition: QueryDecomposition,
        esg_framework: Optional[str],
        search_strategy: str,
        k: int,
    ) -> List[RAGResponse]:
        """Process sub-queries in parallel"""

        tasks = [
            self._simple_query(
                sub_query,
                esg_framework,
                search_strategy,
                max(1, k // len(decomposition.sub_queries)),
            )
            for sub_query in decomposition.sub_queries
        ]

        return await asyncio.gather(*tasks)

    async def _process_sequential_queries(
        self,
        decomposition: QueryDecomposition,
        esg_framework: Optional[str],
        search_strategy: str,
        k: int,
    ) -> List[RAGResponse]:
        """Process sub-queries sequentially, building context"""

        responses = []
        accumulated_context = ""

        for sub_query in decomposition.sub_queries:
            # Add accumulated context to query
            enhanced_query = f"{sub_query}\n\nPrevious context:\n{accumulated_context}"

            response = await self._simple_query(
                enhanced_query,
                esg_framework,
                search_strategy,
                max(1, k // len(decomposition.sub_queries)),
            )

            responses.append(response)
            accumulated_context += f"\n{response.answer}"

        return responses

    async def _synthesize_responses(
        self, original_question: str, sub_responses: List[RAGResponse], esg_framework: Optional[str]
    ) -> RAGResponse:
        """Synthesize multiple sub-responses into final answer"""

        # Combine all source documents
        all_source_docs = []
        for response in sub_responses:
            all_source_docs.extend(response.source_documents)

        # Create synthesis prompt
        sub_answers = [f"Sub-answer {i + 1}: {resp.answer}" for i, resp in enumerate(sub_responses)]
        sub_answers_text = "\n\n".join(sub_answers)

        synthesis_prompt = f"""Based on the following sub-answers to related questions, provide a comprehensive and coherent answer to the original question.

Original Question: {original_question}

Sub-answers:
{sub_answers_text}

Please synthesize these into a single, well-structured response that addresses the original question comprehensively. Maintain all citations and ensure logical flow."""

        synthesis_response = await llm_service.generate(synthesis_prompt)

        # Calculate aggregated metrics
        total_retrieval_time = sum(resp.retrieval_time_ms for resp in sub_responses)
        total_generation_time = sum(resp.generation_time_ms for resp in sub_responses)
        avg_confidence = sum(resp.confidence_score for resp in sub_responses) / len(sub_responses)

        # Combine ESG categories
        all_categories = []
        for response in sub_responses:
            if response.esg_categories:
                all_categories.extend(response.esg_categories)
        unique_categories = list(set(all_categories))

        return RAGResponse(
            answer=synthesis_response.content,
            source_documents=all_source_docs,
            confidence_score=avg_confidence,
            retrieval_time_ms=total_retrieval_time,
            generation_time_ms=total_generation_time,
            total_time_ms=total_retrieval_time + total_generation_time,
            esg_framework=esg_framework,
            esg_categories=unique_categories,
        )

    def _calculate_confidence(
        self, search_results: List[SearchResult], llm_response: LLMResponse
    ) -> float:
        """Calculate confidence score for the response"""

        # Base confidence from search result scores
        if not search_results:
            return 0.0

        avg_search_score = sum(result.score for result in search_results) / len(search_results)

        citation_factor = 1.0 if "[Source" in llm_response.content else 0.8
        confidence = min(1.0, max(0.0, avg_search_score * citation_factor))
        return round(confidence, 2)

    def _extract_esg_categories(self, documents: List[Document]) -> List[str]:
        """Extract ESG categories from documents"""
        categories = set()

        for doc in documents:
            if doc.metadata.get("esg_category"):
                categories.add(doc.metadata["esg_category"])

        return list(categories)

    async def stream_query(
        self,
        question: str,
        esg_framework: Optional[str] = None,
        search_strategy: str = "hybrid",
        k: int = 5,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream RAG response for real-time UI updates"""

        try:
            # First, get context
            enhanced_query = await self._enhance_query(question, esg_framework)
            search_results = await self._multi_stage_retrieval(
                enhanced_query, search_strategy, k, esg_framework
            )
            context_documents = await self._prepare_context(search_results, question)

            # Prepare prompt
            context_parts = []
            for i, doc in enumerate(context_documents, 1):
                source_info = f"Source {i}"
                if doc.metadata.get("filename"):
                    source_info += f" ({doc.metadata['filename']})"
                context_parts.append(f"[{source_info}]\n{doc.page_content}")

            context_string = "\n\n".join(context_parts)

            framework_context = ""
            if esg_framework:
                framework_context = (
                    f"\nSpecific Focus: {esg_framework} framework compliance and requirements."
                )

            prompt = f"""{self.esg_system_prompt}{framework_context}

Context Documents:
{context_string}

Question: {question}

Please provide a comprehensive answer based on the context documents. Include specific citations (e.g., [Source 1]) for all claims and recommendations."""

            # Stream response
            async for chunk in llm_service.stream(prompt):
                yield chunk

        except Exception as e:
            self.logger.error("Streaming query failed", error=str(e))
            raise


# Global RAG service instance
rag_service = AdvancedRAGService()
