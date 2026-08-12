"""Authenticated administration and evidence-grounded report generation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...core.config import settings
from ...core.taxonomy import get_taxonomy
from ...services.agentic_rag_service import agentic_rag_service
from ...services.llm_service import llm_service
from ...services.rag_service import rag_service
from ...services.vector_store_service import vector_store_service
from ..auth import verify_admin_token


router = APIRouter(dependencies=[Depends(verify_admin_token)])


class ReportRequest(BaseModel):
    esg_framework: str | None = Field(default=None, max_length=80)
    report_type: Literal["compliance_summary", "framework_analysis", "gap_analysis", "general"] = (
        "compliance_summary"
    )
    include_recommendations: bool = True
    use_agentic_rag: bool = True


def _validate_framework(framework: str | None) -> None:
    if framework and framework not in get_taxonomy().framework_ids:
        raise HTTPException(status_code=422, detail="Unknown ESG framework")


def _report_prompt(request: ReportRequest, stats: dict) -> str:
    focus = request.esg_framework or "all configured ESG frameworks"
    recommendation_instruction = (
        "Include prioritized recommendations that explicitly cite the supporting document evidence."
        if request.include_recommendations
        else "Do not add a recommendations section."
    )
    return f"""Create an ESG {request.report_type.replace("_", " ")} for {focus}.

Use only claims supported by retrieved repository documents. Distinguish missing
evidence from non-compliance; absence in this repository does not prove that the
organization has not acted. Cite source filenames and identify uncertainty.

Repository inventory:
- Documents: {stats["total_documents"]}
- Chunks: {stats["total_chunks"]}
- Framework coverage: {stats["documents_by_framework"]}
- Category coverage: {stats["documents_by_category"]}

{recommendation_instruction}
"""


@router.get("/system-info")
async def get_system_info() -> dict:
    taxonomy = get_taxonomy()
    return {
        "version": settings.api_version,
        "environment": settings.environment,
        "configured_llm_providers": llm_service.list_providers(),
        "vector_store_provider": settings.vector_store_type,
        "supported_frameworks": taxonomy.framework_ids,
        "supported_file_types": list(settings.supported_extensions),
        "max_document_size_mb": settings.max_document_size_mb,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "default_model": settings.default_model,
    }


@router.post("/generate-report")
async def generate_esg_report(request: ReportRequest) -> dict:
    _validate_framework(request.esg_framework)
    stats = await vector_store_service.stats()
    if not stats["total_documents"]:
        raise HTTPException(
            status_code=409, detail="Upload source documents before generating a report"
        )

    generated_at = datetime.now(UTC)
    if request.use_agentic_rag:
        response = await agentic_rag_service.generate_report(
            report_type=request.report_type,
            framework=request.esg_framework,
            available_data=stats,
        )
        content = response.report_content
        metadata = {
            "agentic_rag": True,
            "quality_score": response.quality_score,
            "sections_generated": response.sections_generated,
            "confidence_scores": response.confidence_scores,
            "research_findings": len(response.research_findings),
            "validation_status": response.validation_results.get("overall_status"),
            "generation_time_ms": response.total_time_ms,
            "agent_phases": [log["phase"] for log in response.agent_execution_log],
        }
    else:
        response = await rag_service.query(
            question=_report_prompt(request, stats),
            esg_framework=request.esg_framework,
            search_strategy="hybrid",
            k=min(20, max(5, stats["total_chunks"])),
            use_query_decomposition=True,
        )
        content = response.answer
        metadata = {
            "agentic_rag": False,
            "confidence_score": response.confidence_score,
            "generation_time_ms": response.total_time_ms,
            "sources_used": len(response.source_documents),
        }

    metadata.update(
        {
            "total_documents": stats["total_documents"],
            "total_chunks": stats["total_chunks"],
            "frameworks_covered": list(stats["documents_by_framework"]),
            "evidence_grounded": True,
        }
    )
    return {
        "report_id": f"report_{generated_at.strftime('%Y%m%d_%H%M%S_%f')}",
        "generated_at": generated_at.isoformat(),
        "framework": request.esg_framework,
        "report_type": request.report_type,
        "content": content,
        "metadata": metadata,
    }


@router.post("/generate-report/stream")
async def stream_esg_report(request: ReportRequest) -> StreamingResponse:
    _validate_framework(request.esg_framework)

    async def generate():
        try:
            stats = await vector_store_service.stats()
            if not stats["total_documents"]:
                yield f"event: error\ndata: {json.dumps({'status': 'error', 'message': 'Upload source documents before generating a report'})}\n\n"
                return
            async for update in agentic_rag_service.stream_report_generation(
                report_type=request.report_type,
                framework=request.esg_framework,
                available_data=stats,
            ):
                yield f"event: update\ndata: {json.dumps(update, default=str)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'status': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
