"""Analytics derived from actual query events and indexed documents."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query

from ...core.taxonomy import get_taxonomy
from ...services.analytics_service import analytics_service
from ...services.vector_store_service import vector_store_service
from ..models import AnalyticsResponse, FrameworkStats, UsageStats


router = APIRouter()


def _period(
    start_date: datetime | None,
    end_date: datetime | None,
) -> tuple[datetime, datetime]:
    end = end_date or datetime.now(UTC)
    start = start_date or end - timedelta(days=30)
    if start > end:
        raise ValueError("start_date must be before end_date")
    return start, end


@router.get("/usage", response_model=AnalyticsResponse)
async def get_usage_analytics(
    start_date: datetime | None = Query(default=None),
    end_date: datetime | None = Query(default=None),
    framework: str | None = Query(default=None),
) -> AnalyticsResponse:
    start, end = _period(start_date, end_date)
    events = analytics_service.events_between(start, end)
    if framework:
        events = [event for event in events if event.framework == framework]

    stats = await vector_store_service.stats()
    framework_events: dict[str, list] = defaultdict(list)
    for event in events:
        if event.framework:
            framework_events[event.framework].append(event)

    framework_ids = [framework] if framework else get_taxonomy().framework_ids
    framework_stats = []
    for framework_id in framework_ids:
        matching = framework_events.get(framework_id, [])
        framework_stats.append(
            FrameworkStats(
                framework=framework_id,
                query_count=len(matching),
                document_count=stats["documents_by_framework"].get(framework_id, 0),
                average_confidence=(
                    sum(event.confidence for event in matching) / len(matching) if matching else 0.0
                ),
            )
        )

    return AnalyticsResponse(
        usage_stats=UsageStats(
            total_queries=len(events),
            total_documents=stats["total_documents"],
            average_response_time_ms=(
                sum(event.duration_ms for event in events) / len(events) if events else 0.0
            ),
            most_used_framework=analytics_service.most_common_framework(events),
            most_queried_category=analytics_service.most_common_category(events),
        ),
        framework_stats=framework_stats,
        period_start=start,
        period_end=end,
    )


@router.get("/metrics")
async def get_system_metrics() -> dict:
    now = datetime.now(UTC)
    events = analytics_service.events_between(analytics_service.started_at, now)
    return {
        "uptime_seconds": max(0, int((now - analytics_service.started_at).total_seconds())),
        "queries_since_start": len(events),
        "average_query_latency_ms": (
            sum(event.duration_ms for event in events) / len(events) if events else 0.0
        ),
        "indexed_documents": (await vector_store_service.stats())["total_documents"],
    }


@router.get("/top-queries")
async def get_top_queries(limit: int = Query(default=10, ge=1, le=100)) -> dict:
    end = datetime.now(UTC)
    start = end - timedelta(days=30)
    events = analytics_service.events_between(start, end)
    return {
        "top_queries": analytics_service.top_queries(events, limit),
        "period_start": start,
        "period_end": end,
        "total_unique_queries": len({event.question for event in events}),
    }


@router.get("/framework-adoption")
async def get_framework_adoption() -> dict:
    stats = await vector_store_service.stats()
    events = analytics_service.events_between(analytics_service.started_at, datetime.now(UTC))
    query_counts = defaultdict(int)
    for event in events:
        if event.framework:
            query_counts[event.framework] += 1
    return {
        "framework_usage": {
            framework: {
                "queries": query_counts[framework],
                "documents": stats["documents_by_framework"].get(framework, 0),
            }
            for framework in get_taxonomy().framework_ids
        }
    }
