"""Small in-process analytics backend with an explicit replaceable interface."""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class QueryEvent:
    timestamp: datetime
    question: str
    framework: str | None
    categories: tuple[str, ...]
    duration_ms: int
    confidence: float


class AnalyticsService:
    """Bounded process-local metrics.

    This intentionally exposes a small interface so deployments can replace it
    with a durable database or telemetry sink without changing API routes.
    """

    def __init__(self, max_events: int = 10_000) -> None:
        self.started_at = datetime.now(UTC)
        self._events: deque[QueryEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def record_query(
        self,
        *,
        question: str,
        framework: str | None,
        categories: list[str],
        duration_ms: int,
        confidence: float,
    ) -> None:
        event = QueryEvent(
            timestamp=datetime.now(UTC),
            question=question,
            framework=framework,
            categories=tuple(categories),
            duration_ms=duration_ms,
            confidence=confidence,
        )
        with self._lock:
            self._events.append(event)

    def events_between(self, start: datetime, end: datetime) -> list[QueryEvent]:
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        with self._lock:
            return [event for event in self._events if start <= event.timestamp <= end]

    @staticmethod
    def most_common_framework(events: list[QueryEvent]) -> str | None:
        counts = Counter(event.framework for event in events if event.framework)
        return counts.most_common(1)[0][0] if counts else None

    @staticmethod
    def most_common_category(events: list[QueryEvent]) -> str | None:
        counts = Counter(category for event in events for category in event.categories)
        return counts.most_common(1)[0][0] if counts else None

    @staticmethod
    def top_queries(events: list[QueryEvent], limit: int) -> list[dict[str, int | str]]:
        return [
            {"query": query, "count": count}
            for query, count in Counter(event.question for event in events).most_common(limit)
        ]


analytics_service = AnalyticsService()
