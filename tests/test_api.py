from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app
from src.services.vector_store_service import MemoryVectorStore, vector_store_service


def reset_store() -> None:
    vector_store_service._stores["memory"] = MemoryVectorStore()


def test_health_and_taxonomy_are_available_without_llm():
    reset_store()
    with TestClient(app) as client:
        health = client.get("/health")
        root = client.get("/")

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["services"]["vector_store"] == "healthy"
    assert "CSRD" in root.json()["supported_frameworks"]


def test_upload_list_stats_detail_search_and_delete_roundtrip():
    reset_store()
    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/upload",
            data={"esg_framework": "CSRD", "document_type": "policy"},
            files={
                "file": (
                    "climate.txt",
                    b"Carbon emissions, climate risk, and renewable energy evidence.",
                    "text/plain",
                )
            },
        )
        assert upload.status_code == 201
        document_hash = upload.json()["document_id"]

        listing = client.get("/api/v1/documents/list")
        stats = client.get("/api/v1/documents/stats")
        detail = client.get(f"/api/v1/documents/document/{document_hash}")
        search = client.post(
            "/api/v1/documents/search",
            json={"query": "renewable energy", "k": 5},
        )
        deleted = client.delete(f"/api/v1/documents/{document_hash}")

    assert listing.json()["total"] == 1
    assert stats.json()["documents_by_framework"] == {"CSRD": 1}
    assert detail.json()["total_chunks"] >= 1
    assert search.json()["total_results"] >= 1
    assert deleted.status_code == 200


def test_admin_routes_require_configured_token():
    reset_store()
    with TestClient(app) as client:
        unauthorized = client.get("/api/v1/admin/system-info")
        authorized = client.get(
            "/api/v1/admin/system-info",
            headers={"Authorization": "Bearer test-admin-token-with-safe-length"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
