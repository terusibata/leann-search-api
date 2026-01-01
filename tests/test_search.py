"""Tests for search API."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def index_with_documents(
    client: TestClient, created_index: str
) -> str:
    """Create an index with sample documents for search tests."""
    documents = [
        {
            "id": "doc_expense",
            "content": "経費精算の申請期限は経費発生日から1ヶ月以内です。申請方法については社内ポータルをご確認ください。",
            "metadata": {
                "category": "manual",
                "department": "経理部",
                "is_public": True,
            },
        },
        {
            "id": "doc_vacation",
            "content": "有給休暇の申請は3日前までに行ってください。緊急の場合は当日申請も可能です。",
            "metadata": {
                "category": "manual",
                "department": "人事部",
                "is_public": True,
            },
        },
        {
            "id": "doc_meeting",
            "content": "会議室の予約はグループウェアから行います。会議室A、B、Cが利用可能です。",
            "metadata": {
                "category": "guide",
                "department": "総務部",
                "is_public": False,
            },
        },
        {
            "id": "doc_error",
            "content": "ERROR_CODE_001: Connection timeout. Please check your network settings.",
            "metadata": {
                "category": "error_log",
                "department": "IT",
                "is_public": False,
            },
        },
    ]

    client.post(
        f"/api/v1/indexes/{created_index}/documents",
        json={"documents": documents},
    )

    return created_index


class TestSemanticSearch:
    """Tests for semantic search."""

    def test_search_basic(self, client: TestClient, index_with_documents: str):
        """Test basic semantic search."""
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search",
            json={"query": "経費精算の期限", "top_k": 5},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "results" in data["data"]
        assert "total_found" in data["data"]
        assert "query_time_ms" in data["data"]

    def test_search_with_metadata_filter(
        self, client: TestClient, index_with_documents: str
    ):
        """Test search with metadata filter."""
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search",
            json={
                "query": "申請方法",
                "top_k": 5,
                "metadata_filters": {"department": {"in": ["経理部", "人事部"]}},
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

        for result in data["data"]["results"]:
            assert result["metadata"]["department"] in ["経理部", "人事部"]

    def test_search_with_options(self, client: TestClient, index_with_documents: str):
        """Test search with custom options."""
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search",
            json={
                "query": "経費",
                "top_k": 3,
                "options": {
                    "include_content": True,
                    "include_metadata": True,
                    "min_score": 0.0,
                },
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

        if data["data"]["results"]:
            result = data["data"]["results"][0]
            assert "content" in result
            assert "metadata" in result
            assert "score" in result

    def test_search_empty_results(self, client: TestClient, index_with_documents: str):
        """Test search with no matching results."""
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search",
            json={
                "query": "完全に関係のないクエリ12345",
                "top_k": 5,
                "options": {"min_score": 0.9},
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_search_nonexistent_index(self, client: TestClient):
        """Test search on non-existent index."""
        response = client.post(
            "/api/v1/indexes/nonexistent/search",
            json={"query": "test"},
        )
        assert response.status_code == 404


class TestGrepSearch:
    """Tests for grep/keyword search."""

    def test_grep_search_basic(self, client: TestClient, index_with_documents: str):
        """Test basic grep search."""
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search/grep",
            json={"query": "ERROR_CODE_001", "top_k": 10},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["results"]) > 0
        assert "match_positions" in data["data"]["results"][0]

    def test_grep_search_case_insensitive(
        self, client: TestClient, index_with_documents: str
    ):
        """Test grep search is case insensitive."""
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search/grep",
            json={"query": "error_code_001", "top_k": 10},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["results"]) > 0

    def test_grep_search_japanese(self, client: TestClient, index_with_documents: str):
        """Test grep search with Japanese text."""
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search/grep",
            json={"query": "経費精算", "top_k": 10},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["results"]) > 0

    def test_grep_search_with_filter(
        self, client: TestClient, index_with_documents: str
    ):
        """Test grep search with metadata filter."""
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search/grep",
            json={
                "query": "申請",
                "top_k": 10,
                "metadata_filters": {"category": {"==": "manual"}},
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

        for result in data["data"]["results"]:
            assert result["metadata"]["category"] == "manual"


class TestHybridSearch:
    """Tests for hybrid search."""

    def test_hybrid_search_basic(self, client: TestClient, index_with_documents: str):
        """Test basic hybrid search."""
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search/hybrid",
            json={
                "query": "経費精算 申請",
                "top_k": 5,
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "results" in data["data"]

    def test_hybrid_search_custom_weights(
        self, client: TestClient, index_with_documents: str
    ):
        """Test hybrid search with custom weights."""
        # Keyword-heavy search
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search/hybrid",
            json={
                "query": "ERROR_CODE_001",
                "top_k": 5,
                "semantic_weight": 0.2,
                "keyword_weight": 0.8,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True


class TestBatchSearch:
    """Tests for batch search."""

    def test_batch_search_basic(self, client: TestClient, index_with_documents: str):
        """Test basic batch search."""
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search/batch",
            json={
                "queries": [
                    {"id": "q1", "query": "経費精算", "top_k": 3},
                    {"id": "q2", "query": "有給休暇", "top_k": 3},
                    {"id": "q3", "query": "会議室予約", "top_k": 3},
                ]
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "results" in data["data"]
        assert "q1" in data["data"]["results"]
        assert "q2" in data["data"]["results"]
        assert "q3" in data["data"]["results"]
        assert "total_query_time_ms" in data["data"]

    def test_batch_search_with_filter(
        self, client: TestClient, index_with_documents: str
    ):
        """Test batch search with shared metadata filter."""
        response = client.post(
            f"/api/v1/indexes/{index_with_documents}/search/batch",
            json={
                "queries": [
                    {"id": "q1", "query": "申請方法", "top_k": 3},
                    {"id": "q2", "query": "期限", "top_k": 3},
                ],
                "metadata_filters": {"is_public": {"is_true": True}},
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

        for query_id, query_results in data["data"]["results"].items():
            for result in query_results["results"]:
                assert result["metadata"]["is_public"] is True
