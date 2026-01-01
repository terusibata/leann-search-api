"""Tests for index management API."""

import pytest
from fastapi.testclient import TestClient


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client: TestClient):
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "leann_version" in data
        assert "embedding_model" in data


class TestListIndexes:
    """Tests for listing indexes."""

    def test_list_indexes_empty(self, client: TestClient):
        """Test listing indexes when none exist."""
        response = client.get("/api/v1/indexes")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "indexes" in data["data"]
        assert isinstance(data["data"]["indexes"], list)

    def test_list_indexes_with_data(self, client: TestClient, created_index: str):
        """Test listing indexes with existing indexes."""
        response = client.get("/api/v1/indexes")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["indexes"]) > 0

        index_names = [idx["name"] for idx in data["data"]["indexes"]]
        assert created_index in index_names


class TestCreateIndex:
    """Tests for creating indexes."""

    def test_create_index_success(self, client: TestClient):
        """Test successful index creation."""
        index_name = "new_test_index"

        response = client.post("/api/v1/indexes", json={"name": index_name})
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == index_name
        assert data["data"]["status"] == "empty"
        assert "settings" in data["data"]
        assert "created_at" in data["data"]

        # Cleanup
        client.delete(f"/api/v1/indexes/{index_name}")

    def test_create_index_with_settings(self, client: TestClient):
        """Test index creation with custom settings."""
        index_name = "custom_settings_index"

        response = client.post(
            "/api/v1/indexes",
            json={
                "name": index_name,
                "settings": {
                    "backend": "hnsw",
                    "graph_degree": 48,
                    "build_complexity": 128,
                    "chunk_size": 256,
                    "chunk_overlap": 32,
                },
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["settings"]["graph_degree"] == 48
        assert data["data"]["settings"]["build_complexity"] == 128
        assert data["data"]["settings"]["chunk_size"] == 256

        # Cleanup
        client.delete(f"/api/v1/indexes/{index_name}")

    def test_create_index_duplicate(self, client: TestClient, created_index: str):
        """Test creating a duplicate index fails."""
        response = client.post("/api/v1/indexes", json={"name": created_index})
        assert response.status_code == 409

    def test_create_index_invalid_name(self, client: TestClient):
        """Test creating index with invalid name."""
        response = client.post("/api/v1/indexes", json={"name": "123invalid"})
        assert response.status_code == 400


class TestGetIndex:
    """Tests for getting index details."""

    def test_get_index_success(self, client: TestClient, created_index: str):
        """Test getting existing index details."""
        response = client.get(f"/api/v1/indexes/{created_index}")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == created_index
        assert "settings" in data["data"]
        assert "statistics" in data["data"]

    def test_get_index_not_found(self, client: TestClient):
        """Test getting non-existent index."""
        response = client.get("/api/v1/indexes/nonexistent_index")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"]["error"]["code"] == "INDEX_NOT_FOUND"


class TestDeleteIndex:
    """Tests for deleting indexes."""

    def test_delete_index_success(self, client: TestClient):
        """Test successful index deletion."""
        # Create an index first
        index_name = "to_delete_index"
        client.post("/api/v1/indexes", json={"name": index_name})

        # Delete it
        response = client.delete(f"/api/v1/indexes/{index_name}")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["data"]["message"]

        # Verify it's gone
        response = client.get(f"/api/v1/indexes/{index_name}")
        assert response.status_code == 404

    def test_delete_index_not_found(self, client: TestClient):
        """Test deleting non-existent index."""
        response = client.delete("/api/v1/indexes/nonexistent_index")
        assert response.status_code == 404


class TestRebuildIndex:
    """Tests for rebuilding indexes."""

    def test_rebuild_index_success(self, client: TestClient, created_index: str):
        """Test successful index rebuild."""
        response = client.post(f"/api/v1/indexes/{created_index}/rebuild")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "message" in data["data"]
        assert "chunk_count" in data["data"]
        assert "build_time_ms" in data["data"]

    def test_rebuild_index_not_found(self, client: TestClient):
        """Test rebuilding non-existent index."""
        response = client.post("/api/v1/indexes/nonexistent_index/rebuild")
        assert response.status_code == 404
