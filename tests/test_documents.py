"""Tests for document management API."""

import pytest
from fastapi.testclient import TestClient


class TestAddDocuments:
    """Tests for adding documents."""

    def test_add_single_document(
        self, client: TestClient, created_index: str, sample_document: dict
    ):
        """Test adding a single document."""
        response = client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": [sample_document]},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["added"] == 1
        assert data["data"]["updated"] == 0
        assert data["data"]["failed"] == 0
        assert len(data["data"]["documents"]) == 1
        assert data["data"]["documents"][0]["id"] == sample_document["id"]
        assert data["data"]["documents"][0]["status"] == "added"
        assert data["data"]["documents"][0]["chunk_count"] > 0

    def test_add_multiple_documents(self, client: TestClient, created_index: str):
        """Test adding multiple documents."""
        documents = [
            {
                "id": f"doc_{i}",
                "content": f"これはテストドキュメント {i} です。",
                "metadata": {"index": i},
            }
            for i in range(5)
        ]

        response = client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": documents},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["added"] == 5

    def test_add_document_update_if_exists(
        self, client: TestClient, created_index: str, sample_document: dict
    ):
        """Test updating existing document."""
        # Add document first
        client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": [sample_document]},
        )

        # Try to add again with update_if_exists
        updated_doc = sample_document.copy()
        updated_doc["content"] = "更新された内容です。"

        response = client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={
                "documents": [updated_doc],
                "options": {"update_if_exists": True},
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["updated"] == 1

    def test_add_document_to_nonexistent_index(
        self, client: TestClient, sample_document: dict
    ):
        """Test adding document to non-existent index."""
        response = client.post(
            "/api/v1/indexes/nonexistent/documents",
            json={"documents": [sample_document]},
        )
        assert response.status_code == 404


class TestListDocuments:
    """Tests for listing documents."""

    def test_list_documents_empty(self, client: TestClient, created_index: str):
        """Test listing documents when none exist."""
        response = client.get(f"/api/v1/indexes/{created_index}/documents")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["documents"] == []
        assert data["data"]["pagination"]["total"] == 0

    def test_list_documents_with_data(
        self, client: TestClient, created_index: str, sample_document: dict
    ):
        """Test listing documents with existing documents."""
        # Add a document first
        client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": [sample_document]},
        )

        response = client.get(f"/api/v1/indexes/{created_index}/documents")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["documents"]) > 0
        assert data["data"]["pagination"]["total"] > 0

    def test_list_documents_pagination(self, client: TestClient, created_index: str):
        """Test document listing pagination."""
        # Add multiple documents
        documents = [
            {"id": f"doc_{i}", "content": f"Document {i}"} for i in range(10)
        ]
        client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": documents},
        )

        # Test pagination
        response = client.get(
            f"/api/v1/indexes/{created_index}/documents",
            params={"page": 1, "per_page": 5},
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]["documents"]) == 5
        assert data["data"]["pagination"]["page"] == 1
        assert data["data"]["pagination"]["per_page"] == 5
        assert data["data"]["pagination"]["total"] == 10
        assert data["data"]["pagination"]["total_pages"] == 2


class TestGetDocument:
    """Tests for getting document details."""

    def test_get_document_success(
        self, client: TestClient, created_index: str, sample_document: dict
    ):
        """Test getting existing document."""
        # Add a document first
        client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": [sample_document]},
        )

        response = client.get(
            f"/api/v1/indexes/{created_index}/documents/{sample_document['id']}"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == sample_document["id"]
        assert data["data"]["content"] == sample_document["content"]
        assert data["data"]["metadata"] == sample_document["metadata"]
        assert "chunks" in data["data"]

    def test_get_document_not_found(self, client: TestClient, created_index: str):
        """Test getting non-existent document."""
        response = client.get(f"/api/v1/indexes/{created_index}/documents/nonexistent")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"]["error"]["code"] == "DOCUMENT_NOT_FOUND"


class TestUpdateDocument:
    """Tests for updating documents."""

    def test_update_document_content(
        self, client: TestClient, created_index: str, sample_document: dict
    ):
        """Test updating document content."""
        # Add a document first
        client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": [sample_document]},
        )

        new_content = "更新された内容です。"
        response = client.put(
            f"/api/v1/indexes/{created_index}/documents/{sample_document['id']}",
            json={"content": new_content},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "updated"

        # Verify update
        get_response = client.get(
            f"/api/v1/indexes/{created_index}/documents/{sample_document['id']}"
        )
        assert get_response.json()["data"]["content"] == new_content

    def test_update_document_metadata(
        self, client: TestClient, created_index: str, sample_document: dict
    ):
        """Test updating document metadata."""
        # Add a document first
        client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": [sample_document]},
        )

        response = client.put(
            f"/api/v1/indexes/{created_index}/documents/{sample_document['id']}",
            json={"metadata": {"version": 2, "reviewed": True}},
        )
        assert response.status_code == 200

        # Verify metadata merge
        get_response = client.get(
            f"/api/v1/indexes/{created_index}/documents/{sample_document['id']}"
        )
        metadata = get_response.json()["data"]["metadata"]
        assert metadata["version"] == 2
        assert metadata["reviewed"] is True
        assert metadata["category"] == "manual"  # Original field preserved

    def test_update_document_not_found(self, client: TestClient, created_index: str):
        """Test updating non-existent document."""
        response = client.put(
            f"/api/v1/indexes/{created_index}/documents/nonexistent",
            json={"content": "test"},
        )
        assert response.status_code == 404


class TestUpdateMetadata:
    """Tests for updating document metadata only."""

    def test_update_metadata_merge(
        self, client: TestClient, created_index: str, sample_document: dict
    ):
        """Test updating metadata with merge."""
        # Add a document first
        client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": [sample_document]},
        )

        response = client.patch(
            f"/api/v1/indexes/{created_index}/documents/{sample_document['id']}/metadata",
            json={"metadata": {"new_field": "value"}, "merge": True},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["metadata"]["new_field"] == "value"
        assert data["data"]["metadata"]["category"] == "manual"

    def test_update_metadata_replace(
        self, client: TestClient, created_index: str, sample_document: dict
    ):
        """Test updating metadata without merge."""
        # Add a document first
        client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": [sample_document]},
        )

        response = client.patch(
            f"/api/v1/indexes/{created_index}/documents/{sample_document['id']}/metadata",
            json={"metadata": {"only_field": "value"}, "merge": False},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["metadata"] == {"only_field": "value"}


class TestDeleteDocument:
    """Tests for deleting documents."""

    def test_delete_document_success(
        self, client: TestClient, created_index: str, sample_document: dict
    ):
        """Test successful document deletion."""
        # Add a document first
        client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": [sample_document]},
        )

        response = client.delete(
            f"/api/v1/indexes/{created_index}/documents/{sample_document['id']}"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["data"]["message"]

        # Verify it's gone
        get_response = client.get(
            f"/api/v1/indexes/{created_index}/documents/{sample_document['id']}"
        )
        assert get_response.status_code == 404

    def test_delete_document_not_found(self, client: TestClient, created_index: str):
        """Test deleting non-existent document."""
        response = client.delete(
            f"/api/v1/indexes/{created_index}/documents/nonexistent"
        )
        assert response.status_code == 404


class TestBulkDelete:
    """Tests for bulk document deletion."""

    def test_bulk_delete_by_ids(self, client: TestClient, created_index: str):
        """Test bulk delete by document IDs."""
        # Add documents first
        documents = [{"id": f"doc_{i}", "content": f"Document {i}"} for i in range(5)]
        client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": documents},
        )

        response = client.post(
            f"/api/v1/indexes/{created_index}/documents/bulk-delete",
            json={"document_ids": ["doc_0", "doc_1", "doc_2"]},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["deleted"] == 3

    def test_bulk_delete_by_filter(self, client: TestClient, created_index: str):
        """Test bulk delete by metadata filter."""
        # Add documents first
        documents = [
            {"id": f"doc_{i}", "content": f"Document {i}", "metadata": {"category": "a" if i < 3 else "b"}}
            for i in range(5)
        ]
        client.post(
            f"/api/v1/indexes/{created_index}/documents",
            json={"documents": documents},
        )

        response = client.post(
            f"/api/v1/indexes/{created_index}/documents/bulk-delete",
            json={"metadata_filter": {"category": {"==": "a"}}},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["deleted"] == 3
