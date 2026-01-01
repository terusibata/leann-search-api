"""Test configuration and fixtures."""

import os
import shutil
import tempfile
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

# Set test environment
os.environ["INDEX_DIR"] = tempfile.mkdtemp()
os.environ["EMBEDDING_MODEL"] = "cl-nagoya/ruri-v3-310m"


@pytest.fixture(scope="session")
def test_index_dir() -> Generator[str, None, None]:
    """Create a temporary index directory for tests."""
    temp_dir = tempfile.mkdtemp()
    os.environ["INDEX_DIR"] = temp_dir
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client(test_index_dir: str) -> Generator[TestClient, None, None]:
    """Create a test client."""
    # Import after setting environment
    from src.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_index_name() -> str:
    """Sample index name for tests."""
    return "test_index"


@pytest.fixture
def sample_document() -> dict:
    """Sample document for tests."""
    return {
        "id": "doc_001",
        "content": "経費精算の申請期限は経費発生日から1ヶ月以内です。申請方法については社内ポータルをご確認ください。",
        "metadata": {
            "category": "manual",
            "department": "経理部",
            "author": "山田太郎",
            "version": 1,
            "is_public": True,
        },
    }


@pytest.fixture
def created_index(client: TestClient, sample_index_name: str) -> Generator[str, None, None]:
    """Create an index for tests and clean up after."""
    # Create index
    response = client.post(
        "/api/v1/indexes", json={"name": sample_index_name}
    )
    assert response.status_code == 200

    yield sample_index_name

    # Cleanup
    client.delete(f"/api/v1/indexes/{sample_index_name}")
