"""Index management service."""

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.config import get_settings
from src.schemas.index import (
    IndexDetailInfo,
    IndexInfo,
    IndexSettings,
    IndexStatistics,
)

logger = structlog.get_logger()


class IndexService:
    """Service for managing LEANN indexes."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_dir = self.settings.index_path
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _get_index_path(self, index_name: str) -> Path:
        """Get path for an index."""
        return self.index_dir / index_name

    def _get_metadata_path(self, index_name: str) -> Path:
        """Get path for index metadata."""
        return self._get_index_path(index_name) / "metadata.json"

    def _get_documents_path(self, index_name: str) -> Path:
        """Get path for documents storage."""
        return self._get_index_path(index_name) / "documents"

    def _get_leann_index_path(self, index_name: str) -> Path:
        """Get path for LEANN index file."""
        return self._get_index_path(index_name) / "index.leann"

    def _load_metadata(self, index_name: str) -> dict[str, Any]:
        """Load index metadata."""
        metadata_path = self._get_metadata_path(index_name)
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}

    def _save_metadata(self, index_name: str, metadata: dict[str, Any]) -> None:
        """Save index metadata."""
        metadata_path = self._get_metadata_path(index_name)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _count_documents(self, index_name: str) -> int:
        """Count documents in an index."""
        docs_path = self._get_documents_path(index_name)
        if not docs_path.exists():
            return 0
        return len(list(docs_path.glob("*.json")))

    def _count_chunks(self, index_name: str) -> int:
        """Count chunks in an index."""
        metadata = self._load_metadata(index_name)
        return metadata.get("chunk_count", 0)

    def _get_index_status(self, index_name: str) -> str:
        """Get index status."""
        leann_path = self._get_leann_index_path(index_name)
        doc_count = self._count_documents(index_name)

        if doc_count == 0:
            return "empty"
        elif leann_path.exists():
            return "ready"
        else:
            return "building"

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        return self._get_index_path(index_name).exists()

    def list_indexes(self) -> list[IndexInfo]:
        """List all indexes."""
        indexes: list[IndexInfo] = []

        for path in self.index_dir.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                try:
                    metadata = self._load_metadata(path.name)
                    settings_data = metadata.get("settings", {})
                    indexes.append(
                        IndexInfo(
                            name=path.name,
                            created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now(timezone.utc).isoformat())),
                            updated_at=datetime.fromisoformat(metadata["updated_at"]) if metadata.get("updated_at") else None,
                            document_count=self._count_documents(path.name),
                            chunk_count=self._count_chunks(path.name),
                            status=self._get_index_status(path.name),
                            settings=IndexSettings(**settings_data),
                        )
                    )
                except Exception as e:
                    logger.warning("Failed to load index", index_name=path.name, error=str(e))

        return indexes

    def create_index(self, name: str, settings: IndexSettings | None = None) -> IndexInfo:
        """Create a new index."""
        index_path = self._get_index_path(name)
        index_path.mkdir(parents=True, exist_ok=True)

        # Create documents directory
        docs_path = self._get_documents_path(name)
        docs_path.mkdir(parents=True, exist_ok=True)

        # Use default settings if not provided
        if settings is None:
            settings = IndexSettings(
                backend=self.settings.leann_backend,
                embedding_model=self.settings.embedding_model,
                graph_degree=self.settings.graph_degree,
                build_complexity=self.settings.build_complexity,
                chunk_size=self.settings.default_chunk_size,
                chunk_overlap=self.settings.default_chunk_overlap,
            )

        now = datetime.now(timezone.utc)

        metadata = {
            "name": name,
            "created_at": now.isoformat(),
            "updated_at": None,
            "settings": settings.model_dump(),
            "chunk_count": 0,
            "total_characters": 0,
        }

        self._save_metadata(name, metadata)

        return IndexInfo(
            name=name,
            created_at=now,
            updated_at=None,
            document_count=0,
            chunk_count=0,
            status="empty",
            settings=settings,
        )

    def get_index(self, name: str) -> IndexDetailInfo | None:
        """Get index details."""
        if not self.index_exists(name):
            return None

        metadata = self._load_metadata(name)
        settings_data = metadata.get("settings", {})

        # Calculate statistics
        docs_path = self._get_documents_path(name)
        metadata_fields: set[str] = set()
        total_chars = metadata.get("total_characters", 0)

        if docs_path.exists():
            for doc_file in docs_path.glob("*.json"):
                try:
                    with open(doc_file) as f:
                        doc = json.load(f)
                        if doc.get("metadata"):
                            metadata_fields.update(doc["metadata"].keys())
                except Exception:
                    pass

        chunk_count = self._count_chunks(name)
        avg_chunk_size = total_chars / chunk_count if chunk_count > 0 else 0

        return IndexDetailInfo(
            name=name,
            created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now(timezone.utc).isoformat())),
            updated_at=datetime.fromisoformat(metadata["updated_at"]) if metadata.get("updated_at") else None,
            document_count=self._count_documents(name),
            chunk_count=chunk_count,
            status=self._get_index_status(name),
            settings=IndexSettings(**settings_data),
            statistics=IndexStatistics(
                total_characters=total_chars,
                avg_chunk_size=avg_chunk_size,
                metadata_fields=list(metadata_fields),
            ),
        )

    def delete_index(self, name: str) -> bool:
        """Delete an index."""
        index_path = self._get_index_path(name)
        if not index_path.exists():
            return False

        shutil.rmtree(index_path)
        return True

    def rebuild_index(self, name: str, settings: IndexSettings | None = None) -> tuple[int, int]:
        """Rebuild an index. Returns (chunk_count, build_time_ms)."""
        from src.services.document_service import DocumentService

        start_time = time.time()

        metadata = self._load_metadata(name)

        if settings:
            metadata["settings"] = settings.model_dump()
            self._save_metadata(name, metadata)

        # Get document service and rebuild
        doc_service = DocumentService()
        chunk_count = doc_service.rebuild_index(name)

        build_time_ms = int((time.time() - start_time) * 1000)

        # Update metadata
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        metadata["chunk_count"] = chunk_count
        self._save_metadata(name, metadata)

        return chunk_count, build_time_ms

    def update_metadata(
        self,
        name: str,
        chunk_count: int | None = None,
        total_characters: int | None = None,
    ) -> None:
        """Update index metadata."""
        metadata = self._load_metadata(name)
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        if chunk_count is not None:
            metadata["chunk_count"] = chunk_count
        if total_characters is not None:
            metadata["total_characters"] = total_characters

        self._save_metadata(name, metadata)


# Singleton instance
_index_service: IndexService | None = None


def get_index_service() -> IndexService:
    """Get index service instance."""
    global _index_service
    if _index_service is None:
        _index_service = IndexService()
    return _index_service
