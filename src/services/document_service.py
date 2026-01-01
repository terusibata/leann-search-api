"""Document management service."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.config import get_settings
from src.schemas.document import (
    ChunkInfo,
    DocumentAddResult,
    DocumentDetail,
    DocumentInfo,
)
from src.schemas.index import IndexSettings

logger = structlog.get_logger()


class DocumentService:
    """Service for managing documents in LEANN indexes."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_dir = self.settings.index_path
        self._embedding_model = None
        self._leann_builder = None

    def _get_documents_path(self, index_name: str) -> Path:
        """Get path for documents storage."""
        return self.index_dir / index_name / "documents"

    def _get_document_path(self, index_name: str, doc_id: str) -> Path:
        """Get path for a specific document."""
        return self._get_documents_path(index_name) / f"{doc_id}.json"

    def _get_leann_index_path(self, index_name: str) -> Path:
        """Get path for LEANN index file."""
        return self.index_dir / index_name / "index.leann"

    def _get_chunks_path(self, index_name: str) -> Path:
        """Get path for chunks storage."""
        return self.index_dir / index_name / "chunks"

    def _load_document(self, index_name: str, doc_id: str) -> dict[str, Any] | None:
        """Load a document from storage."""
        doc_path = self._get_document_path(index_name, doc_id)
        if not doc_path.exists():
            return None
        with open(doc_path) as f:
            return json.load(f)

    def _save_document(self, index_name: str, doc_id: str, data: dict[str, Any]) -> None:
        """Save a document to storage."""
        doc_path = self._get_document_path(index_name, doc_id)
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(doc_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _delete_document_file(self, index_name: str, doc_id: str) -> bool:
        """Delete a document file."""
        doc_path = self._get_document_path(index_name, doc_id)
        if doc_path.exists():
            doc_path.unlink()
            return True
        return False

    def _chunk_text(
        self, text: str, chunk_size: int = 512, chunk_overlap: int = 64
    ) -> list[str]:
        """Split text into chunks."""
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            if end >= text_len:
                chunks.append(text[start:])
                break

            # Try to find a good break point
            break_point = end
            for sep in ["\n\n", "\n", "。", ".", " "]:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos != -1:
                    break_point = pos + len(sep)
                    break

            chunks.append(text[start:break_point])
            start = break_point - chunk_overlap

        return chunks

    def _get_index_settings(self, index_name: str) -> IndexSettings:
        """Get settings for an index."""
        metadata_path = self.index_dir / index_name / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                return IndexSettings(**metadata.get("settings", {}))
        return IndexSettings()

    def document_exists(self, index_name: str, doc_id: str) -> bool:
        """Check if a document exists."""
        return self._get_document_path(index_name, doc_id).exists()

    def add_documents(
        self,
        index_name: str,
        documents: list[dict[str, Any]],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        update_if_exists: bool = False,
    ) -> list[DocumentAddResult]:
        """Add documents to an index."""
        settings = self._get_index_settings(index_name)
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap

        results: list[DocumentAddResult] = []
        total_chunks = 0
        total_chars = 0

        chunks_path = self._get_chunks_path(index_name)
        chunks_path.mkdir(parents=True, exist_ok=True)

        for doc in documents:
            doc_id = doc.get("id") or str(uuid.uuid4())
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            try:
                existing = self._load_document(index_name, doc_id)
                if existing and not update_if_exists:
                    results.append(
                        DocumentAddResult(
                            id=doc_id,
                            chunk_count=0,
                            status="failed",
                            error="Document already exists",
                        )
                    )
                    continue

                # Chunk the content
                chunks = self._chunk_text(content, chunk_size, chunk_overlap)

                now = datetime.now(timezone.utc)

                # Save document data
                doc_data = {
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "chunk_count": len(chunks),
                    "created_at": now.isoformat() if not existing else existing.get("created_at", now.isoformat()),
                    "updated_at": now.isoformat(),
                }
                self._save_document(index_name, doc_id, doc_data)

                # Save chunks
                for i, chunk_content in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "content": chunk_content,
                        "position": i,
                        "metadata": metadata,
                    }
                    chunk_path = chunks_path / f"{chunk_id}.json"
                    with open(chunk_path, "w") as f:
                        json.dump(chunk_data, f, indent=2, default=str)

                total_chunks += len(chunks)
                total_chars += len(content)

                status = "updated" if existing else "added"
                results.append(
                    DocumentAddResult(id=doc_id, chunk_count=len(chunks), status=status)
                )

            except Exception as e:
                logger.error("Failed to add document", doc_id=doc_id, error=str(e))
                results.append(
                    DocumentAddResult(
                        id=doc_id, chunk_count=0, status="failed", error=str(e)
                    )
                )

        # Update index metadata
        from src.services.index_service import get_index_service

        index_service = get_index_service()
        metadata_path = self.index_dir / index_name / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                index_meta = json.load(f)
            current_chunks = index_meta.get("chunk_count", 0)
            current_chars = index_meta.get("total_characters", 0)
            index_service.update_metadata(
                index_name,
                chunk_count=current_chunks + total_chunks,
                total_characters=current_chars + total_chars,
            )

        return results

    def list_documents(
        self,
        index_name: str,
        page: int = 1,
        per_page: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        metadata_filter: dict[str, Any] | None = None,
    ) -> tuple[list[DocumentInfo], int]:
        """List documents in an index."""
        docs_path = self._get_documents_path(index_name)
        if not docs_path.exists():
            return [], 0

        documents: list[DocumentInfo] = []

        for doc_file in docs_path.glob("*.json"):
            try:
                with open(doc_file) as f:
                    doc = json.load(f)

                # Apply metadata filter
                if metadata_filter and not self._matches_filter(
                    doc.get("metadata", {}), metadata_filter
                ):
                    continue

                content = doc.get("content", "")
                content_preview = content[:200] + "..." if len(content) > 200 else content

                documents.append(
                    DocumentInfo(
                        id=doc["id"],
                        content_preview=content_preview,
                        metadata=doc.get("metadata"),
                        chunk_count=doc.get("chunk_count", 0),
                        created_at=datetime.fromisoformat(doc.get("created_at", datetime.now(timezone.utc).isoformat())),
                        updated_at=datetime.fromisoformat(doc["updated_at"]) if doc.get("updated_at") else None,
                    )
                )
            except Exception as e:
                logger.warning("Failed to load document", file=doc_file, error=str(e))

        # Sort
        reverse = sort_order == "desc"
        if sort_by == "created_at":
            documents.sort(key=lambda d: d.created_at, reverse=reverse)
        elif sort_by == "updated_at":
            documents.sort(key=lambda d: d.updated_at or d.created_at, reverse=reverse)
        elif sort_by == "id":
            documents.sort(key=lambda d: d.id, reverse=reverse)

        total = len(documents)

        # Paginate
        start = (page - 1) * per_page
        end = start + per_page
        documents = documents[start:end]

        return documents, total

    def get_document(self, index_name: str, doc_id: str) -> DocumentDetail | None:
        """Get document details."""
        doc = self._load_document(index_name, doc_id)
        if not doc:
            return None

        # Load chunks
        chunks_path = self._get_chunks_path(index_name)
        chunks: list[ChunkInfo] = []

        if chunks_path.exists():
            for chunk_file in sorted(chunks_path.glob(f"{doc_id}_chunk_*.json")):
                try:
                    with open(chunk_file) as f:
                        chunk_data = json.load(f)
                        chunks.append(
                            ChunkInfo(
                                chunk_id=chunk_data["chunk_id"],
                                content=chunk_data["content"],
                                position=chunk_data["position"],
                            )
                        )
                except Exception:
                    pass

        return DocumentDetail(
            id=doc["id"],
            content=doc.get("content", ""),
            metadata=doc.get("metadata"),
            chunks=chunks,
            chunk_count=doc.get("chunk_count", len(chunks)),
            created_at=datetime.fromisoformat(doc.get("created_at", datetime.now(timezone.utc).isoformat())),
            updated_at=datetime.fromisoformat(doc["updated_at"]) if doc.get("updated_at") else None,
        )

    def update_document(
        self,
        index_name: str,
        doc_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        merge_metadata: bool = True,
    ) -> DocumentAddResult | None:
        """Update a document."""
        existing = self._load_document(index_name, doc_id)
        if not existing:
            return None

        # Merge or replace metadata
        if metadata is not None:
            if merge_metadata and existing.get("metadata"):
                new_metadata = {**existing["metadata"], **metadata}
            else:
                new_metadata = metadata
        else:
            new_metadata = existing.get("metadata")

        # If content is updated, re-chunk
        if content is not None:
            settings = self._get_index_settings(index_name)

            # Delete old chunks
            chunks_path = self._get_chunks_path(index_name)
            if chunks_path.exists():
                for chunk_file in chunks_path.glob(f"{doc_id}_chunk_*.json"):
                    chunk_file.unlink()

            # Create new chunks
            chunks = self._chunk_text(content, settings.chunk_size, settings.chunk_overlap)

            for i, chunk_content in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_data = {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "content": chunk_content,
                    "position": i,
                    "metadata": new_metadata,
                }
                chunk_path = chunks_path / f"{chunk_id}.json"
                with open(chunk_path, "w") as f:
                    json.dump(chunk_data, f, indent=2, default=str)

            existing["content"] = content
            existing["chunk_count"] = len(chunks)
        else:
            # Update metadata in chunks too
            chunks_path = self._get_chunks_path(index_name)
            if chunks_path.exists() and metadata is not None:
                for chunk_file in chunks_path.glob(f"{doc_id}_chunk_*.json"):
                    with open(chunk_file) as f:
                        chunk_data = json.load(f)
                    chunk_data["metadata"] = new_metadata
                    with open(chunk_file, "w") as f:
                        json.dump(chunk_data, f, indent=2, default=str)

        existing["metadata"] = new_metadata
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_document(index_name, doc_id, existing)

        return DocumentAddResult(
            id=doc_id, chunk_count=existing.get("chunk_count", 0), status="updated"
        )

    def update_metadata_only(
        self,
        index_name: str,
        doc_id: str,
        metadata: dict[str, Any],
        merge: bool = True,
    ) -> dict[str, Any] | None:
        """Update only the metadata of a document (no re-indexing needed)."""
        existing = self._load_document(index_name, doc_id)
        if not existing:
            return None

        if merge and existing.get("metadata"):
            new_metadata = {**existing["metadata"], **metadata}
        else:
            new_metadata = metadata

        existing["metadata"] = new_metadata
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_document(index_name, doc_id, existing)

        # Update metadata in chunks
        chunks_path = self._get_chunks_path(index_name)
        if chunks_path.exists():
            for chunk_file in chunks_path.glob(f"{doc_id}_chunk_*.json"):
                with open(chunk_file) as f:
                    chunk_data = json.load(f)
                chunk_data["metadata"] = new_metadata
                with open(chunk_file, "w") as f:
                    json.dump(chunk_data, f, indent=2, default=str)

        return new_metadata

    def delete_document(self, index_name: str, doc_id: str) -> bool:
        """Delete a document."""
        if not self.document_exists(index_name, doc_id):
            return False

        # Delete document file
        self._delete_document_file(index_name, doc_id)

        # Delete chunks
        chunks_path = self._get_chunks_path(index_name)
        if chunks_path.exists():
            for chunk_file in chunks_path.glob(f"{doc_id}_chunk_*.json"):
                chunk_file.unlink()

        return True

    def bulk_delete(
        self,
        index_name: str,
        document_ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete multiple documents."""
        deleted_count = 0

        if document_ids:
            for doc_id in document_ids:
                if self.delete_document(index_name, doc_id):
                    deleted_count += 1
        elif metadata_filter:
            docs_path = self._get_documents_path(index_name)
            if docs_path.exists():
                for doc_file in list(docs_path.glob("*.json")):
                    try:
                        with open(doc_file) as f:
                            doc = json.load(f)
                        if self._matches_filter(doc.get("metadata", {}), metadata_filter):
                            if self.delete_document(index_name, doc["id"]):
                                deleted_count += 1
                    except Exception:
                        pass

        return deleted_count

    def _get_chunk_mapping_path(self, index_name: str) -> Path:
        """Get path for chunk ID mapping file."""
        return self.index_dir / index_name / "chunk_mapping.json"

    def _save_chunk_mapping(self, index_name: str, mapping: list[str]) -> None:
        """Save chunk ID mapping (index position -> chunk_id)."""
        mapping_path = self._get_chunk_mapping_path(index_name)
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)

    def _load_chunk_mapping(self, index_name: str) -> list[str]:
        """Load chunk ID mapping."""
        mapping_path = self._get_chunk_mapping_path(index_name)
        if mapping_path.exists():
            with open(mapping_path) as f:
                return json.load(f)
        return []

    def rebuild_index(self, index_name: str) -> int:
        """Rebuild the LEANN index. Returns chunk count."""
        chunks_path = self._get_chunks_path(index_name)
        leann_path = self._get_leann_index_path(index_name)

        if not chunks_path.exists():
            return 0

        chunk_files = sorted(chunks_path.glob("*.json"))
        if not chunk_files:
            return 0

        try:
            from leann import LeannBuilder
        except ImportError:
            logger.warning("LEANN not available, skipping index build")
            return len(chunk_files)

        settings = self._get_index_settings(index_name)

        # Create builder with settings
        builder = LeannBuilder(
            backend_name=settings.backend,
            embedding_model=settings.embedding_model,
            embedding_mode="sentence-transformers",
            graph_degree=settings.graph_degree,
            build_complexity=settings.build_complexity,
        )

        # Track chunk_id mapping (index position -> chunk_id)
        chunk_mapping: list[str] = []

        for chunk_file in chunk_files:
            with open(chunk_file) as f:
                chunk_data = json.load(f)
            builder.add_text(chunk_data["content"])
            chunk_mapping.append(chunk_data["chunk_id"])

        # Build the index
        builder.build_index(str(leann_path))

        # Save chunk mapping for search
        self._save_chunk_mapping(index_name, chunk_mapping)

        logger.info(
            "LEANN index built successfully",
            index_name=index_name,
            chunk_count=len(chunk_files),
        )

        return len(chunk_files)

    def _matches_filter(
        self, metadata: dict[str, Any], filters: dict[str, Any]
    ) -> bool:
        """Check if metadata matches the filter criteria."""
        for field, condition in filters.items():
            if not isinstance(condition, dict):
                # Simple equality check
                if metadata.get(field) != condition:
                    return False
                continue

            value = metadata.get(field)

            for op, expected in condition.items():
                if op == "==":
                    if value != expected:
                        return False
                elif op == "!=":
                    if value == expected:
                        return False
                elif op == "<":
                    if value is None or value >= expected:
                        return False
                elif op == "<=":
                    if value is None or value > expected:
                        return False
                elif op == ">":
                    if value is None or value <= expected:
                        return False
                elif op == ">=":
                    if value is None or value < expected:
                        return False
                elif op == "in":
                    if value not in expected:
                        return False
                elif op == "not_in":
                    if value in expected:
                        return False
                elif op == "contains":
                    if isinstance(value, str):
                        if expected not in value:
                            return False
                    elif isinstance(value, list):
                        if expected not in value:
                            return False
                    else:
                        return False
                elif op == "starts_with":
                    if not isinstance(value, str) or not value.startswith(expected):
                        return False
                elif op == "ends_with":
                    if not isinstance(value, str) or not value.endswith(expected):
                        return False
                elif op == "is_true":
                    if value is not True:
                        return False
                elif op == "is_false":
                    if value is not False:
                        return False

        return True


# Singleton instance
_document_service: DocumentService | None = None


def get_document_service() -> DocumentService:
    """Get document service instance."""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service
