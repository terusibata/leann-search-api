"""Search service."""

import json
import re
import time
from pathlib import Path
from typing import Any

import structlog

from src.config import get_settings
from src.schemas.search import (
    BatchSearchResultItem,
    GrepSearchResult,
    SearchResult,
)
from src.services.document_service import DocumentService, get_document_service

logger = structlog.get_logger()


class SearchService:
    """Service for searching documents in LEANN indexes."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_dir = self.settings.index_path
        self._searcher_cache: dict[str, Any] = {}

    def _get_leann_index_path(self, index_name: str) -> Path:
        """Get path for LEANN index file."""
        return self.index_dir / index_name / "index.leann"

    def _get_chunks_path(self, index_name: str) -> Path:
        """Get path for chunks storage."""
        return self.index_dir / index_name / "chunks"

    def _get_searcher(self, index_name: str):
        """Get or create a LEANN searcher for an index."""
        if index_name in self._searcher_cache:
            return self._searcher_cache[index_name]

        leann_path = self._get_leann_index_path(index_name)
        if not leann_path.exists():
            logger.debug("LEANN index not found", index_name=index_name, path=str(leann_path))
            return None

        try:
            from leann import LeannSearcher

            searcher = LeannSearcher(str(leann_path))
            self._searcher_cache[index_name] = searcher
            logger.info("LEANN searcher created", index_name=index_name)
            return searcher
        except ImportError:
            logger.warning("LEANN package not available")
            return None
        except Exception as e:
            logger.error("Failed to create LEANN searcher", index_name=index_name, error=str(e))
            return None

    def _load_chunk(self, index_name: str, chunk_id: str) -> dict[str, Any] | None:
        """Load a chunk by ID."""
        chunk_path = self._get_chunks_path(index_name) / f"{chunk_id}.json"
        if not chunk_path.exists():
            return None
        with open(chunk_path) as f:
            return json.load(f)

    def _load_all_chunks(self, index_name: str) -> list[dict[str, Any]]:
        """Load all chunks for an index."""
        chunks_path = self._get_chunks_path(index_name)
        if not chunks_path.exists():
            return []

        chunks: list[dict[str, Any]] = []
        for chunk_file in sorted(chunks_path.glob("*.json")):
            try:
                with open(chunk_file) as f:
                    chunks.append(json.load(f))
            except Exception:
                pass
        return chunks

    def _apply_metadata_filter(
        self, chunks: list[dict[str, Any]], filters: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Filter chunks by metadata."""
        if not filters:
            return chunks

        doc_service = get_document_service()
        return [
            chunk
            for chunk in chunks
            if doc_service._matches_filter(chunk.get("metadata", {}), filters)
        ]

    def search(
        self,
        index_name: str,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        search_complexity: int = 32,
        include_content: bool = True,
        include_metadata: bool = True,
        min_score: float = 0.0,
    ) -> tuple[list[SearchResult], int, int]:
        """
        Perform semantic search.
        Returns (results, total_found, query_time_ms).
        """
        start_time = time.time()

        # Load chunks and chunk mapping
        doc_service = get_document_service()
        chunk_mapping = doc_service._load_chunk_mapping(index_name)
        chunks = self._load_all_chunks(index_name)
        chunk_map = {c.get("chunk_id"): c for c in chunks}

        # Try to use LEANN searcher first
        searcher = self._get_searcher(index_name)
        if searcher is not None and chunk_mapping:
            try:
                # Get more results for filtering
                fetch_k = top_k * 2 if metadata_filters else top_k
                leann_results = searcher.search(query, top_k=fetch_k)

                search_results: list[SearchResult] = []

                # Process LEANN results
                # LEANN returns results with indices corresponding to add_text order
                if isinstance(leann_results, list):
                    # Results are returned as list of (index, score) or similar
                    for item in leann_results:
                        if isinstance(item, tuple) and len(item) >= 2:
                            idx, score = item[0], item[1]
                        elif isinstance(item, dict):
                            idx = item.get("index", item.get("id", 0))
                            score = item.get("score", item.get("similarity", 1.0))
                        else:
                            continue

                        if score < min_score:
                            continue

                        # Map index to chunk_id
                        if 0 <= idx < len(chunk_mapping):
                            chunk_id = chunk_mapping[idx]
                            chunk = chunk_map.get(chunk_id)

                            if chunk:
                                # Apply metadata filter
                                if metadata_filters:
                                    if not doc_service._matches_filter(
                                        chunk.get("metadata", {}), metadata_filters
                                    ):
                                        continue

                                search_results.append(
                                    SearchResult(
                                        document_id=chunk.get("document_id", ""),
                                        chunk_id=chunk_id,
                                        content=chunk.get("content") if include_content else None,
                                        metadata=chunk.get("metadata") if include_metadata else None,
                                        score=float(score),
                                        position=chunk.get("position", 0),
                                    )
                                )

                                if len(search_results) >= top_k:
                                    break

                elif hasattr(leann_results, '__iter__'):
                    # Try to iterate over results
                    for idx, item in enumerate(leann_results):
                        if idx >= len(chunk_mapping):
                            break

                        score = float(item) if isinstance(item, (int, float)) else 1.0

                        if score < min_score:
                            continue

                        chunk_id = chunk_mapping[idx]
                        chunk = chunk_map.get(chunk_id)

                        if chunk:
                            if metadata_filters:
                                if not doc_service._matches_filter(
                                    chunk.get("metadata", {}), metadata_filters
                                ):
                                    continue

                            search_results.append(
                                SearchResult(
                                    document_id=chunk.get("document_id", ""),
                                    chunk_id=chunk_id,
                                    content=chunk.get("content") if include_content else None,
                                    metadata=chunk.get("metadata") if include_metadata else None,
                                    score=score,
                                    position=chunk.get("position", 0),
                                )
                            )

                            if len(search_results) >= top_k:
                                break

                if search_results:
                    query_time_ms = int((time.time() - start_time) * 1000)
                    return search_results, len(search_results), query_time_ms

            except Exception as e:
                logger.warning("LEANN search failed, falling back to brute force", error=str(e))

        # Fallback: brute-force search using embeddings
        return self._brute_force_search(
            index_name,
            query,
            top_k,
            metadata_filters,
            include_content,
            include_metadata,
            min_score,
            start_time,
        )

    def _brute_force_search(
        self,
        index_name: str,
        query: str,
        top_k: int,
        metadata_filters: dict[str, Any] | None,
        include_content: bool,
        include_metadata: bool,
        min_score: float,
        start_time: float,
    ) -> tuple[list[SearchResult], int, int]:
        """Brute-force search using direct embedding similarity."""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.settings.embedding_model)
        except ImportError:
            logger.warning("sentence-transformers not available for fallback search")
            query_time_ms = int((time.time() - start_time) * 1000)
            return [], 0, query_time_ms
        except Exception as e:
            logger.error("Failed to load embedding model", error=str(e))
            query_time_ms = int((time.time() - start_time) * 1000)
            return [], 0, query_time_ms

        chunks = self._load_all_chunks(index_name)
        chunks = self._apply_metadata_filter(chunks, metadata_filters)

        if not chunks:
            query_time_ms = int((time.time() - start_time) * 1000)
            return [], 0, query_time_ms

        # Encode query
        query_embedding = model.encode(query, normalize_embeddings=True)

        # Encode all chunks and compute similarities
        chunk_texts = [c.get("content", "") for c in chunks]
        chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)

        # Compute cosine similarities
        similarities = chunk_embeddings @ query_embedding

        # Sort by similarity
        sorted_indices = similarities.argsort()[::-1]

        results: list[SearchResult] = []
        for idx in sorted_indices[:top_k]:
            score = float(similarities[idx])
            if score < min_score:
                break

            chunk = chunks[idx]
            results.append(
                SearchResult(
                    document_id=chunk.get("document_id", ""),
                    chunk_id=chunk.get("chunk_id", ""),
                    content=chunk.get("content") if include_content else None,
                    metadata=chunk.get("metadata") if include_metadata else None,
                    score=score,
                    position=chunk.get("position", 0),
                )
            )

        query_time_ms = int((time.time() - start_time) * 1000)
        return results, len(results), query_time_ms

    def grep_search(
        self,
        index_name: str,
        query: str,
        top_k: int = 20,
        metadata_filters: dict[str, Any] | None = None,
    ) -> tuple[list[GrepSearchResult], int, int]:
        """
        Perform keyword/grep search.
        Returns (results, total_found, query_time_ms).
        """
        start_time = time.time()

        chunks = self._load_all_chunks(index_name)
        chunks = self._apply_metadata_filter(chunks, metadata_filters)

        results: list[GrepSearchResult] = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        for chunk in chunks:
            content = chunk.get("content", "")
            matches = list(pattern.finditer(content))

            if matches:
                match_positions = [[m.start(), m.end()] for m in matches]
                results.append(
                    GrepSearchResult(
                        document_id=chunk.get("document_id", ""),
                        chunk_id=chunk.get("chunk_id", ""),
                        content=content,
                        metadata=chunk.get("metadata"),
                        match_positions=match_positions,
                    )
                )

                if len(results) >= top_k:
                    break

        query_time_ms = int((time.time() - start_time) * 1000)
        return results, len(results), query_time_ms

    def hybrid_search(
        self,
        index_name: str,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        metadata_filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchResult], int, int]:
        """
        Perform hybrid search combining semantic and keyword search.
        Returns (results, total_found, query_time_ms).
        """
        start_time = time.time()

        # Get more results from both methods
        fetch_k = top_k * 3

        # Semantic search
        semantic_results, _, _ = self.search(
            index_name,
            query,
            top_k=fetch_k,
            metadata_filters=metadata_filters,
            include_content=True,
            include_metadata=True,
        )

        # Grep search
        grep_results, _, _ = self.grep_search(
            index_name, query, top_k=fetch_k, metadata_filters=metadata_filters
        )

        # Combine scores
        score_map: dict[str, dict[str, Any]] = {}

        for result in semantic_results:
            score_map[result.chunk_id] = {
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "content": result.content,
                "metadata": result.metadata,
                "position": result.position,
                "semantic_score": result.score,
                "keyword_score": 0.0,
            }

        max_grep_score = len(grep_results) if grep_results else 1
        for i, result in enumerate(grep_results):
            grep_score = (max_grep_score - i) / max_grep_score

            if result.chunk_id in score_map:
                score_map[result.chunk_id]["keyword_score"] = grep_score
            else:
                score_map[result.chunk_id] = {
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "content": result.content,
                    "metadata": result.metadata,
                    "position": 0,
                    "semantic_score": 0.0,
                    "keyword_score": grep_score,
                }

        # Calculate combined scores
        for chunk_id, data in score_map.items():
            data["combined_score"] = (
                data["semantic_score"] * semantic_weight
                + data["keyword_score"] * keyword_weight
            )

        # Sort by combined score
        sorted_results = sorted(
            score_map.values(), key=lambda x: x["combined_score"], reverse=True
        )

        results = [
            SearchResult(
                document_id=r["document_id"],
                chunk_id=r["chunk_id"],
                content=r["content"],
                metadata=r["metadata"],
                score=r["combined_score"],
                position=r["position"],
            )
            for r in sorted_results[:top_k]
        ]

        query_time_ms = int((time.time() - start_time) * 1000)
        return results, len(results), query_time_ms

    def batch_search(
        self,
        index_name: str,
        queries: list[dict[str, Any]],
        metadata_filters: dict[str, Any] | None = None,
    ) -> tuple[dict[str, BatchSearchResultItem], int]:
        """
        Perform batch search for multiple queries.
        Returns (results_dict, total_query_time_ms).
        """
        start_time = time.time()
        results: dict[str, BatchSearchResultItem] = {}

        for query_item in queries:
            query_id = query_item.get("id", "")
            query_text = query_item.get("query", "")
            query_top_k = query_item.get("top_k", 5)

            search_results, total_found, _ = self.search(
                index_name,
                query_text,
                top_k=query_top_k,
                metadata_filters=metadata_filters,
            )

            results[query_id] = BatchSearchResultItem(
                results=search_results, total_found=total_found
            )

        total_query_time_ms = int((time.time() - start_time) * 1000)
        return results, total_query_time_ms

    def invalidate_cache(self, index_name: str) -> None:
        """Invalidate the searcher cache for an index."""
        if index_name in self._searcher_cache:
            del self._searcher_cache[index_name]


# Singleton instance
_search_service: SearchService | None = None


def get_search_service() -> SearchService:
    """Get search service instance."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
