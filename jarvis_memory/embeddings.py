"""ChromaDB embedding store — sidecar index for semantic search.

ChromaDB runs alongside Neo4j as a read-optimized embedding index.
Neo4j remains the system of record. If ChromaDB is lost, it can be
rebuilt from Neo4j via rebuild_from_neo4j().

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (local, no API key).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .config import CHROMADB_PATH, CHROMADB_COLLECTION, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """ChromaDB wrapper with graceful fallback.

    All public methods return empty/False if ChromaDB is unavailable,
    so callers never need to handle import errors or connection failures.
    """

    def __init__(
        self,
        path: str = CHROMADB_PATH,
        collection_name: str = CHROMADB_COLLECTION,
        model: str = EMBEDDING_MODEL,
    ):
        self._client = None
        self._collection = None
        self._available = False

        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            self._ef = SentenceTransformerEmbeddingFunction(model_name=model)
            self._client = chromadb.PersistentClient(path=path)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.info(f"ChromaDB ready: {path} ({self._collection.count()} embeddings)")
        except Exception as e:
            logger.warning(f"ChromaDB unavailable, falling back to text search: {e}")

    def health_check(self) -> bool:
        """Return True if ChromaDB is operational."""
        if not self._available or not self._collection:
            return False
        try:
            self._collection.count()
            return True
        except Exception:
            return False

    def embed_and_store(
        self,
        memory_id: str,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Add or update an embedding in ChromaDB.

        Args:
            memory_id: Unique identifier (matches Neo4j node UUID).
            text: Content to embed.
            metadata: Optional metadata dict (wing, room, hall, created_at).

        Returns:
            True if stored successfully.
        """
        if not self._available:
            return False

        try:
            # ChromaDB metadata values must be str, int, float, or bool
            clean_meta = {}
            if metadata:
                for k, v in metadata.items():
                    if v is not None and isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    elif v is not None:
                        clean_meta[k] = str(v)

            self._collection.upsert(
                ids=[memory_id],
                documents=[text],
                metadatas=[clean_meta] if clean_meta else None,
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to embed {memory_id}: {e}")
            return False

    def search(
        self,
        query: str,
        limit: int = 10,
        where_filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Semantic search via ChromaDB.

        Args:
            query: Search query text.
            limit: Maximum results.
            where_filter: ChromaDB where clause, e.g. {"wing": "navi", "room": "auth"}.

        Returns:
            List of dicts with keys: id, distance, similarity, metadata.
            Sorted by similarity descending. Empty list on failure.
        """
        if not self._available:
            return []

        try:
            kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": limit,
            }
            if where_filter:
                # Remove None values — ChromaDB doesn't accept them in where
                clean = {k: v for k, v in where_filter.items() if v is not None}
                if clean:
                    kwargs["where"] = clean

            results = self._collection.query(**kwargs)

            items = []
            if results and results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
                metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

                for i, mid in enumerate(ids):
                    dist = distances[i]
                    # ChromaDB cosine distance: 0 = identical, 2 = opposite
                    # Convert to similarity: 1 - (distance / 2)
                    similarity = max(0.0, 1.0 - (dist / 2.0))
                    items.append({
                        "id": mid,
                        "distance": dist,
                        "similarity": similarity,
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                    })

            return items
        except Exception as e:
            logger.warning(f"ChromaDB search failed: {e}")
            return []

    def delete(self, memory_id: str) -> bool:
        """Remove an embedding from ChromaDB.

        Args:
            memory_id: UUID to remove.

        Returns:
            True if deleted.
        """
        if not self._available:
            return False

        try:
            self._collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.warning(f"Failed to delete {memory_id} from ChromaDB: {e}")
            return False

    def count(self) -> int:
        """Return the number of embeddings stored."""
        if not self._available:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def rebuild_from_neo4j(self, driver) -> dict[str, Any]:
        """Full reindex from Neo4j. Use for recovery or initial backfill.

        Args:
            driver: Neo4j driver instance.

        Returns:
            Dict with stats: total_found, embedded, errors.
        """
        if not self._available:
            return {"error": "ChromaDB not available"}

        from .rooms import detect_room, get_hall

        total = 0
        embedded = 0
        errors = 0

        try:
            with driver.session() as db:
                # Fetch all active memories across all node types
                result = db.run("""
                    MATCH (n)
                    WHERE (n:EntityNode OR n:EpisodicNode OR n:Episode OR n:Entity)
                      AND coalesce(n.lifecycle_status, 'active') IN ['active', 'confirmed']
                    RETURN n.uuid AS uuid,
                           coalesce(n.content, n.name, n.summary, n.fact, '') AS text,
                           coalesce(n.group_id, 'jarvis-global') AS group_id,
                           coalesce(n.memory_type, n.episode_type, 'fact') AS memory_type,
                           n.created_at AS created_at
                """)

                for record in result:
                    total += 1
                    text = record["text"]
                    if not text or len(text.strip()) < 10:
                        continue

                    memory_type = record["memory_type"]
                    group_id = record["group_id"]
                    room = detect_room(text, group_id)
                    hall = get_hall(memory_type)

                    metadata = {
                        "wing": group_id,
                        "room": room,
                        "hall": hall,
                        "memory_type": memory_type,
                    }
                    if record["created_at"]:
                        metadata["created_at"] = str(record["created_at"])

                    ok = self.embed_and_store(record["uuid"], text, metadata)
                    if ok:
                        embedded += 1
                    else:
                        errors += 1

            return {"total_found": total, "embedded": embedded, "errors": errors}
        except Exception as e:
            logger.error(f"Rebuild from Neo4j failed: {e}")
            return {"error": str(e), "total_found": total, "embedded": embedded, "errors": errors}
