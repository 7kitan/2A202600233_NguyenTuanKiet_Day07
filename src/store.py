from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            self._use_chroma = True
            client = chromadb.Client()
            self._collection = client.create_collection(name=collection_name)

        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
        }

    def _search_records(
        self, query: str, records: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        query_embedding: list[float] = self._embedding_fn(query)

        simi_scores = []
        for r in records:
            if "embeddings" not in r:
                r["embeddings"] = self._embedding_fn(r["content"])
            score = _dot(r["embeddings"], query_embedding)
            simi_scores.append((score, r))

        simi_scores.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in simi_scores[:top_k]]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """

        # TODO: embed each doc and add to store
        def _normalize(md, doc_id):
            md = dict(md) if md else {}
            md["doc_id"] = doc_id
            return md

        for doc in docs:
            embedding = self._embedding_fn(doc.content)
            metadata = _normalize(doc.metadata, doc.id)

            if self._collection:
                self._collection.add(
                    ids=[doc.id],
                    documents=[doc.content],
                    metadatas=[metadata],
                    embeddings=[embedding],
                )
            else:
                record = {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": metadata,
                    "embedding": embedding,
                }
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product similarity.
        For ChromaDB: convert distance → similarity.
        """
        query_embedding: list[float] = self._embedding_fn(query)

        if self._collection:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )

            return [
                {
                    "id": doc_id,
                    "content": doc,
                    "metadata": metadata,
                    "score": 1 - distance,  # convert distance → similarity
                }
                for doc_id, doc, metadata, distance in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]

        else:
            simi_scores = []

            for r in self._store:
                if "embeddings" not in r:
                    r["embeddings"] = self._embedding_fn(r["content"])

                score = _dot(r["embeddings"], query_embedding)
                simi_scores.append((score, r))

            simi_scores.sort(key=lambda x: x[0], reverse=True)

            return [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "score": score,
                }
                for score, r in simi_scores[:top_k]
            ]

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._collection:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(
        self, query: str, top_k: int = 3, metadata_filter: dict = None
    ) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if self._collection:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                where=metadata_filter,
            )
            return [
                {
                    "id": doc_id,
                    "content": doc,
                    "metadata": metadata,
                    "score": score,
                }
                for doc_id, doc, metadata, score in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        else:
            filtered_records = [
                r
                for r in self._store
                if metadata_filter is None
                or all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        if self._collection:
            # delete by metadata filter
            delete_result = self._collection.delete(where={"doc_id": doc_id})
            if delete_result["deleted"] > 0:
                return True
            return False
        else:
            removed = False
            new_store = []

            for r in self._store:
                if r["metadata"].get("doc_id") == doc_id:
                    removed = True
                else:
                    new_store.append(r)

            self._store = new_store
            return removed
