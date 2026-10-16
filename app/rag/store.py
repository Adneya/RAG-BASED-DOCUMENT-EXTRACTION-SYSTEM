from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.rag.chunking import RagChunk, chunk_content
from app.rag.embeddings import HashingEmbeddingFunction
from app.schemas import ExtractedContent, RagDocument, RetrievedContext
from app.tools.text_processing import keywords, summarize_text


@dataclass(frozen=True)
class IngestResult:
    documents: list[RagDocument]
    chunks_indexed: int


class RagStore:
    def __init__(
        self,
        *,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
        chunk_tokens: int | None = None,
        chunk_overlap_tokens: int | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir or settings.rag_store_dir)
        self.collection_name = collection_name or settings.rag_collection_name
        self.chunk_tokens = chunk_tokens or settings.rag_chunk_tokens
        self.chunk_overlap_tokens = (
            chunk_overlap_tokens or settings.rag_chunk_overlap_tokens
        )
        self.embedding_function = HashingEmbeddingFunction(
            dimensions=embedding_dimensions or settings.rag_embedding_dimensions
        )
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self.conversation_history: list[str] = []

    def ingest_contents(self, contents: list[ExtractedContent]) -> IngestResult:
        chunks: list[RagChunk] = []
        for content in contents:
            chunks.extend(
                chunk_content(
                    content,
                    target_tokens=self.chunk_tokens,
                    overlap_tokens=self.chunk_overlap_tokens,
                )
            )

        if not chunks:
            return IngestResult(documents=[], chunks_indexed=0)

        documents = [chunk.text for chunk in chunks]
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=documents,
            embeddings=self.embedding_function(documents),
            metadatas=[metadata_for_chunk(chunk) for chunk in chunks],
        )
        return IngestResult(
            documents=summarize_chunks(chunks),
            chunks_indexed=len(chunks),
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedContext]:
        if not query.strip() or self.count_chunks() == 0:
            return []

        top_k = top_k or settings.rag_top_k
        where = where_for_document_ids(document_ids)
        rewritten_query = self.rewrite_query(query)
        vector_hits = self._vector_hits(rewritten_query, where=where)
        keyword_hits = self._keyword_hits(rewritten_query, where=where)
        combined = self._merge_retrieval_scores(vector_hits, keyword_hits)
        if not combined:
            return []

        ranked = self._rerank_hits(rewritten_query, combined)
        contexts = [
            RetrievedContext(
                chunk_id=str(item["chunk_id"]),
                document_id=str(item["document_id"]),
                filename=str(item["filename"]),
                source_type=str(item["source_type"]),
                text=str(item["text"]),
                score=float(item["score"]),
                chunk_index=int(item.get("chunk_index") or 0),
                page_number=item.get("page_number"),
            )
            for item in ranked[: max(top_k * 4, top_k)]
        ]
        if document_ids:
            allowed = set(document_ids)
            contexts = [hit for hit in contexts if hit.document_id in allowed]
        return contexts[:top_k]

    def _rerank_hits(
        self,
        query: str,
        hits: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply a deterministic lexical reranking pass after hybrid retrieval."""
        terms = {
            term
            for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", query.lower())
            if term not in {"the", "this", "that", "with", "from", "what", "about"}
        }
        phrase = re.sub(r"\s+", " ", query.lower()).strip()
        reranked: list[dict[str, Any]] = []
        for hit in hits.values():
            text = str(hit.get("text", "")).lower()
            matched_terms = sum(1 for term in terms if term in text)
            coverage = matched_terms / len(terms) if terms else 0.0
            phrase_bonus = 0.15 if phrase and phrase in text else 0.0
            score = float(hit.get("score", 0.0)) + coverage * 0.35 + phrase_bonus
            enriched = dict(hit)
            enriched["score"] = round(score, 4)
            reranked.append(enriched)
        return sorted(
            reranked,
            key=lambda hit: (float(hit["score"]), str(hit.get("chunk_id", ""))),
            reverse=True,
        )

    def rewrite_query(self, query: str) -> str:
        cleaned = re.sub(r"\s+", " ", (query or "")).strip()
        if not cleaned:
            return ""

        lowered = cleaned.lower()
        base_terms = keywords(cleaned, limit=12)
        expanded: list[str] = []
        for term in base_terms:
            expanded.append(term)

        if re.search(r"\b(deadline|deadlines|due date|due|milestone|schedule|date)\b", lowered):
            expanded.extend([
                "deadline",
                "deadlines",
                "due",
                "due date",
                "milestone",
                "submission date",
                "delivery date",
            ])
        if re.search(r"\b(compare|comparison|difference|vs|versus)\b", lowered):
            expanded.extend(["compare", "difference", "similarity", "contrast", "summary"])
        if re.search(r"\b(policy|contract|terms|clause|agreement)\b", lowered):
            expanded.extend(["policy", "contract", "terms", "clause", "agreement"])
        if re.search(r"\b(research|paper|report|finding|result)\b", lowered):
            expanded.extend(["research", "report", "finding", "result", "analysis"])

        if re.search(r"\b(what about|how about|tell me about)\b", lowered):
            expanded.append("key details")

        unique_terms = []
        seen: set[str] = set()
        for term in [cleaned, *expanded]:
            normalized = term.strip().lower()
            if normalized and normalized not in seen:
                unique_terms.append(term.strip())
                seen.add(normalized)
        return " ".join(unique_terms)[:400]

    def summarize_document(self, document_id: str) -> str:
        result = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        documents = result.get("documents") or []
        if not documents:
            return "No document summary available yet."
        text = "\n\n".join(str(item) for item in documents if str(item).strip())
        return summarize_text(text) if text.strip() else "No document summary available yet."

    def _vector_hits(self, query: str, *, where: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        try:
            result = self._collection.query(
                query_embeddings=self.embedding_function([query]),
                n_results=max(10, self.count_chunks()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            if where:
                result = self._collection.query(
                    query_embeddings=self.embedding_function([query]),
                    n_results=max(10, self.count_chunks()),
                    include=["documents", "metadatas", "distances"],
                )
            else:
                raise

        hits: dict[str, dict[str, Any]] = {}
        ids = first_result_list(result.get("ids"))
        documents = first_result_list(result.get("documents"))
        metadatas = first_result_list(result.get("metadatas"))
        distances = first_result_list(result.get("distances"))
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            document = documents[index] if index < len(documents) else ""
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits[str(chunk_id)] = {
                "chunk_id": chunk_id,
                "document_id": metadata.get("document_id", ""),
                "filename": metadata.get("filename", "Unknown"),
                "source_type": metadata.get("source_type", "unknown"),
                "text": document,
                "chunk_index": int(metadata.get("chunk_index") or 0),
                "page_number": metadata.get("page_number"),
                "score": round(1 / (1 + max(distance, 0.0)), 4),
            }
        return hits

    def _keyword_hits(self, query: str, *, where: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        terms = [
            term
            for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", query.lower())
            if term not in {"the", "this", "that", "with", "from"}
        ]
        if not terms:
            return {}
        collection_result = self._collection.get(
            where=where,
            include=["documents", "metadatas"],
            limit=max(50, self.count_chunks()),
        )
        hits: dict[str, dict[str, Any]] = {}
        docs = collection_result.get("documents") or []
        metadatas = collection_result.get("metadatas") or []
        ids = collection_result.get("ids") or []
        for index, chunk_id in enumerate(ids):
            text = str(docs[index] if index < len(docs) else "")
            metadata = metadatas[index] if index < len(metadatas) else {}
            if not text:
                continue
            score = 0.0
            for term in terms:
                score += text.lower().count(term)
            if score > 0:
                hits[str(chunk_id)] = {
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id", ""),
                    "filename": metadata.get("filename", "Unknown"),
                    "source_type": metadata.get("source_type", "unknown"),
                    "text": text,
                    "chunk_index": int(metadata.get("chunk_index") or 0),
                    "page_number": metadata.get("page_number"),
                    "score": round(score, 4),
                }
        return hits

    def _merge_retrieval_scores(
        self,
        vector_hits: dict[str, dict[str, Any]],
        keyword_hits: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for chunk_id, hit in vector_hits.items():
            merged[chunk_id] = dict(hit)
            merged[chunk_id]["score"] = float(hit.get("score", 0.0))
        for chunk_id, hit in keyword_hits.items():
            if chunk_id not in merged:
                merged[chunk_id] = dict(hit)
            merged[chunk_id]["score"] = (
                float(merged[chunk_id].get("score", 0.0))
                + float(hit.get("score", 0.0)) * 0.5
            )
        return merged

    def list_documents(self) -> list[RagDocument]:
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        documents: dict[str, RagDocument] = {}

        for metadata in metadatas:
            if not metadata:
                continue
            document_id = str(metadata.get("document_id") or "")
            if not document_id:
                continue
            current = documents.get(document_id)
            token_count = int(metadata.get("token_count") or 0)
            if current is None:
                documents[document_id] = RagDocument(
                    document_id=document_id,
                    filename=str(metadata.get("filename") or "Unknown"),
                    source_type=str(metadata.get("source_type") or "unknown"),
                    chunk_count=1,
                    token_count=token_count,
                )
            else:
                current.chunk_count += 1
                current.token_count += token_count

        return sorted(documents.values(), key=lambda item: item.filename.lower())

    def delete_document(self, document_id: str) -> int:
        existing = self._collection.get(where={"document_id": document_id})
        ids = existing.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def count_chunks(self) -> int:
        return int(self._collection.count())

    def close(self) -> None:
        try:
            self._client._system.stop()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self._client.clear_system_cache()
        except Exception:
            pass


def metadata_for_chunk(chunk: RagChunk) -> dict[str, str | int | float | bool]:
    return {
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "source_type": chunk.source_type,
        "chunk_index": chunk.chunk_index,
        "page_number": chunk.page_number or 0,
        "token_count": chunk.token_count,
        "section_title": chunk.section_title or "",
    }


def summarize_chunks(chunks: list[RagChunk]) -> list[RagDocument]:
    documents: dict[str, RagDocument] = {}
    for chunk in chunks:
        current = documents.get(chunk.document_id)
        if current is None:
            documents[chunk.document_id] = RagDocument(
                document_id=chunk.document_id,
                filename=chunk.filename,
                source_type=chunk.source_type,
                chunk_count=1,
                token_count=chunk.token_count,
            )
        else:
            current.chunk_count += 1
            current.token_count += chunk.token_count
    return list(documents.values())


def where_for_document_ids(document_ids: list[str] | None) -> dict[str, Any] | None:
    if not document_ids:
        return None
    unique_ids = sorted(set(document_ids))
    if len(unique_ids) == 1:
        return {"document_id": unique_ids[0]}
    return {"document_id": {"$in": unique_ids}}


def contexts_from_query_result(result: dict[str, Any]) -> list[RetrievedContext]:
    ids = first_result_list(result.get("ids"))
    documents = first_result_list(result.get("documents"))
    metadatas = first_result_list(result.get("metadatas"))
    distances = first_result_list(result.get("distances"))

    contexts: list[RetrievedContext] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        distance = float(distances[index]) if index < len(distances) else 1.0
        page_number = int(metadata.get("page_number") or 0) or None
        contexts.append(
            RetrievedContext(
                chunk_id=str(chunk_id),
                document_id=str(metadata.get("document_id") or ""),
                filename=str(metadata.get("filename") or "Unknown"),
                source_type=str(metadata.get("source_type") or "unknown"),
                text=str(documents[index] if index < len(documents) else ""),
                score=round(1 / (1 + max(distance, 0.0)), 4),
                chunk_index=int(metadata.get("chunk_index") or 0),
                page_number=page_number,
            )
        )
    return contexts


def first_result_list(value: Any) -> list[Any]:
    if not value:
        return []
    first = value[0]
    return first if isinstance(first, list) else value


@lru_cache(maxsize=1)
def get_default_rag_store() -> RagStore:
    return RagStore()
