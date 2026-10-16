import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.agent.pipeline import AgentPipeline
from app.rag.chunking import chunk_content
from app.rag.store import RagStore
from app.schemas import ExtractedContent, RagDocument, RetrievedContext


class DummyUpload:
    def __init__(self, filename: str):
        self.filename = filename


class RagChunkingTests(unittest.TestCase):
    def test_chunk_content_preserves_pdf_page_metadata(self):
        content = ExtractedContent(
            filename="guide.pdf",
            source_type="pdf",
            text=(
                "[Page 1]\n"
                "Alpha retrieval systems split document text into chunks. "
                "Chunk overlap keeps neighboring context available.\n\n"
                "[Page 2]\n"
                "Beta vector search retrieves the most relevant chunks. "
                "Citations show which source supported the answer."
            ),
        )

        chunks = chunk_content(content, target_tokens=24, overlap_tokens=4)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual({chunk.page_number for chunk in chunks}, {1, 2})
        self.assertTrue(all(chunk.document_id for chunk in chunks))


class RagStoreTests(unittest.TestCase):
    def test_store_indexes_and_retrieves_relevant_chunks(self):
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        try:
            store = RagStore(
                persist_dir=temp_dir.name,
                collection_name="test_documents",
                chunk_tokens=40,
                chunk_overlap_tokens=5,
                embedding_dimensions=64,
            )
            result = store.ingest_contents(
                [
                    ExtractedContent(
                        filename="launch-notes.txt",
                        source_type="text",
                        text=(
                            "The launch deadline is Friday. "
                            "Maya owns the checklist. "
                            "The budget was already approved."
                        ),
                    )
                ]
            )

            hits = store.retrieve("What is the launch deadline?", top_k=2)
            documents = store.list_documents()
            store.close()
        finally:
            temp_dir.cleanup()

        self.assertEqual(result.chunks_indexed, 1)
        self.assertEqual(len(documents), 1)
        self.assertTrue(hits)
        self.assertEqual(hits[0].filename, "launch-notes.txt")
        self.assertIn("deadline", hits[0].text.lower())

    def test_query_rewriting_and_summary_indexing(self):
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        try:
            store = RagStore(
                persist_dir=temp_dir.name,
                collection_name="test_documents_summary",
                chunk_tokens=32,
                chunk_overlap_tokens=4,
                embedding_dimensions=64,
            )
            result = store.ingest_contents(
                [
                    ExtractedContent(
                        filename="roadmap.txt",
                        source_type="text",
                        text=(
                            "## Milestones\nThe release deadline is June 30. "
                            "The final review is due on July 2. "
                            "The team must sign off before the last launch."
                        ),
                    )
                ]
            )
            rewritten = store.rewrite_query("what about deadlines?")
            summary = store.summarize_document(result.documents[0].document_id)
            store.close()
        finally:
            temp_dir.cleanup()

        self.assertIn("deadline", rewritten.lower())
        self.assertIn("due", rewritten.lower())
        self.assertTrue(summary)
        self.assertIn("deadline", summary.lower())

    def test_reranker_prefers_exact_query_coverage(self):
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        try:
            store = RagStore(
                persist_dir=temp_dir.name,
                collection_name="test_documents_reranking",
                chunk_tokens=32,
                chunk_overlap_tokens=4,
                embedding_dimensions=64,
            )
            result = store._rerank_hits(
                "launch deadline",
                {
                    "related": {
                        "chunk_id": "related",
                        "text": "The launch schedule is being reviewed.",
                        "score": 0.9,
                    },
                    "exact": {
                        "chunk_id": "exact",
                        "text": "The launch deadline is Friday.",
                        "score": 0.8,
                    },
                },
            )
            store.close()
        finally:
            temp_dir.cleanup()

        self.assertEqual(result[0]["chunk_id"], "exact")
        self.assertGreater(result[0]["score"], result[1]["score"])


class FakeRagStore:
    def __init__(self):
        self.document_ids = None

    def rewrite_query(self, query):
        return query.strip()

    def ingest_contents(self, contents):
        return SimpleNamespace(
            documents=[
                RagDocument(
                    document_id="doc-1",
                    filename=contents[0].filename,
                    source_type=contents[0].source_type,
                    chunk_count=1,
                    token_count=20,
                )
            ],
            chunks_indexed=1,
        )

    def retrieve(self, query, *, top_k=None, document_ids=None):
        self.document_ids = document_ids
        return [
            RetrievedContext(
                chunk_id="chunk-1",
                document_id="doc-1",
                filename="policy.txt",
                source_type="text",
                text="The refund deadline is 30 days after purchase.",
                score=0.91,
                chunk_index=0,
            )
        ]


class RagPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_answer_uses_retrieved_chunks(self):
        fake_store = FakeRagStore()

        with (
            patch("app.agent.pipeline.extract_upload", new_callable=AsyncMock) as extract_upload,
            patch("app.agent.responses.ask_llm", return_value=None),
        ):
            extract_upload.return_value = ExtractedContent(
                filename="policy.txt",
                source_type="text",
                text="The refund deadline is 30 days after purchase.",
            )

            response = await AgentPipeline(rag_store=fake_store).run(
                query="What is the refund deadline?",
                files=[DummyUpload("policy.txt")],
            )

        self.assertEqual(fake_store.document_ids, ["doc-1"])
        self.assertEqual(response.status, "completed")
        self.assertEqual(len(response.retrieved_contexts), 1)
        self.assertIn("30 days", response.final_output)
        self.assertIn("Sources:", response.final_output)


if __name__ == "__main__":
    unittest.main()
