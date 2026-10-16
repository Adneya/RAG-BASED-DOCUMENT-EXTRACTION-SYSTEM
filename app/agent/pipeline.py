from __future__ import annotations

import re

from fastapi import UploadFile

from app.agent.intent import infer_intent
from app.agent.responses import answer_from_rag, build_research_perspectives, estimate_text_tokens, render_response
from app.config import settings
from app.rag.store import RagStore, get_default_rag_store
from app.schemas import (
    AgentResponse,
    CostEstimate,
    ExtractedContent,
    PlanStep,
    RetrievedContext,
)
from app.tools.extraction import extract_upload
from app.tools.text_processing import clean_text, keywords
from app.tools.youtube import detect_youtube_urls, fetch_youtube_transcript


class AgentPipeline:
    def __init__(self, rag_store: RagStore | None = None) -> None:
        self._rag_store = rag_store
        self.conversation_memory: list[str] = []

    def _prepare_query(self, query: str) -> str:
        rewritten = self.rag_store.rewrite_query(query)
        if not rewritten:
            return query.strip()
        memory_context = "\n".join(self.conversation_memory[-3:])
        if memory_context and re.match(r"^(it|that|this|they|those|these|he|she|the)\b", query.strip(), re.IGNORECASE):
            return f"{rewritten} Context: {memory_context}"
        return rewritten

    @property
    def rag_store(self) -> RagStore:
        if self._rag_store is None:
            self._rag_store = get_default_rag_store()
        return self._rag_store

    async def run(
        self,
        query: str,
        files: list[UploadFile] | None,
    ) -> AgentResponse:
        plan: list[PlanStep] = []
        extracted: list[ExtractedContent] = []
        indexed_document_ids: list[str] = []
        uploads = [file for file in files or [] if file.filename]

        if uploads:
            plan.append(
                PlanStep(
                    tool="input_extractor",
                    action=f"Extract content from {len(uploads)} uploaded file(s)",
                    status="running",
                )
            )
            for upload in uploads:
                extracted.append(await extract_upload(upload))
            plan[-1].status = "completed"
            plan[-1].detail = "Parsed text, image OCR, PDF text/OCR, or audio transcription as applicable."
        else:
            plan.append(
                PlanStep(
                    tool="input_extractor",
                    action="No uploaded files to extract",
                    status="skipped",
                )
            )

        if extracted:
            indexed_document_ids.extend(
                self.index_contents(
                    extracted,
                    plan,
                    action="Index uploaded content for retrieval",
                )
            )

        search_query = self._prepare_query(query)
        combined = clean_text(search_query + "\n\n" + "\n\n".join(item.text for item in extracted))
        youtube_urls = detect_youtube_urls(combined)
        plan.append(
            PlanStep(
                tool="intent_router",
                action="Infer user goal and required tool sequence",
                status="running",
            )
        )
        decision = infer_intent(search_query, extracted, youtube_urls)
        plan[-1].status = "completed"
        plan[-1].detail = decision.reason

        if decision.intent == "clarify":
            plan.append(
                PlanStep(
                    tool="follow_up_guard",
                    action="Stop and ask for clarification",
                    status="completed",
                    detail="The mandatory follow-up rule was triggered.",
                )
            )
            return AgentResponse(
                status="needs_clarification",
                follow_up_question=decision.follow_up_question,
                final_output=decision.follow_up_question or "Could you clarify the task?",
                extracted_texts=extracted,
                plan_trace=plan,
                cost_estimate=build_cost_estimate(query, extracted),
                perspectives=build_research_perspectives(query, decision.follow_up_question or "Could you clarify the task?", []),
            )

        if decision.intent == "youtube_summary":
            plan.append(
                PlanStep(
                    tool="youtube_transcript",
                    action=f"Fetch transcript for {len(youtube_urls)} detected YouTube URL(s)",
                    status="running",
                )
            )
            youtube_contents: list[ExtractedContent] = []
            for url in youtube_urls:
                transcript = fetch_youtube_transcript(url)
                youtube_contents.append(
                    ExtractedContent(
                        filename=f"YouTube {transcript.video_id or url}",
                        source_type="youtube",
                        text=transcript.text,
                        warnings=transcript.warnings,
                    )
                )
            extracted.extend(youtube_contents)
            failures = sum(1 for item in youtube_contents if not item.text)
            plan[-1].status = "partial" if failures else "completed"  # type: ignore[assignment]
            plan[-1].detail = (
                f"{failures} transcript(s) could not be fetched."
                if failures
                else "Transcript fetch completed."
            )
            indexed_document_ids.extend(
                self.index_contents(
                    youtube_contents,
                    plan,
                    action="Index YouTube transcript content for retrieval",
                )
            )

        retrieved_contexts: list[RetrievedContext] = []
        response_intent = decision.intent
        final_output: str | None = None

        if decision.intent == "context_answer":
            retrieved_contexts = self.retrieve_contexts(
                search_query,
                plan,
                document_ids=indexed_document_ids or None,
            )
            if retrieved_contexts:
                final_output = answer_from_rag(search_query, retrieved_contexts)
            else:
                final_output = render_response(decision.intent, search_query, extracted)

        elif decision.intent == "conversation":
            retrieved_contexts = self.retrieve_contexts(search_query, plan)
            if should_use_rag(search_query, retrieved_contexts):
                response_intent = "context_answer"
                final_output = answer_from_rag(search_query, retrieved_contexts)
            else:
                retrieved_contexts = []
                final_output = render_response(decision.intent, search_query, extracted)

        plan.append(
            PlanStep(
                tool="response_builder",
                action=f"Generate final text output for intent: {response_intent}",
                status="running",
            )
        )
        if final_output is None:
            final_output = render_response(decision.intent, query, extracted)
        plan[-1].status = "completed"
        plan[-1].detail = "Final text-only output generated."

        if final_output:
            self.conversation_memory.append(f"Q: {query}\nA: {final_output}")
            if len(self.conversation_memory) > 5:
                self.conversation_memory = self.conversation_memory[-5:]

        status = "partial" if any(item.warnings for item in extracted) else "completed"
        return AgentResponse(
            status=status,
            final_output=final_output,
            extracted_texts=extracted,
            retrieved_contexts=retrieved_contexts,
            perspectives=build_research_perspectives(query, final_output, retrieved_contexts),
            plan_trace=plan,
            cost_estimate=build_cost_estimate(query, extracted),
        )

    def index_contents(
        self,
        contents: list[ExtractedContent],
        plan: list[PlanStep],
        *,
        action: str,
    ) -> list[str]:
        plan.append(
            PlanStep(
                tool="rag_indexer",
                action=action,
                status="running",
            )
        )
        try:
            result = self.rag_store.ingest_contents(contents)
        except Exception as exc:
            plan[-1].status = "failed"
            plan[-1].detail = f"RAG indexing failed: {type(exc).__name__}: {exc}"
            return []

        if result.chunks_indexed == 0:
            plan[-1].status = "skipped"
            plan[-1].detail = "No readable text was available to index."
            return []

        plan[-1].status = "completed"
        plan[-1].detail = (
            f"Indexed {result.chunks_indexed} chunk(s) across "
            f"{len(result.documents)} document(s)."
        )
        return [document.document_id for document in result.documents]

    def retrieve_contexts(
        self,
        query: str,
        plan: list[PlanStep],
        *,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedContext]:
        plan.append(
            PlanStep(
                tool="rag_retriever",
                action="Retrieve relevant indexed chunks",
                status="running",
            )
        )
        try:
            contexts = self.rag_store.retrieve(
                query,
                top_k=settings.rag_top_k,
                document_ids=document_ids,
            )
        except Exception as exc:
            plan[-1].status = "failed"
            plan[-1].detail = f"RAG retrieval failed: {type(exc).__name__}: {exc}"
            return []

        if not contexts:
            plan[-1].status = "skipped"
            plan[-1].detail = "No relevant indexed chunks were found."
            return []

        plan[-1].status = "completed"
        plan[-1].detail = f"Retrieved {len(contexts)} chunk(s) for grounded answering."
        return contexts


def build_cost_estimate(query: str, contents: list[ExtractedContent]) -> CostEstimate:
    input_tokens = estimate_text_tokens(query, contents)
    output_tokens = 700
    estimated_usd: float | None = None
    note = "Token estimate only. Set price env vars to estimate USD for your chosen model."
    if (
        settings.input_price_per_million_tokens > 0
        or settings.output_price_per_million_tokens > 0
    ):
        estimated_usd = round(
            (input_tokens / 1_000_000 * settings.input_price_per_million_tokens)
            + (output_tokens / 1_000_000 * settings.output_price_per_million_tokens),
            6,
        )
        note = "USD estimate uses INPUT_PRICE_PER_MILLION_TOKENS and OUTPUT_PRICE_PER_MILLION_TOKENS."

    return CostEstimate(
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_usd=estimated_usd,
        note=note,
    )


def should_use_rag(query: str, contexts: list[RetrievedContext]) -> bool:
    if not contexts:
        return False

    query_terms = set(keywords(query, limit=12))
    if not query_terms:
        return False

    for context in contexts[:3]:
        lowered = context.text.lower()
        if any(term in lowered for term in query_terms):
            return True
    return False

