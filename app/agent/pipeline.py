from __future__ import annotations

from fastapi import UploadFile

from app.agent.intent import infer_intent
from app.agent.responses import estimate_text_tokens, render_response
from app.config import settings
from app.schemas import AgentResponse, CostEstimate, ExtractedContent, PlanStep
from app.tools.extraction import extract_upload
from app.tools.text_processing import clean_text
from app.tools.youtube import detect_youtube_urls, fetch_youtube_transcript


class AgentPipeline:
    async def run(
        self,
        query: str,
        files: list[UploadFile] | None,
    ) -> AgentResponse:
        plan: list[PlanStep] = []
        extracted: list[ExtractedContent] = []
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

        combined = clean_text(query + "\n\n" + "\n\n".join(item.text for item in extracted))
        youtube_urls = detect_youtube_urls(combined)
        plan.append(
            PlanStep(
                tool="intent_router",
                action="Infer user goal and required tool sequence",
                status="running",
            )
        )
        decision = infer_intent(query, extracted, youtube_urls)
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

        plan.append(
            PlanStep(
                tool="response_builder",
                action=f"Generate final text output for intent: {decision.intent}",
                status="running",
            )
        )
        final_output = render_response(decision.intent, query, extracted)
        plan[-1].status = "completed"
        plan[-1].detail = "Final text-only output generated."

        status = "partial" if any(item.warnings for item in extracted) else "completed"
        return AgentResponse(
            status=status,
            final_output=final_output,
            extracted_texts=extracted,
            plan_trace=plan,
            cost_estimate=build_cost_estimate(query, extracted),
        )


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

