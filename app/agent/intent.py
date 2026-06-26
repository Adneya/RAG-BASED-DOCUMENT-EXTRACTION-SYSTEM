from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.schemas import ExtractedContent


IntentName = Literal[
    "clarify",
    "extract_text",
    "youtube_summary",
    "summary",
    "sentiment",
    "code_explanation",
    "audio_summary",
    "compare_sources",
    "action_items",
    "context_answer",
    "conversation",
]


@dataclass(frozen=True)
class IntentDecision:
    intent: IntentName
    reason: str
    follow_up_question: str | None = None


AMBIGUOUS_FILE_QUERIES = {
    "",
    "analyze",
    "analyse",
    "process",
    "review",
    "check",
    "help",
    "do this",
    "do it",
    "handle this",
}


def infer_intent(
    query: str,
    extracted: list[ExtractedContent],
    youtube_urls: list[str],
) -> IntentDecision:
    query_clean = (query or "").strip()
    query_lower = query_clean.lower()
    has_files = bool(extracted)
    has_text = any(item.text.strip() for item in extracted)
    has_audio = any(item.source_type == "audio" for item in extracted)

    if not query_clean and not has_files:
        return IntentDecision(
            "clarify",
            "No query or input content was provided.",
            "What would you like me to help with?",
        )

    if has_files and query_lower in AMBIGUOUS_FILE_QUERIES:
        return IntentDecision(
            "clarify",
            "Files were provided, but the requested task is unclear.",
            "What do you want me to do with the uploaded content?",
        )

    if youtube_urls and asks_for_youtube(query_lower):
        return IntentDecision(
            "youtube_summary",
            "A YouTube URL was found and the query asks to use it.",
        )

    if asks_for_comparison(query_lower) and len([item for item in extracted if item.text]) >= 2:
        return IntentDecision(
            "compare_sources",
            "The query asks for cross-input comparison across multiple sources.",
        )

    if asks_for_action_items(query_lower):
        return IntentDecision(
            "action_items",
            "The query asks for action items or tasks.",
        )

    if asks_for_sentiment(query_lower):
        target = "uploaded content" if has_text else "query text"
        return IntentDecision("sentiment", f"The query asks for sentiment on {target}.")

    if asks_for_code_explanation(query_lower) or (
        "explain" in query_lower and looks_like_code(combined_text(extracted))
    ):
        return IntentDecision(
            "code_explanation",
            "The query asks for code explanation or code-like text was detected.",
        )

    if has_audio and asks_for_summary(query_lower):
        return IntentDecision(
            "audio_summary",
            "Audio input was provided and the query asks for a summary.",
        )

    if asks_for_summary(query_lower):
        return IntentDecision("summary", "The query asks for summarization.")

    if asks_for_extraction(query_lower):
        return IntentDecision("extract_text", "The query asks for raw extracted text.")

    if has_text and query_clean:
        return IntentDecision(
            "context_answer",
            "The query asks a question over the available uploaded content.",
        )

    if not has_text and has_files:
        return IntentDecision(
            "clarify",
            "The uploaded files did not yield usable text.",
            "I could not extract usable text. Would you like to try another file or provide a text query?",
        )

    return IntentDecision("conversation", "No files are needed; answer conversationally.")


def combined_text(extracted: list[ExtractedContent]) -> str:
    return "\n\n".join(item.text for item in extracted if item.text)


def asks_for_youtube(query_lower: str) -> bool:
    terms = ("youtube", "yt", "video", "transcript", "url", "link", "hit")
    summary_terms = ("summary", "summarize", "summarise", "recap", "explain")
    return any(term in query_lower for term in terms) and (
        any(term in query_lower for term in summary_terms) or "transcript" in query_lower
    )


def asks_for_summary(query_lower: str) -> bool:
    return any(
        term in query_lower
        for term in ("summary", "summarize", "summarise", "recap", "tl;dr", "tldr")
    )


def asks_for_sentiment(query_lower: str) -> bool:
    return any(
        term in query_lower
        for term in ("sentiment", "tone", "positive", "negative", "neutral")
    )


def asks_for_extraction(query_lower: str) -> bool:
    return any(
        term in query_lower
        for term in ("extract", "transcript", "ocr", "read text", "raw text")
    )


def asks_for_code_explanation(query_lower: str) -> bool:
    return any(
        term in query_lower
        for term in (
            "explain code",
            "explain this code",
            "what does this code",
            "detect bugs",
            "time complexity",
            "rewrite code",
        )
    )


def asks_for_comparison(query_lower: str) -> bool:
    return any(
        term in query_lower
        for term in (
            "same topic",
            "compare",
            "difference",
            "different",
            "similar",
            "overlap",
            "both discuss",
        )
    )


def asks_for_action_items(query_lower: str) -> bool:
    return any(
        term in query_lower
        for term in ("action item", "action items", "todo", "to-do", "tasks", "next steps")
    )


def looks_like_code(text: str) -> bool:
    if not text:
        return False
    code_patterns = [
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\bfunction\s+\w*\s*\(",
        r"\bfor\s*\(",
        r"\bwhile\s*\(",
        r"\bif\s*\(",
        r"\bimport\s+\w+",
        r"#include\s*<",
        r"console\.log",
        r"SELECT\s+.+\s+FROM",
    ]
    if any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in code_patterns):
        return True
    symbols = sum(text.count(char) for char in "{}();=<>")
    return symbols >= 8 and "\n" in text

