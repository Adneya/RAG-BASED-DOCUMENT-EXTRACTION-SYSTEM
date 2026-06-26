from __future__ import annotations

import re
from collections import Counter

from app.agent.intent import looks_like_code
from app.agent.llm import ask_llm
from app.schemas import ExtractedContent
from app.tools.text_processing import (
    clean_text,
    estimate_tokens,
    extract_relevant_sentences,
    keywords,
    split_sentences,
    summarize_text,
    truncate_words,
)


POSITIVE_WORDS = {
    "achieve",
    "benefit",
    "best",
    "clear",
    "excellent",
    "good",
    "great",
    "happy",
    "improve",
    "love",
    "positive",
    "success",
    "useful",
    "win",
}
NEGATIVE_WORDS = {
    "bad",
    "blocker",
    "bug",
    "confusing",
    "delay",
    "fail",
    "issue",
    "negative",
    "problem",
    "risk",
    "sad",
    "slow",
    "worse",
    "wrong",
}


def render_response(intent: str, query: str, contents: list[ExtractedContent]) -> str:
    text = join_content(contents)
    if intent == "extract_text":
        return format_extracted_text(contents)
    if intent in {"summary", "audio_summary", "youtube_summary"}:
        summary = summarize_text(text or task_payload(query))
        if intent == "audio_summary":
            duration_lines = [
                f"{item.filename}: {item.duration_seconds} seconds"
                for item in contents
                if item.source_type == "audio" and item.duration_seconds is not None
            ]
            if duration_lines:
                summary += "\n\nDuration:\n" + "\n".join(f"- {line}" for line in duration_lines)
        return summary
    if intent == "sentiment":
        return sentiment_analysis(text or query)
    if intent == "code_explanation":
        return explain_code(text or query)
    if intent == "compare_sources":
        return compare_sources(contents)
    if intent == "action_items":
        return action_items(text or query)
    if intent == "context_answer":
        return answer_from_context(query, contents)
    return conversational_answer(query)


def join_content(contents: list[ExtractedContent]) -> str:
    return clean_text(
        "\n\n".join(
            f"Source: {item.filename}\n{item.text}" for item in contents if item.text.strip()
        )
    )


def format_extracted_text(contents: list[ExtractedContent]) -> str:
    if not contents:
        return "No files were uploaded."

    blocks: list[str] = []
    for item in contents:
        confidence = (
            f" OCR confidence: {item.ocr_confidence:.2f}."
            if item.ocr_confidence is not None
            else ""
        )
        warnings = (
            "\nWarnings:\n" + "\n".join(f"- {warning}" for warning in item.warnings)
            if item.warnings
            else ""
        )
        body = item.text or "No readable text was extracted."
        blocks.append(
            f"{item.filename} ({item.source_type}).{confidence}\n{body}{warnings}"
        )
    return "\n\n".join(blocks)


def sentiment_analysis(text: str) -> str:
    words = re.findall(r"[A-Za-z']+", text.lower())
    counts = Counter(words)
    positive = sum(counts[word] for word in POSITIVE_WORDS)
    negative = sum(counts[word] for word in NEGATIVE_WORDS)
    total = positive + negative

    if positive > negative:
        label = "Positive"
        confidence = min(0.95, 0.55 + (positive - negative) / max(total + 4, 1))
        cue = "positive cues outweighed negative cues"
    elif negative > positive:
        label = "Negative"
        confidence = min(0.95, 0.55 + (negative - positive) / max(total + 4, 1))
        cue = "negative cues outweighed positive cues"
    else:
        label = "Neutral"
        confidence = 0.58 if total else 0.5
        cue = "the text did not contain enough polarized language"

    return (
        f"Label: {label}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Justification: {cue}."
    )


def explain_code(text: str) -> str:
    if not looks_like_code(text):
        return (
            "I do not see enough code-like structure to explain. "
            "Please provide a clearer code snippet or image."
        )

    language = detect_language(text)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    functions = detect_functions(text)
    bugs = detect_code_risks(text)
    complexity = estimate_time_complexity(text)

    explanation = [
        f"Language: {language}",
        f"What it does: The snippet has {len(lines)} non-empty lines"
        + (
            f" and defines {', '.join(functions[:5])}."
            if functions
            else ". It appears to run top-level logic rather than named functions."
        ),
        "Flow: " + summarize_code_flow(lines),
        "Potential bugs or risks:",
    ]
    explanation.extend(f"- {bug}" for bug in bugs)
    explanation.append(f"Time complexity: {complexity}")
    return "\n".join(explanation)


def detect_language(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bdef\s+\w+\s*\(", text):
        return "Python"
    if "console.log" in lowered or re.search(r"\bfunction\s+\w*\s*\(", text):
        return "JavaScript/TypeScript"
    if "#include" in lowered:
        return "C or C++"
    if "public static void main" in lowered:
        return "Java"
    if re.search(r"\bselect\b.+\bfrom\b", lowered, re.DOTALL):
        return "SQL"
    return "Unknown"


def detect_functions(text: str) -> list[str]:
    patterns = [
        r"\bdef\s+(\w+)\s*\(",
        r"\bfunction\s+(\w+)\s*\(",
        r"\bclass\s+(\w+)",
    ]
    names: list[str] = []
    for pattern in patterns:
        names.extend(re.findall(pattern, text))
    return names


def detect_code_risks(text: str) -> list[str]:
    risks: list[str] = []
    lowered = text.lower()
    if "eval(" in lowered:
        risks.append("Uses eval, which can execute untrusted input.")
    if "except:" in lowered or "catch (" in lowered and not re.search(r"console|throw|log", lowered):
        risks.append("Exception handling may hide failures without reporting them.")
    if re.search(r"/\s*0(?!\d)", text):
        risks.append("Contains an apparent division by zero.")
    if re.search(r"while\s+(true|1)", lowered):
        risks.append("Contains an unbounded loop; verify the exit condition.")
    if "password" in lowered and ("print(" in lowered or "console.log" in lowered):
        risks.append("May print sensitive data.")
    if not risks:
        risks.append("No obvious bug was detected by the local static checks.")
    return risks


def estimate_time_complexity(text: str) -> str:
    lines = text.splitlines()
    max_depth = 0
    current_depth = 0
    for line in lines:
        stripped = line.strip()
        indent_depth = (len(line) - len(line.lstrip(" "))) // 4
        loop_here = bool(re.match(r"(for|while)\b", stripped)) or bool(
            re.match(r"(for|while)\s*\(", stripped)
        )
        if loop_here:
            current_depth = max(current_depth, indent_depth) + 1
            max_depth = max(max_depth, current_depth)
        else:
            current_depth = min(current_depth, indent_depth)
    if max_depth >= 3:
        return "Likely O(n^3) or worse because nested loops were detected."
    if max_depth == 2:
        return "Likely O(n^2) because two nested loops were detected."
    if max_depth == 1:
        return "Likely O(n) because a single loop over input-like data was detected."
    return "Likely O(1) for the visible code, unless called functions hide additional work."


def summarize_code_flow(lines: list[str]) -> str:
    preview = " ".join(line.strip() for line in lines[:5])
    return truncate_words(preview, 35)


def compare_sources(contents: list[ExtractedContent]) -> str:
    usable = [item for item in contents if item.text.strip()]
    if len(usable) < 2:
        return "At least two readable inputs are needed for comparison."

    keyword_sets = [set(keywords(item.text, limit=20)) for item in usable]
    shared = set.intersection(*keyword_sets) if keyword_sets else set()
    union = set.union(*keyword_sets) if keyword_sets else set()
    similarity = len(shared) / len(union) if union else 0.0

    if similarity >= 0.25:
        verdict = "They appear to discuss the same broad topic."
    elif similarity >= 0.1:
        verdict = "They partially overlap, but they are not tightly aligned."
    else:
        verdict = "They appear to discuss different topics."

    source_lines = [
        f"- {item.filename}: {', '.join(keywords(item.text, limit=6)) or 'no clear keywords'}"
        for item in usable
    ]
    shared_terms = ", ".join(sorted(shared)[:12]) or "no strong shared terms"
    return (
        f"Verdict: {verdict}\n"
        f"Similarity score: {similarity:.2f}\n"
        f"Shared terms: {shared_terms}\n\n"
        "Source signals:\n"
        + "\n".join(source_lines)
    )


def action_items(text: str) -> str:
    candidates = []
    action_terms = (
        "action",
        "todo",
        "to-do",
        "owner",
        "due",
        "assign",
        "follow up",
        "next step",
        "need to",
        "needs to",
        "must",
        "should",
        "will",
    )
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if any(term in lowered for term in action_terms):
            candidates.append(truncate_words(sentence, 32))

    if not candidates:
        return "Action items:\n- No explicit action items were found in the extracted text."
    return "Action items:\n" + "\n".join(f"- {item}" for item in candidates[:12])


def answer_from_context(query: str, contents: list[ExtractedContent]) -> str:
    text = join_content(contents)
    llm_answer = ask_llm(
        "Answer using only the provided context. Keep the answer text-only and concise.",
        f"Question: {query}\n\nContext:\n{text}",
    )
    if llm_answer:
        return clean_text(llm_answer)

    relevant = extract_relevant_sentences(text, query, limit=6)
    if not relevant:
        return "I could not find an answer in the extracted content."
    return "Based on the extracted content:\n" + "\n".join(
        f"- {truncate_words(sentence, 35)}" for sentence in relevant
    )


def conversational_answer(query: str) -> str:
    llm_answer = ask_llm(
        "You are a friendly, helpful assistant. Return text only.",
        query,
    )
    if llm_answer:
        return clean_text(llm_answer)
    return (
        "I can help with that. In this local fallback mode, open-ended general answers "
        "are limited unless OPENAI_API_KEY is configured. You can still upload text, "
        "images, PDFs, or audio and ask for extraction, summaries, sentiment, code "
        "explanation, action items, or cross-input comparison."
    )


def task_payload(query: str) -> str:
    cleaned = re.sub(
        r"^\s*(summari[sz]e|summary|tl;dr|tldr|recap|analy[sz]e|explain)\s*(this|the)?\s*:?\s*",
        "",
        query,
        flags=re.IGNORECASE,
    )
    return cleaned.strip() or query


def estimate_text_tokens(query: str, contents: list[ExtractedContent]) -> int:
    return estimate_tokens(query + "\n\n" + join_content(contents))
