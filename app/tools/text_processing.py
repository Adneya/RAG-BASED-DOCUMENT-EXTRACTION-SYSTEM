from __future__ import annotations

import re
from collections import Counter


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip(" -\t") for part in parts if part.strip(" -\t")]


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(",.;:") + "..."


def keywords(text: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    counts = Counter(word for word in words if word not in STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def summarize_text(text: str) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return (
            "1-line summary: No readable text was available.\n\n"
            "3 bullets:\n"
            "- No content could be summarized.\n"
            "- Try uploading a clearer file or adding a more specific prompt.\n"
            "- The agent will summarize once text is available.\n\n"
            "5-sentence summary:\nNo readable text was available."
        )

    ranked = rank_sentences(text, sentences)
    one_line = truncate_words(ranked[0], 28)
    bullets = [truncate_words(sentence, 24) for sentence in ranked[:3]]
    while len(bullets) < 3:
        bullets.append("Additional detail was not present in the available text.")

    five = [truncate_words(sentence, 32) for sentence in ranked[:5]]
    while len(five) < 5:
        five.append("No further distinct detail was available.")

    return (
        f"1-line summary: {one_line}\n\n"
        "3 bullets:\n"
        + "\n".join(f"- {bullet}" for bullet in bullets)
        + "\n\n5-sentence summary:\n"
        + " ".join(sentence.rstrip(".!?") + "." for sentence in five)
    )


def rank_sentences(text: str, sentences: list[str]) -> list[str]:
    key_terms = keywords(text, limit=20)
    if not key_terms:
        return sentences
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        score = sum(lowered.count(term) for term in key_terms)
        score += 1 if 8 <= len(sentence.split()) <= 35 else 0
        scored.append((score, -index, sentence))
    ranked = [sentence for _, _, sentence in sorted(scored, reverse=True)]
    return ranked or sentences


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def extract_relevant_sentences(text: str, query: str, limit: int = 6) -> list[str]:
    sentences = split_sentences(text)
    terms = set(keywords(query, limit=10))
    if not terms:
        return sentences[:limit]
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        score = sum(1 for term in terms if term in lowered)
        if score:
            scored.append((score, -index, sentence))
    if not scored:
        return sentences[:limit]
    return [sentence for _, _, sentence in sorted(scored, reverse=True)[:limit]]

