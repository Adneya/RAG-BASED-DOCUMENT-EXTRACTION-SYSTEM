from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.schemas import ExtractedContent
from app.tools.text_processing import clean_text, estimate_tokens, split_sentences


PAGE_MARKER = re.compile(r"^\[Page\s+(\d+)(?:\s+OCR)?\]\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class RagChunk:
    id: str
    document_id: str
    filename: str
    source_type: str
    text: str
    chunk_index: int
    page_number: int | None
    token_count: int
    section_title: str | None = None


def document_id_for_content(content: ExtractedContent) -> str:
    digest = hashlib.sha256()
    digest.update(content.filename.encode("utf-8", errors="ignore"))
    digest.update(b"\0")
    digest.update(content.source_type.encode("utf-8", errors="ignore"))
    digest.update(b"\0")
    digest.update(content.text.encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:24]


def chunk_content(
    content: ExtractedContent,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[RagChunk]:
    text = clean_text(content.text)
    if not text:
        return []

    document_id = document_id_for_content(content)
    chunks: list[RagChunk] = []
    chunk_index = 0

    for page_number, page_text in iter_page_sections(text):
        for section_title, section_text in iter_section_blocks(page_text):
            for chunk_text in chunk_text_block(
                section_text,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            ):
                chunk_id = stable_chunk_id(document_id, chunk_index, chunk_text)
                chunks.append(
                    RagChunk(
                        id=chunk_id,
                        document_id=document_id,
                        filename=content.filename,
                        source_type=content.source_type,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        page_number=page_number,
                        token_count=estimate_tokens(chunk_text),
                        section_title=section_title,
                    )
                )
                chunk_index += 1

    return chunks


def iter_page_sections(text: str) -> list[tuple[int | None, str]]:
    matches = list(PAGE_MARKER.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[int | None, str]] = []
    if matches[0].start() > 0:
        prefix = clean_text(text[: matches[0].start()])
        if prefix:
            sections.append((None, prefix))

    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        page_text = clean_text(text[start:end])
        if page_text:
            sections.append((page_number, page_text))

    return sections


def iter_section_blocks(text: str) -> list[tuple[str | None, str]]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    blocks: list[tuple[str | None, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_lines
        body = clean_text("\n".join(current_lines))
        if body:
            blocks.append((current_title, body))
        current_lines = []

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines:
                current_lines.append("")
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            flush()
            current_title = clean_text(heading_match.group(2)) or current_title
            continue
        current_lines.append(line)

    flush()

    if not blocks:
        return [(None, cleaned)]
    return blocks


def chunk_text_block(
    text: str,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if estimate_tokens(text) <= target_tokens:
        return [text]

    sentences = split_sentences(text) or [text]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    target_tokens = max(40, target_tokens)
    overlap_tokens = max(0, min(overlap_tokens, target_tokens // 2))

    for sentence in sentences:
        sentence_parts = split_long_sentence(sentence, target_tokens)
        for part in sentence_parts:
            part_tokens = estimate_tokens(part)
            if current and current_tokens + part_tokens > target_tokens:
                chunk = clean_text(" ".join(current))
                chunks.append(chunk)
                overlap = trailing_words(chunk, overlap_tokens)
                current = [overlap] if overlap else []
                current_tokens = estimate_tokens(overlap) if overlap else 0

            current.append(part)
            current_tokens += part_tokens

    if current:
        chunks.append(clean_text(" ".join(current)))

    return [chunk for chunk in chunks if chunk]


def split_long_sentence(sentence: str, target_tokens: int) -> list[str]:
    if estimate_tokens(sentence) <= target_tokens:
        return [sentence]

    words = sentence.split()
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for word in words:
        word_tokens = estimate_tokens(word)
        if current and current_tokens + word_tokens > target_tokens:
            parts.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += word_tokens

    if current:
        parts.append(" ".join(current))
    return parts


def trailing_words(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    words = text.split()
    return " ".join(words[-overlap_tokens:])


def stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{document_id}-{chunk_index:04d}-{digest}"
