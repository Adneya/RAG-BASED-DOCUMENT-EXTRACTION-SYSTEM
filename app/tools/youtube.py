from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from app.config import settings
from app.tools.text_processing import clean_text


YOUTUBE_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/[^\s<>)]+|youtu\.be/[^\s<>)]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class YouTubeTranscript:
    url: str
    video_id: str
    text: str
    warnings: list[str]


def detect_youtube_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in YOUTUBE_URL_PATTERN.finditer(text or ""):
        url = match.group(0).rstrip(".,;:)]}")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def extract_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "youtu.be":
        return parsed.path.strip("/") or None
    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            return parts[1]
    return None


def _transcript_text_item(item: object) -> str:
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(getattr(item, "text", ""))


def _transcript_items(transcript: object) -> object:
    to_raw_data = getattr(transcript, "to_raw_data", None)
    if callable(to_raw_data):
        return to_raw_data()
    return transcript


def _fetch_transcript(api_class: type, video_id: str, languages: list[str]) -> object:
    legacy_get_transcript = getattr(api_class, "get_transcript", None)
    if callable(legacy_get_transcript):
        return legacy_get_transcript(video_id, languages=languages)

    return api_class().fetch(video_id, languages=languages)


def fetch_youtube_transcript(url: str) -> YouTubeTranscript:
    video_id = extract_video_id(url)
    if not video_id:
        return YouTubeTranscript(url, "", "", ["Could not detect the YouTube video id."])

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception as exc:  # pragma: no cover - optional dependency
        return YouTubeTranscript(
            url,
            video_id,
            "",
            [f"YouTube transcript dependency is unavailable: {exc}"],
        )

    try:
        transcript = _fetch_transcript(
            YouTubeTranscriptApi, video_id, list(settings.youtube_languages)
        )
        text = clean_text(" ".join(_transcript_text_item(item) for item in _transcript_items(transcript)))
        return YouTubeTranscript(url, video_id, text, [])
    except Exception as exc:  # pragma: no cover - network/external content
        return YouTubeTranscript(
            url,
            video_id,
            "",
            [f"Transcript could not be fetched for this video: {exc}"],
        )
