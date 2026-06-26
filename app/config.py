from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Multi-Input Agent")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_transcription_model: str = os.getenv(
        "OPENAI_TRANSCRIPTION_MODEL", "whisper-1"
    )
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    youtube_languages: tuple[str, ...] = tuple(
        part.strip()
        for part in os.getenv("YOUTUBE_TRANSCRIPT_LANGS", "en,en-US,en-GB").split(",")
        if part.strip()
    )
    input_price_per_million_tokens: float = float(
        os.getenv("INPUT_PRICE_PER_MILLION_TOKENS", "0")
    )
    output_price_per_million_tokens: float = float(
        os.getenv("OUTPUT_PRICE_PER_MILLION_TOKENS", "0")
    )


settings = Settings()
