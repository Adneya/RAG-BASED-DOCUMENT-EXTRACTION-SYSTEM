from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_file(path: Path | None = None) -> None:
    env_path = path or Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Multi-Input Agent")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    gemini_chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    gemini_audio_model: str = os.getenv("GEMINI_AUDIO_MODEL", "gemini-2.5-flash")
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
