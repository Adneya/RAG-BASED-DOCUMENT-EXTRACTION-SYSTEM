from __future__ import annotations

import os
import tempfile
import wave
from io import BytesIO
from pathlib import Path

from app.config import settings
from app.tools.text_processing import clean_text


def audio_duration_seconds(data: bytes, filename: str) -> float | None:
    suffix = Path(filename).suffix.lower()
    if suffix != ".wav":
        return None
    try:
        with wave.open(BytesIO(data), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return round(frames / float(rate), 2) if rate else None
    except Exception:
        return None


def transcribe_audio(data: bytes, filename: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if not settings.openai_api_key:
        return (
            "",
            [
                "Audio transcription requires OPENAI_API_KEY. "
                "Set it in the deployment environment to enable MP3, WAV, and M4A transcription."
            ],
        )

    temp_path: str | None = None
    suffix = Path(filename).suffix or ".audio"
    try:
        from openai import OpenAI

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(data)
            temp_path = temp.name

        client = OpenAI(api_key=settings.openai_api_key)
        with open(temp_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=settings.openai_transcription_model,
                file=audio_file,
                response_format="text",
            )
        return clean_text(str(result)), warnings
    except Exception as exc:  # pragma: no cover - external service
        warnings.append(f"Audio transcription failed: {exc}")
        return "", warnings
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

