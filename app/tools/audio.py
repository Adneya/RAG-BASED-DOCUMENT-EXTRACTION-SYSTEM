from __future__ import annotations

import mimetypes
import os
import subprocess
import tempfile
import wave
from io import BytesIO
from pathlib import Path

from app.agent.gemini import generate_transcript_from_audio
from app.config import settings
from app.tools.text_processing import clean_text


def audio_duration_seconds(data: bytes, filename: str) -> float | None:
    suffix = Path(filename).suffix.lower()
    wav_duration = wav_duration_seconds(data) if suffix == ".wav" else None
    if wav_duration is not None:
        return wav_duration
    return ffprobe_duration_seconds(data, filename)


def wav_duration_seconds(data: bytes) -> float | None:
    try:
        with wave.open(BytesIO(data), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return round(frames / float(rate), 2) if rate else None
    except Exception:
        return None


def ffprobe_duration_seconds(data: bytes, filename: str) -> float | None:
    suffix = Path(filename).suffix or ".audio"
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(data)
            temp_path = temp.name

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                temp_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return None
        return round(float(result.stdout.strip()), 2)
    except Exception:
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def transcribe_audio(data: bytes, filename: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if not settings.gemini_api_key:
        return (
            "",
            [
                "Audio transcription requires GEMINI_API_KEY. "
                "Set it in the deployment environment to enable MP3, WAV, and M4A transcription."
            ],
        )

    mime_type = mimetypes.guess_type(filename)[0] or "audio/mpeg"
    prompt = "Generate a verbatim transcript of the audio. Return only the transcript text."
    try:
        transcript, error = generate_transcript_from_audio(
            model=settings.gemini_audio_model,
            audio_bytes=data,
            mime_type=mime_type,
            prompt=prompt,
        )
        if error:
            raise RuntimeError(error)
        if len(data) > 20 * 1024 * 1024:
            warnings.append(
                "Audio file is larger than 20 MB, so inline Gemini transcription may fail. "
                "Consider trimming the file or splitting it into smaller clips."
            )
        return clean_text(transcript or ""), warnings
    except Exception as exc:  # pragma: no cover - external service
        warnings.append(f"Audio transcription failed: {exc}")
        return "", warnings
