from __future__ import annotations

import base64
from typing import Any

import httpx

from app.config import settings

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _gemini_api_key() -> str | None:
    return settings.gemini_api_key


def generate_text(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> tuple[str | None, str | None]:
    api_key = _gemini_api_key()
    if not api_key:
        return None, "GEMINI_API_KEY is not configured."

    payload: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ]
    }
    generation_config: dict[str, Any] = {}
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if generation_config:
        payload["generationConfig"] = generation_config

    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
    try:
        response = httpx.post(url, params={"key": api_key}, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None, "Gemini returned no candidates."
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        return text or None, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def generate_transcript_from_audio(
    *,
    model: str,
    audio_bytes: bytes,
    mime_type: str,
    prompt: str,
) -> tuple[str | None, str | None]:
    api_key = _gemini_api_key()
    if not api_key:
        return None, "GEMINI_API_KEY is not configured."

    payload: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(audio_bytes).decode("utf-8"),
                        }
                    },
                ],
            }
        ]
    }
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
    try:
        response = httpx.post(url, params={"key": api_key}, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None, "Gemini returned no candidates."
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        return text or None, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
