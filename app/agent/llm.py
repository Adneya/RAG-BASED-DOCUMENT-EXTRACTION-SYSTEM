from __future__ import annotations

from app.agent.gemini import generate_text
from app.config import settings

_last_error: str | None = None


def llm_available() -> bool:
    return bool(settings.gemini_api_key)


def last_llm_error() -> str | None:
    return _last_error


def ask_llm(system_prompt: str, user_prompt: str) -> str | None:
    global _last_error
    _last_error = None

    text, error = generate_text(
        model=settings.gemini_chat_model,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.2,
    )
    if error:
        _last_error = error
    return text
