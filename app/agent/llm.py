from __future__ import annotations

from app.config import settings


def llm_available() -> bool:
    return bool(settings.openai_api_key)


def ask_llm(system_prompt: str, user_prompt: str) -> str | None:
    if not settings.openai_api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return None

