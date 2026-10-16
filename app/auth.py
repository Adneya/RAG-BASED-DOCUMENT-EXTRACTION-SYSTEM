from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import settings

DB_PATH = Path(settings.auth_db_path)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_auth() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL
            );
            """
        )


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        salt_hex, digest_hex = encoded.split("$", 1)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 240_000)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def create_user(name: str, email: str, password: str) -> dict[str, str]:
    user_id = secrets.token_urlsafe(12)
    created_at = datetime.now(timezone.utc).isoformat()
    with connect() as connection:
        connection.execute("INSERT INTO users (id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)", (user_id, name.strip(), email.strip().lower(), hash_password(password), created_at))
    return {"id": user_id, "name": name.strip(), "email": email.strip().lower()}


def authenticate(email: str, password: str) -> dict[str, str] | None:
    with connect() as connection:
        user = connection.execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    if not user or not verify_password(password, user["password_hash"]):
        return None
    return {"id": user["id"], "name": user["name"], "email": user["email"]}


def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    with connect() as connection:
        connection.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)", (token, user_id, expires_at))
    return token


def user_from_session(token: str | None) -> dict[str, str] | None:
    if not token:
        return None
    with connect() as connection:
        row = connection.execute("SELECT users.id, users.name, users.email, sessions.expires_at FROM sessions JOIN users ON users.id = sessions.user_id WHERE sessions.token = ?", (token,)).fetchone()
    if not row or datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


def delete_session(token: str | None) -> None:
    if token:
        with connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token = ?", (token,))


initialize_auth()
