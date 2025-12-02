"""JWT token management and API key verification."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from src.rag.core.exceptions import AuthenticationError, TokenExpiredError, InvalidTokenError
from src.rag.core.logging import get_logger

logger = get_logger(__name__)


def create_access_token(
    payload: dict[str, Any],
    secret_key: str,
    algorithm: str = "HS256",
    expires_minutes: int = 60,
) -> str:
    try:
        from jose import jwt
    except ImportError:
        raise RuntimeError("python-jose not installed. Run: pip install python-jose[cryptography]")

    now = datetime.now(timezone.utc)
    data = {
        **payload,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
        "jti": secrets.token_hex(16),
    }
    return jwt.encode(data, secret_key, algorithm=algorithm)


def create_refresh_token(
    user_id: str,
    secret_key: str,
    algorithm: str = "HS256",
    expires_days: int = 30,
) -> str:
    return create_access_token(
        {"sub": user_id, "type": "refresh"},
        secret_key,
        algorithm,
        expires_minutes=expires_days * 24 * 60,
    )


def decode_access_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256",
) -> dict[str, Any]:
    try:
        from jose import ExpiredSignatureError, JWTError, jwt
    except ImportError:
        raise RuntimeError("python-jose not installed")

    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        if payload.get("type") == "refresh":
            raise InvalidTokenError("Refresh token cannot be used for API access")
        return payload
    except ExpiredSignatureError as exc:
        raise TokenExpiredError("Access token has expired") from exc
    except JWTError as exc:
        raise InvalidTokenError(f"Invalid token: {exc}") from exc


def hash_api_key(raw_key: str) -> str:
    """Store only the SHA-256 hash of API keys, never the plaintext."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key. Returns (raw_key, hashed_key)."""
    raw = f"rag_{secrets.token_urlsafe(32)}"
    return raw, hash_api_key(raw)


async def verify_api_key(raw_key: str, cache) -> dict | None:
    """Look up an API key hash in cache. Returns user dict or None."""
    key_hash = hash_api_key(raw_key)
    cache_key = f"apikey:{key_hash}"
    user_data = await cache.get(cache_key)
    if user_data:
        logger.debug("api_key_verified", key_prefix=raw_key[:8])
        return user_data
    return None


async def store_api_key(raw_key: str, user_data: dict, cache, ttl: int = 86400) -> None:
    """Store an API key hash → user mapping in cache."""
    key_hash = hash_api_key(raw_key)
    await cache.set(f"apikey:{key_hash}", user_data, ttl=ttl)
