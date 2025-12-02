"""JWT and API key authentication."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from src.rag.core.exceptions import InvalidTokenError, TokenExpiredError
from src.rag.core.logging import get_logger

logger = get_logger(__name__)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_secret(secret: str) -> str:
    """One-way hash for API keys (stored in DB, never the raw key)."""
    return hashlib.sha256(secret.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Return (raw_key, hashed_key). Store only the hash."""
    raw = f"rag_{secrets.token_urlsafe(32)}"
    return raw, hash_secret(raw)


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def create_access_token(
    subject: str,
    *,
    secret_key: str,
    algorithm: str = "HS256",
    expires_minutes: int = 60,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    claims: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
        "type": "access",
        **(extra_claims or {}),
    }
    return jwt.encode(claims, secret_key, algorithm=algorithm)


def create_refresh_token(
    subject: str,
    *,
    secret_key: str,
    algorithm: str = "HS256",
    expires_days: int = 30,
) -> str:
    now = datetime.now(timezone.utc)
    claims: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(days=expires_days),
        "type": "refresh",
    }
    return jwt.encode(claims, secret_key, algorithm=algorithm)


async def verify_token(token: str, secret_key: str, algorithm: str = "HS256") -> dict[str, Any]:
    """Decode and validate a JWT. Raises on expiry or bad signature."""
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except JWTError as exc:
        msg = str(exc).lower()
        if "expired" in msg:
            raise TokenExpiredError("Token has expired", cause=exc) from exc
        raise InvalidTokenError("Invalid token", cause=exc) from exc


async def verify_api_key(raw_key: str) -> dict[str, Any]:
    """
    Validate a raw API key.  In production, look up hash_secret(raw_key)
    in the database and return the associated user record.
    """
    if not raw_key.startswith("rag_"):
        from src.rag.core.exceptions import AuthenticationError
        raise AuthenticationError("Invalid API key format")
    # Placeholder: real implementation queries the DB
    return {"sub": "api-key-user", "type": "api_key", "roles": ["reader"]}
