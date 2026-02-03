"""Role-based access control."""

from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Callable

from fastapi import HTTPException, status

from src.rag.core.exceptions import AuthorizationError


class Role(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    READER = "reader"


class Permission(str, Enum):
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    SEARCH = "search"
    COLLECTION_MANAGE = "collection:manage"
    USER_MANAGE = "user:manage"
    METRICS_READ = "metrics:read"
    EVAL_RUN = "eval:run"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.ADMIN: frozenset(Permission),
    Role.OPERATOR: frozenset({
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.DOCUMENT_DELETE,
        Permission.SEARCH,
        Permission.COLLECTION_MANAGE,
        Permission.METRICS_READ,
        Permission.EVAL_RUN,
    }),
    Role.READER: frozenset({
        Permission.DOCUMENT_READ,
        Permission.SEARCH,
    }),
}


def has_permission(user: dict, permission: Permission) -> bool:
    role_str = user.get("role", Role.READER.value)
    try:
        role = Role(role_str)
    except ValueError:
        return False
    return permission in ROLE_PERMISSIONS.get(role, frozenset())


def require_permission(permission: Permission) -> Callable:
    """FastAPI dependency that enforces a permission check on the current user."""
    def dependency(current_user: dict) -> dict:
        if not has_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission.value}' required",
            )
        return current_user
    return dependency
