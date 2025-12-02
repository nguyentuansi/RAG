"""Role-Based Access Control."""

from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Callable

from src.rag.core.exceptions import AuthorizationError


class Role(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    READER = "reader"


class Permission(str, Enum):
    # Document permissions
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    # Search permissions
    SEARCH = "search:execute"
    # Collection management
    COLLECTION_MANAGE = "collection:manage"
    # Admin
    USER_MANAGE = "user:manage"
    METRICS_READ = "metrics:read"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.READER: frozenset({
        Permission.DOCUMENT_READ,
        Permission.SEARCH,
    }),
    Role.OPERATOR: frozenset({
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.DOCUMENT_DELETE,
        Permission.SEARCH,
        Permission.COLLECTION_MANAGE,
        Permission.METRICS_READ,
    }),
    Role.ADMIN: frozenset(Permission),
}


def get_permissions(role: Role | str) -> frozenset[Permission]:
    """Return the permission set for a role."""
    try:
        r = Role(role)
    except ValueError:
        return frozenset()
    return ROLE_PERMISSIONS.get(r, frozenset())


def has_permission(user: dict, permission: Permission) -> bool:
    """Check whether a user dict (from JWT claims) holds a given permission."""
    roles: list[str] = user.get("roles", [])
    for role_str in roles:
        if permission in get_permissions(role_str):
            return True
    return False


def require_permission(permission: Permission) -> Callable:
    """
    FastAPI-compatible dependency factory.

    Usage::

        @router.get("/admin-only")
        async def handler(
            user: CurrentUserDep,
            _: None = Depends(require_permission(Permission.USER_MANAGE)),
        ) -> ...:
    """
    async def _check(current_user: dict) -> None:
        if not has_permission(current_user, permission):
            raise AuthorizationError(
                f"Permission '{permission}' required",
                details={"required": permission, "user_roles": current_user.get("roles", [])},
            )

    return _check
