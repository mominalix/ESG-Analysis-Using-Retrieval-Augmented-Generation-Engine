"""Authentication dependencies for privileged API routes."""

from __future__ import annotations

from secrets import compare_digest

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..core.config import reveal, settings


bearer_scheme = HTTPBearer(auto_error=False)


def verify_admin_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    """Validate the configured admin bearer token in constant time."""
    expected_token = reveal(settings.admin_token)
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Administrative endpoints are disabled until ADMIN_TOKEN is configured",
        )

    supplied_token = credentials.credentials if credentials else ""
    if not supplied_token or not compare_digest(supplied_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
