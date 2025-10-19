"""
API Security Components - Extracted from foundation.py
Authentication, rate limiting, and permission management
"""

import time
import logging
from typing import Dict, Optional, Any, Callable
from datetime import datetime
from functools import wraps

try:
    from fastapi import HTTPException, Depends, Request, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    HTTPBearer = None

from src.core.auth_foundation import (
    get_auth_manager,
    Permission,
    authenticate_request,
)


class RateLimiter:
    """
    Rate limiting for API endpoints with configurable windows and limits
    """

    def __init__(self):
        self.requests = {}  # {client_id: [(timestamp, endpoint), ...]}
        self.logger = logging.getLogger(__name__)

    def _cleanup_old_requests(self, client_id: str, window_minutes: int = 60):
        """Remove requests outside the time window"""
        if client_id not in self.requests:
            return

        cutoff_time = time.time() - (window_minutes * 60)
        self.requests[client_id] = [
            (timestamp, endpoint)
            for timestamp, endpoint in self.requests[client_id]
            if timestamp > cutoff_time
        ]

        if not self.requests[client_id]:
            del self.requests[client_id]

    async def check_rate_limit(
        self,
        client_id: str,
        endpoint: str,
        requests_per_hour: int = 100,
        window_minutes: int = 60,
    ) -> bool:
        """
        Check if client has exceeded rate limit for endpoint
        Returns True if request is allowed, raises HTTPException if blocked
        """
        try:
            current_time = time.time()

            # Clean up old requests
            self._cleanup_old_requests(client_id, window_minutes)

            # Count recent requests
            if client_id in self.requests:
                recent_requests = len(self.requests[client_id])

                if recent_requests >= requests_per_hour:
                    self.logger.warning(
                        f"🚫 Rate limit exceeded for {client_id} on {endpoint}: {recent_requests}/{requests_per_hour}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded. Maximum {requests_per_hour} requests per hour allowed.",
                    )

            # Record this request
            if client_id not in self.requests:
                self.requests[client_id] = []

            self.requests[client_id].append((current_time, endpoint))

            return True

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ Rate limiting error: {e}")
            return True  # Allow request on error to avoid blocking

    def get_rate_limit_status(
        self, client_id: str, window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get current rate limit status for client"""
        self._cleanup_old_requests(client_id, window_minutes)

        request_count = len(self.requests.get(client_id, []))

        return {
            "client_id": client_id,
            "current_requests": request_count,
            "window_minutes": window_minutes,
            "oldest_request": min(
                [ts for ts, _ in self.requests.get(client_id, [])], default=None
            ),
            "newest_request": max(
                [ts for ts, _ in self.requests.get(client_id, [])], default=None
            ),
        }


class SecurityManager:
    """
    Security management for authentication and session handling
    """

    def __init__(self):
        self.security = HTTPBearer() if FASTAPI_AVAILABLE else None
        self.rate_limiter = RateLimiter()
        self.active_sessions = {}  # session_id -> session_data
        self.logger = logging.getLogger(__name__)

    async def get_current_session(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = None,
    ) -> Dict[str, Any]:
        """
        Extract and validate current session from request
        """
        try:
            # Get client identifier
            client_id = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")

            # Basic session info
            session_info = {
                "client_id": client_id,
                "user_agent": user_agent,
                "endpoint": request.url.path,
                "method": request.method,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # If credentials provided, authenticate
            if credentials:
                try:
                    auth_manager = get_auth_manager()
                    user_info = await authenticate_request(credentials.credentials)
                    session_info.update(
                        {
                            "authenticated": True,
                            "user_id": user_info.get("user_id"),
                            "user_role": user_info.get("role", "user"),
                            "permissions": user_info.get("permissions", []),
                        }
                    )
                except Exception as auth_error:
                    self.logger.warning(f"⚠️ Authentication failed: {auth_error}")
                    session_info["authenticated"] = False
            else:
                session_info["authenticated"] = False

            return session_info

        except Exception as e:
            self.logger.error(f"❌ Session extraction failed: {e}")
            return {"client_id": "unknown", "authenticated": False, "error": str(e)}

    async def validate_permissions(
        self, session: Dict[str, Any], required_permission: Permission
    ) -> bool:
        """
        Validate that session has required permission
        """
        try:
            if not session.get("authenticated", False):
                return False

            user_permissions = session.get("permissions", [])
            return required_permission.value in user_permissions

        except Exception as e:
            self.logger.error(f"❌ Permission validation failed: {e}")
            return False


# Global instances
_security_manager = None
_rate_limiter = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


# Dependency injection functions for FastAPI


async def require_auth(
    request: Request, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> Dict[str, Any]:
    """
    FastAPI dependency to require authentication
    """
    security_manager = get_security_manager()
    session = await security_manager.get_current_session(request, credentials)

    if not session.get("authenticated", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return session


async def get_current_user(request: Request) -> Dict[str, Any]:
    """
    FastAPI dependency to get current user (optional authentication)
    """
    security_manager = get_security_manager()

    # Try to get credentials but don't require them
    auth_header = request.headers.get("authorization")
    credentials = None

    if auth_header and auth_header.startswith("Bearer "):
        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=auth_header[7:]
        )

    session = await security_manager.get_current_session(request, credentials)
    return session


def require_permission(permission: Permission):
    """
    Decorator to require specific permission for endpoint access
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get session from kwargs (should be injected by require_auth dependency)
            session = None
            for arg in kwargs.values():
                if isinstance(arg, dict) and "authenticated" in arg:
                    session = arg
                    break

            if not session:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Session not found - authentication required",
                )

            security_manager = get_security_manager()
            if not await security_manager.validate_permissions(session, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission.value}",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def rate_limit(requests_per_hour: int = 100, window_minutes: int = 60):
    """
    Decorator to apply rate limiting to endpoints
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if hasattr(arg, "client"):  # FastAPI Request object
                    request = arg
                    break

            if not request:
                for arg in kwargs.values():
                    if hasattr(arg, "client"):
                        request = arg
                        break

            if request:
                client_id = request.client.host if request.client else "unknown"
                endpoint = request.url.path

                rate_limiter = get_rate_limiter()
                await rate_limiter.check_rate_limit(
                    client_id=client_id,
                    endpoint=endpoint,
                    requests_per_hour=requests_per_hour,
                    window_minutes=window_minutes,
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator
