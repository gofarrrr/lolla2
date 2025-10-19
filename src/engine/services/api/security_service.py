"""
API Security Service - Authentication and Authorization
=====================================================

REFACTORING TARGET: Extract SecurityManager from foundation.py
PATTERN: Service Extraction with Security Strategy
GOAL: Create focused, testable security service

Responsibility:
- API authentication and authorization
- Permission validation and enforcement
- Security context management
- Rate limiting coordination

Benefits:
- Single Responsibility Principle for security
- Easily testable security logic
- Clear security interfaces
- Pluggable authentication strategies
"""

import logging
from typing import Dict, List, Any, Callable
from datetime import datetime, timedelta
from functools import wraps

try:
    from fastapi import HTTPException, Depends, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    HTTPBearer = None

# Import auth foundation
from src.core.auth_foundation import (
    get_auth_manager,
    Permission,
    UserRole,
)

logger = logging.getLogger(__name__)


class APISecurityService:
    """
    API security service for authentication and authorization

    Responsibility: Centralized API security management
    Complexity Target: Grade B (≤10 per method)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.auth_manager = get_auth_manager()

        # Security configuration
        self.security_config = {
            "require_auth": True,
            "default_permissions": [Permission.READ_BASIC],
            "admin_permissions": [Permission.ADMIN_FULL],
            "token_expiry_hours": 24,
            "failed_login_threshold": 5,
        }

        # Rate limiting tracking
        self.failed_attempts = {}
        self.active_sessions = {}

    def get_security_scheme(self):
        """
        Get FastAPI security scheme

        Complexity: Grade A (1)
        """
        if not FASTAPI_AVAILABLE:
            return None

        return HTTPBearer(
            scheme_name="Bearer Token", description="JWT token for API authentication"
        )

    async def authenticate_api_request(
        self, credentials: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """
        Authenticate API request with JWT token

        Complexity: Target B (≤10)
        """
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication credentials required",
            )

        try:
            # Validate token format
            token = credentials.credentials
            if not token or len(token) < 10:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token format",
                )

            # Authenticate with auth manager
            auth_result = await self.auth_manager.authenticate_token(token)

            if not auth_result.success:
                self._record_failed_attempt(
                    token[:10]
                )  # Log partial token for debugging
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Authentication failed: {auth_result.error_message}",
                )

            # Create security context
            security_context = {
                "user_id": auth_result.user_id,
                "user_role": auth_result.user_role,
                "permissions": auth_result.permissions,
                "session_id": auth_result.session_id,
                "authenticated_at": datetime.now().isoformat(),
            }

            self._record_successful_auth(auth_result.user_id)
            return security_context

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service unavailable",
            )

    def validate_permissions(
        self, security_context: Dict[str, Any], required_permissions: List[Permission]
    ) -> bool:
        """
        Validate user permissions for API access

        Complexity: Target B (≤10)
        """
        if not security_context:
            return False

        user_permissions = security_context.get("permissions", [])
        user_role = security_context.get("user_role", UserRole.GUEST)

        # Admin bypass
        if user_role == UserRole.ADMIN or Permission.ADMIN_FULL in user_permissions:
            return True

        # Check required permissions
        for required_permission in required_permissions:
            if required_permission not in user_permissions:
                self.logger.warning(
                    f"Permission denied: User {security_context.get('user_id')} "
                    f"missing {required_permission}"
                )
                return False

        return True

    def require_permission_dependency(self, permission: Permission):
        """
        Create FastAPI dependency for permission checking

        Complexity: Target B (≤10)
        """

        async def permission_checker(
            security_context: Dict[str, Any] = Depends(self.authenticate_api_request),
        ):
            if not self.validate_permissions(security_context, [permission]):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {permission}",
                )
            return security_context

        return permission_checker

    def _record_failed_attempt(self, token_prefix: str):
        """
        Record failed authentication attempt

        Complexity: Target B (≤10)
        """
        now = datetime.now()

        if token_prefix not in self.failed_attempts:
            self.failed_attempts[token_prefix] = []

        # Add failed attempt
        self.failed_attempts[token_prefix].append(now)

        # Clean old attempts (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        self.failed_attempts[token_prefix] = [
            attempt
            for attempt in self.failed_attempts[token_prefix]
            if attempt > cutoff
        ]

        # Check for too many failures
        recent_failures = len(self.failed_attempts[token_prefix])
        if recent_failures >= self.security_config["failed_login_threshold"]:
            self.logger.warning(
                f"High number of failed attempts for token {token_prefix}: {recent_failures}"
            )

    def _record_successful_auth(self, user_id: str):
        """
        Record successful authentication

        Complexity: Target B (≤10)
        """
        self.active_sessions[user_id] = {
            "authenticated_at": datetime.now(),
            "last_activity": datetime.now(),
        }

        # Clean old sessions (older than 24 hours)
        cutoff = datetime.now() - timedelta(
            hours=self.security_config["token_expiry_hours"]
        )
        self.active_sessions = {
            uid: session
            for uid, session in self.active_sessions.items()
            if session["last_activity"] > cutoff
        }

    def get_security_stats(self) -> Dict[str, Any]:
        """
        Get security service statistics

        Complexity: Target B (≤10)
        """
        now = datetime.now()
        recent_cutoff = now - timedelta(minutes=30)

        # Count recent failed attempts
        recent_failures = 0
        for attempts in self.failed_attempts.values():
            recent_failures += sum(1 for attempt in attempts if attempt > recent_cutoff)

        # Count active sessions
        active_sessions = len(
            [
                session
                for session in self.active_sessions.values()
                if session["last_activity"] > recent_cutoff
            ]
        )

        return {
            "active_sessions": active_sessions,
            "recent_failed_attempts": recent_failures,
            "total_tracked_tokens": len(self.failed_attempts),
            "security_config": self.security_config.copy(),
        }


class APIPermissionDecorators:
    """
    API permission decorators for endpoint security

    Responsibility: Decorators for endpoint-level security
    Complexity Target: Grade B (≤10 per method)
    """

    def __init__(self, security_service: APISecurityService):
        self.security_service = security_service

    def require_permission(self, permission: Permission):
        """
        Decorator for requiring specific permissions

        Complexity: Target B (≤10)
        """

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract security context from kwargs or args
                security_context = kwargs.get("security_context")

                if not security_context:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required",
                    )

                if not self.security_service.validate_permissions(
                    security_context, [permission]
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission required: {permission}",
                    )

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def admin_only(self, func: Callable):
        """
        Decorator for admin-only endpoints

        Complexity: Target B (≤10)
        """
        return self.require_permission(Permission.ADMIN_FULL)(func)


# Singleton instances for injection
_security_service_instance = None
_permission_decorators_instance = None


def get_api_security_service() -> APISecurityService:
    """Factory function for API security service"""
    global _security_service_instance
    if _security_service_instance is None:
        _security_service_instance = APISecurityService()
    return _security_service_instance


def get_api_permission_decorators() -> APIPermissionDecorators:
    """Factory function for API permission decorators"""
    global _permission_decorators_instance
    if _permission_decorators_instance is None:
        security_service = get_api_security_service()
        _permission_decorators_instance = APIPermissionDecorators(security_service)
    return _permission_decorators_instance
