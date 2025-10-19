"""
Supabase JWT Authentication Middleware
Validates Supabase JWT tokens from the frontend instead of custom auth system
"""

import jwt
import logging
import os
from typing import Dict, Optional, Any
from functools import wraps
from datetime import datetime

try:
    from fastapi import HTTPException, Depends, Request, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    HTTPBearer = None


logger = logging.getLogger(__name__)

# Initialize security scheme
security = HTTPBearer() if FASTAPI_AVAILABLE else None


class SupabaseUser:
    """Simplified user representation from Supabase JWT"""

    def __init__(self, jwt_payload: Dict[str, Any]):
        self.user_id = jwt_payload.get("sub")
        self.email = jwt_payload.get("email")
        self.role = jwt_payload.get("role", "authenticated")
        self.aud = jwt_payload.get("aud")
        self.exp = jwt_payload.get("exp")
        self.iat = jwt_payload.get("iat")
        self.iss = jwt_payload.get("iss")
        self.metadata = jwt_payload.get("user_metadata", {})
        self.app_metadata = jwt_payload.get("app_metadata", {})
        self.raw_payload = jwt_payload


class SupabaseAuthMiddleware:
    """Middleware to validate Supabase JWT tokens"""

    def __init__(self):
        # Get JWT secret from Supabase service role key
        # Supabase access tokens are verified with the service role secret (not the anon key)
        self.jwt_secret = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not self.jwt_secret:
            logger.error(
                "SUPABASE_SERVICE_ROLE_KEY not found. Authentication will fail."
            )

        self.supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        if not self.supabase_url:
            logger.warning("NEXT_PUBLIC_SUPABASE_URL not found.")

        # Support demo mode for development
        self.demo_mode = (
            os.getenv("NODE_ENV") == "development" or os.getenv("DEMO_MODE") == "true"
        )

    def verify_jwt_token(self, token: str) -> Optional[SupabaseUser]:
        """Verify and decode Supabase JWT token"""
        try:
            # Handle demo mode tokens
            if self.demo_mode and token.startswith("demo-"):
                logger.info("🎭 Demo mode: Accepting demo token")
                # Return a mock user for demo mode
                return SupabaseUser(
                    {
                        "sub": "demo-user-123",
                        "email": "demo@example.com",
                        "role": "authenticated",
                        "aud": "authenticated",
                        "exp": int(datetime.utcnow().timestamp())
                        + 3600,  # 1 hour from now
                        "iat": int(datetime.utcnow().timestamp()),
                        "iss": self.supabase_url or "demo",
                    }
                )

            # For real Supabase tokens, verify with service role secret
            if not self.jwt_secret:
                logger.error("No JWT secret available for token verification")
                return None

            # Supabase uses HS256 algorithm
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_aud": False},  # Supabase tokens may have different aud
            )

            # Validate token is not expired
            if datetime.fromtimestamp(payload.get("exp", 0)) < datetime.utcnow():
                logger.warning("Token expired")
                return None

            # Validate issuer matches Supabase URL (if configured)
            if self.supabase_url and not payload.get("iss", "").startswith(
                self.supabase_url
            ):
                logger.warning(
                    f"Invalid issuer: {payload.get('iss')} (expected: {self.supabase_url})"
                )
                return None

            logger.info(
                f"✅ Valid Supabase JWT token for user: {payload.get('email', 'unknown')}"
            )
            return SupabaseUser(payload)

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None


# Global middleware instance
_auth_middleware: Optional[SupabaseAuthMiddleware] = None


def get_auth_middleware() -> SupabaseAuthMiddleware:
    """Get or create auth middleware instance"""
    global _auth_middleware
    if _auth_middleware is None:
        _auth_middleware = SupabaseAuthMiddleware()
    return _auth_middleware


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> SupabaseUser:
    """FastAPI dependency to get current authenticated user"""
    if not FASTAPI_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FastAPI not available",
        )

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    middleware = get_auth_middleware()
    user = middleware.verify_jwt_token(credentials.credentials)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_optional_user(request: Request) -> Optional[SupabaseUser]:
    """Get user if authenticated, None if not (for optional auth endpoints)"""
    try:
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header.split(" ")[1]
        middleware = get_auth_middleware()
        return middleware.verify_jwt_token(token)

    except Exception as e:
        logger.debug(f"Optional auth failed: {e}")
        return None


def require_auth(func):
    """Decorator for non-FastAPI functions requiring authentication"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # This would need to be adapted based on how you're calling it
        # For now, it's a placeholder for manual auth checking
        pass

    return wrapper


# Utility functions for permission checking
def has_admin_role(user: SupabaseUser) -> bool:
    """Check if user has admin role"""
    return user.app_metadata.get("role") == "admin" or user.role == "admin"


def is_service_role(user: SupabaseUser) -> bool:
    """Check if token is from service role"""
    return user.role == "service_role"


def can_access_resource(user: SupabaseUser, resource_user_id: str) -> bool:
    """Check if user can access resource owned by another user"""
    # Users can access their own resources
    if user.user_id == resource_user_id:
        return True

    # Admins can access any resource
    if has_admin_role(user):
        return True

    # Service role can access any resource
    if is_service_role(user):
        return True

    return False


def verify_token() -> str:
    """Compatibility function for Senior Advisor API - temporary bypass for testing"""
    return "admin_user"
