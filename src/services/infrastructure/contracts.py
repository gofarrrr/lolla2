"""
Infrastructure Service Contracts
B1 - Responsibility Analysis (Red Team Amendment Applied)
"""
from typing import Protocol, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Rate limiting contracts
class IRateLimiter(Protocol):
    """Rate limiter interface with Redis backend support"""
    
    async def check_rate_limit(
        self, 
        identifier: str, 
        limit: int, 
        window: int, 
        **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if rate limit allows request"""
        ...
    
    async def penalize(self, identifier: str, penalty_seconds: int) -> None:
        """Apply penalty to identifier"""
        ...
    
    async def get_quota(self, identifier: str) -> Dict[str, Any]:
        """Get current quota status"""
        ...
    
    async def reset_quota(self, identifier: str) -> None:
        """Reset quota for identifier"""
        ...
    
    async def health_check(self) -> bool:
        """Check rate limiter backend health"""
        ...

class IRequestAuthenticator(Protocol):
    """Request authentication interface"""
    
    async def authenticate_bearer_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate bearer token"""
        ...
    
    async def validate_hmac_signature(
        self, 
        payload: bytes, 
        signature: str, 
        secret: str
    ) -> bool:
        """Validate HMAC signature"""
        ...
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions"""
        ...
    
    async def create_session(self, user_data: Dict[str, Any]) -> str:
        """Create authenticated session"""
        ...

# Rate limiting configuration (Red Team Amendment)
@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    backend: str = "memory"  # redis|memory
    buckets: Dict[str, str] = None  # endpoint:rate pairs
    redis_url: Optional[str] = None
    shadow_mode: bool = False  # observe-only mode
    circuit_breaker_enabled: bool = True

@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    remaining: int
    reset_time: int
    identifier: str
    backend_used: str = "memory"

# Authentication configuration
@dataclass
class AuthConfig:
    """Authentication configuration"""
    jwt_secret: Optional[str] = None
    hmac_secret: Optional[str] = None
    bearer_token_enabled: bool = True
    session_timeout: int = 3600

# Infrastructure errors
class InfrastructureError(Exception):
    """Base infrastructure error"""
    pass

class RateLimitError(InfrastructureError):
    """Rate limiting error"""
    pass

class AuthenticationError(InfrastructureError):
    """Authentication error"""
    pass

class BackendConnectionError(InfrastructureError):
    """Backend connection error"""
    pass