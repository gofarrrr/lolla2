"""
Authentication Service - Protocol-based Design
B2 - Infrastructure Service Extraction (V1 Implementation)
"""
import logging
import hashlib
import hmac
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .contracts import (
    IRequestAuthenticator, AuthConfig,
    AuthenticationError
)

logger = logging.getLogger(__name__)

class V1AuthenticationService:
    """
    V1 Authentication Service Implementation
    Protocol-based design with bearer token and HMAC validation
    """
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.logger = logger
        
        # Simple in-memory session store (would use Redis in production)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # User permissions cache
        self.user_permissions: Dict[str, List[str]] = {
            "admin": ["read", "write", "admin"],
            "user": ["read", "write"],
            "guest": ["read"]
        }
    
    async def authenticate_bearer_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate bearer token
        Simple implementation for V1 - would integrate with proper auth provider
        """
        try:
            # Simple token validation (placeholder for JWT/OAuth)
            if not token or len(token) < 10:
                return None
            
            # Mock user extraction from token
            if token.startswith("admin_"):
                return {
                    "user_id": "admin_user",
                    "username": "admin",
                    "role": "admin",
                    "authenticated_at": datetime.now().isoformat()
                }
            elif token.startswith("user_"):
                return {
                    "user_id": "regular_user", 
                    "username": "user",
                    "role": "user",
                    "authenticated_at": datetime.now().isoformat()
                }
            elif token.startswith("guest_"):
                return {
                    "user_id": "guest_user",
                    "username": "guest", 
                    "role": "guest",
                    "authenticated_at": datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Bearer token authentication failed: {e}")
            raise AuthenticationError(f"Token validation failed: {e}")
    
    async def validate_hmac_signature(
        self, 
        payload: bytes, 
        signature: str, 
        secret: str
    ) -> bool:
        """
        Validate HMAC signature for API authentication
        """
        try:
            if not secret:
                secret = self.config.hmac_secret or "default_secret"
            
            # Calculate expected signature
            expected = hmac.new(
                secret.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # Secure comparison
            return hmac.compare_digest(signature, expected)
            
        except Exception as e:
            self.logger.error(f"HMAC validation failed: {e}")
            return False
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """
        Get user permissions from cache/database
        """
        try:
            # Extract role from sessions or user data
            user_role = "guest"  # Default
            
            for session_data in self.sessions.values():
                if session_data.get("user_id") == user_id:
                    user_role = session_data.get("role", "guest")
                    break
            
            return self.user_permissions.get(user_role, ["read"])
            
        except Exception as e:
            self.logger.error(f"Permission lookup failed for {user_id}: {e}")
            return ["read"]  # Safe default
    
    async def create_session(self, user_data: Dict[str, Any]) -> str:
        """
        Create authenticated session
        """
        try:
            # Generate session token
            session_id = hashlib.sha256(
                f"{user_data.get('user_id', 'unknown')}_{time.time()}".encode()
            ).hexdigest()[:32]
            
            # Store session data
            self.sessions[session_id] = {
                **user_data,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(seconds=self.config.session_timeout)).isoformat(),
                "session_id": session_id
            }
            
            # Cleanup expired sessions
            await self._cleanup_expired_sessions()
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
            raise AuthenticationError(f"Session creation failed: {e}")
    
    async def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Validate existing session
        """
        try:
            if session_id not in self.sessions:
                return None
            
            session_data = self.sessions[session_id]
            
            # Check expiration
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.now() > expires_at:
                del self.sessions[session_id]
                return None
            
            return session_data
            
        except Exception as e:
            self.logger.error(f"Session validation failed: {e}")
            return None
    
    async def invalidate_session(self, session_id: str) -> None:
        """
        Invalidate session (logout)
        """
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
                self.logger.info(f"Session {session_id} invalidated")
        except Exception as e:
            self.logger.error(f"Session invalidation failed: {e}")
    
    async def _cleanup_expired_sessions(self) -> None:
        """
        Clean up expired sessions
        """
        try:
            now = datetime.now()
            expired_sessions = []
            
            for session_id, session_data in self.sessions.items():
                expires_at = datetime.fromisoformat(session_data["expires_at"])
                if now > expires_at:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            self.logger.error(f"Session cleanup failed: {e}")
    
    def get_auth_stats(self) -> Dict[str, Any]:
        """
        Get authentication statistics
        """
        return {
            "active_sessions": len(self.sessions),
            "config": {
                "bearer_token_enabled": self.config.bearer_token_enabled,
                "session_timeout": self.config.session_timeout,
                "hmac_configured": bool(self.config.hmac_secret)
            },
            "permissions_configured": len(self.user_permissions)
        }

class AuthenticationServiceFactory:
    """
    Factory for creating authentication services
    """
    
    @staticmethod
    def create_v1_auth_service(config: AuthConfig = None) -> V1AuthenticationService:
        """Create V1 authentication service"""
        if not config:
            config = AuthConfig()
        
        return V1AuthenticationService(config)
    
    @staticmethod
    def create_from_env() -> V1AuthenticationService:
        """Create authentication service from environment"""
        import os
        
        config = AuthConfig(
            jwt_secret=os.getenv("JWT_SECRET"),
            hmac_secret=os.getenv("HMAC_SECRET"),
            bearer_token_enabled=os.getenv("BEARER_TOKEN_ENABLED", "true").lower() == "true",
            session_timeout=int(os.getenv("SESSION_TIMEOUT", "3600"))
        )
        
        return V1AuthenticationService(config)