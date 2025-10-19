"""
Auth utilities for API route protection (Supabase JWT enforcement).

Behavior:
- Requires Authorization: Bearer <token> header.
- If SUPABASE_JWT_SECRET is set and PyJWT is available, verify token signature.
- Otherwise, accept token presence (useful for local/dev without secret) and mark request as authenticated.

Set REQUIRE_JWT_VERIFICATION=true to force signature verification (401 if not possible).
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import os

security = HTTPBearer(auto_error=False)


def require_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if credentials is None or not credentials.scheme.lower() == "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    # Enforce signature verification when configured
    require_verify = os.getenv("REQUIRE_JWT_VERIFICATION", "false").lower() == "true"
    secret = os.getenv("SUPABASE_JWT_SECRET")

    if secret:
        try:
            import jwt  # PyJWT
            # Supabase default algorithm is HS256
            jwt.decode(token, secret, algorithms=["HS256"])  # Will raise on failure
            return token
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    else:
        if require_verify:
            # Secret missing but verification required
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT verification required")
        # Best-effort mode: token presence accepted
        return token

