"""FastAPI middleware for schema validation and version tracking."""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import json
from ..schema.validation import schema_registry

class SchemaValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for schema validation and version tracking."""
    
    async def dispatch(self, request: Request, call_next):
        # Check if endpoint requires schema validation
        endpoint = request.url.path
        if self._should_validate(endpoint):
            try:
                # Get request body
                body = await request.json()
                
                # Determine schema name from endpoint
                schema_name = self._get_schema_name(endpoint)
                
                # Validate against schema
                schema_registry.validate(schema_name, body)
                
                # Add schema version to response headers
                response = await call_next(request)
                schema_version = schema_registry.get_schema_version(schema_name)
                if schema_version:
                    response.headers["X-Schema-Version"] = schema_version.version
                    response.headers["X-Schema-Hash"] = schema_version.hash
                return response
                
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid JSON in request body"}
                )
            except HTTPException as e:
                return JSONResponse(
                    status_code=e.status_code,
                    content={"detail": str(e.detail)}
                )
                
        return await call_next(request)
        
    def _should_validate(self, endpoint: str) -> bool:
        """Determine if endpoint requires schema validation."""
        validation_paths = [
            "/api/v53/analyze",
            "/api/v53/services",
        ]
        return any(endpoint.startswith(path) for path in validation_paths)
        
    def _get_schema_name(self, endpoint: str) -> str:
        """Map endpoint to schema name."""
        schema_mapping = {
            "/api/v53/analyze": "analysis_request",
            "/api/v53/services": "services_status",
        }
        return schema_mapping.get(endpoint, "default")
