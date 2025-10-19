from __future__ import annotations

import logging
from typing import Iterable, Tuple

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class DeprecationHeaderMiddleware(BaseHTTPMiddleware):
    """Annotate legacy engine API responses with deprecation headers.

    Adds `X-Api-Version: legacy` and a short deprecation message unless the path
    matches one of the allowed prefixes (e.g., `/api/v53` Lean routes).
    """

    def __init__(
        self,
        app,
        *,
        legacy_prefixes: Iterable[str] = ("/api/",),
        modern_prefixes: Iterable[str] = ("/api/v53/",),
        deprecation_message: str = "Legacy endpoint scheduled for removal. Migrate to /api/v53 equivalents.",
    ) -> None:
        super().__init__(app)
        self.legacy_prefixes: Tuple[str, ...] = tuple(legacy_prefixes)
        self.modern_prefixes: Tuple[str, ...] = tuple(modern_prefixes)
        self.deprecation_message = deprecation_message

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        path = request.url.path
        if self._is_legacy_path(path):
            response.headers.setdefault("X-Api-Version", "legacy")
            response.headers.setdefault("X-Api-Deprecation", self.deprecation_message)
            logger.info(
                "Legacy API response tagged",
                extra={"path": path, "component": "deprecation_middleware"},
            )

        return response

    def _is_legacy_path(self, path: str) -> bool:
        return path.startswith(self.legacy_prefixes) and not path.startswith(
            self.modern_prefixes
        )
