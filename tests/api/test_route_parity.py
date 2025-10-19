from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Iterable, Tuple

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.parity


def _route_signature(routes: Iterable) -> set[Tuple[str, Tuple[str, ...]]]:
    """Return (path, methods) tuples for comparison."""
    signature: set[Tuple[str, Tuple[str, ...]]] = set()
    for route in routes:
        if getattr(route, "methods", None):
            signature.add((route.path, tuple(sorted(route.methods))))
    return signature


def test_stateful_analysis_route_signatures_match_legacy():
    try:
        from src.api.routes.stateful_analysis_routes import (
            router as lean_router,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"stateful analysis routes unavailable: {exc}")

    from src.engine.api.analysis_execution_api_v53 import router as legacy_router

    assert _route_signature(lean_router.routes) == _route_signature(legacy_router.routes)


def test_stateful_analysis_status_endpoint():
    try:
        from src.api.routes.stateful_analysis_routes import (
            router as lean_router,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"stateful analysis routes unavailable: {exc}")

    app = FastAPI()
    app.include_router(lean_router)
    client = TestClient(app)

    response = client.get("/api/v53/analysis/v53-status")
    assert response.status_code == 200
    data = response.json()
    assert data.get("v53_compliance") is True


def test_progressive_questions_route_signatures_match_legacy():
    try:
        from src.api.routes.progressive_questions import router as lean_router
    except ModuleNotFoundError as exc:
        pytest.skip(f"progressive questions routes unavailable: {exc}")

    from src.engine.api.progressive_questions import router as legacy_router

    assert _route_signature(lean_router.routes) == _route_signature(legacy_router.routes)


def test_foundation_routes_present():
    from src.api.routes.foundation_routes import foundation_app

    client = TestClient(foundation_app.app)
    # Health endpoint should exist; when Supabase is unavailable we expect 503
    response = client.get("/api/health/enhanced")
    assert response.status_code in (200, 503)
