import pathlib
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.middleware import DeprecationHeaderMiddleware


def _create_app() -> TestClient:
    app = FastAPI()
    app.add_middleware(DeprecationHeaderMiddleware)

    @app.get("/api/legacy")
    def _legacy():
        return {"status": "legacy"}

    @app.get("/api/v53/modern")
    def _modern():
        return {"status": "modern"}

    return TestClient(app)


def test_legacy_route_receives_deprecation_headers():
    client = _create_app()
    response = client.get("/api/legacy")
    assert response.status_code == 200
    assert response.headers["X-Api-Version"] == "legacy"
    assert "Legacy endpoint" in response.headers["X-Api-Deprecation"]


def test_modern_route_skips_deprecation_headers():
    client = _create_app()
    response = client.get("/api/v53/modern")
    assert response.status_code == 200
    assert "X-Api-Version" not in response.headers
    assert "X-Api-Deprecation" not in response.headers
