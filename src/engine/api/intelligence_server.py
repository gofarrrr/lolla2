#!/usr/bin/env python3
"""
Intelligence Mode API Server
Standalone FastAPI server for MetisODR cognitive intelligence endpoints
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("❌ FastAPI not available - install with: pip install fastapi uvicorn")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_intelligence_app():
    """Create FastAPI app for Intelligence Mode"""

    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI and uvicorn are required")

    # Create FastAPI app
    app = FastAPI(
        title="METIS Intelligence Mode API",
        version="1.0.0",
        description="MetisODR Cognitive Intelligence System API",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3001",
            "http://localhost:3000",
        ],  # Frontend URLs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include intelligence router
    try:
        from src.engine.api.intelligence_api import router as intelligence_router

        app.include_router(intelligence_router)
        logger.info("✅ Intelligence API router included")
    except Exception as e:
        logger.error(f"❌ Failed to include intelligence router: {e}")

    # Import and include progressive questions router
    try:
        from src.engine.api.progressive_questions import router as questions_router

        app.include_router(questions_router)
        logger.info("✅ Progressive Questions API router included")
    except Exception as e:
        logger.error(f"❌ Failed to include progressive questions router: {e}")
        # Create fallback router
        from fastapi import APIRouter

        fallback_router = APIRouter(
            prefix="/api/v1/intelligence", tags=["intelligence"]
        )

        # Removed non-canonical /health endpoint to avoid duplication with /api/v53/health

        app.include_router(fallback_router)
        logger.info("⚠️ Using fallback router")

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "METIS Intelligence Mode API",
            "status": "running",
            "version": "1.0.0",
            "docs": "/docs",
        }

    # Removed non-canonical /health endpoint; canonical health lives in main app under /api/v53/health

    return app


def run_server(host: str = "0.0.0.0", port: int = 8001, reload: bool = True):
    """Run the Intelligence Mode API server"""

    if not FASTAPI_AVAILABLE:
        logger.error("❌ Cannot run server - FastAPI not available")
        return False

    logger.info("🚀 Starting METIS Intelligence Mode API Server")
    logger.info(f"   - Host: {host}")
    logger.info(f"   - Port: {port}")
    logger.info(f"   - Reload: {reload}")
    logger.info(f"   - Docs: http://{host}:{port}/docs")

    try:
        app = create_intelligence_app()
        uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")
        return True

    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="METIS Intelligence Mode API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")

    args = parser.parse_args()

    success = run_server(host=args.host, port=args.port, reload=not args.no_reload)

    if not success:
        sys.exit(1)
