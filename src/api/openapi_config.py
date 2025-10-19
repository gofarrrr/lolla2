"""
OpenAPI/Swagger Configuration for METIS V5.3 Platform

This module provides comprehensive OpenAPI documentation configuration
including metadata, tags, security schemes, and examples.
"""

from typing import Dict, Any, List


def get_openapi_metadata() -> Dict[str, Any]:
    """Get enhanced OpenAPI metadata for Swagger documentation."""
    return {
        "title": "METIS V5.3 Phoenix Platform API",
        "description": """
# METIS V5.3 Cognitive Intelligence Platform

## Overview

The METIS V5.3 platform implements a service-oriented architecture for strategic analysis
using AI-powered cognitive consultants. The platform delivers high-quality strategic
recommendations through a 10-stage pipeline with complete transparency.

## Key Features

- **10-Stage Cognitive Pipeline**: Socratic questioning, problem structuring, consultant selection, parallel analysis, and synthesis
- **Glass-Box Transparency**: Complete audit trail of all decisions and LLM interactions via UnifiedContextStream
- **Multi-Provider Resilience**: Automatic failover across DeepSeek, Anthropic, and OpenRouter
- **Iterative Refinement**: Checkpoint-based execution with feedback loops for quality improvement
- **Cost Optimization**: DeepSeek-first strategy providing 585% cost savings vs Claude-only

## Architecture

### Service-Oriented Design

The platform consists of **20 specialized services** across **4 clusters**:

1. **Reliability Services**: Validation, failure detection, feedback orchestration
2. **Selection Services**: Model selection, consultant team optimization, diversity scoring
3. **Application Services**: Depth enrichment, iteration management, artifact extraction
4. **Integration Services**: LLM provider management, research aggregation, persistence

### API Versions

- **V1 API** (`/api/*`): Legacy endpoints (deprecated, maintained for compatibility)
- **V2 API** (`/api/v2/*`): Modern modular endpoints (recommended)

### Authentication

Most endpoints are currently public for development. Production deployment will require:
- Bearer token authentication
- API key management
- Rate limiting

## Getting Started

### 1. Create an Engagement

```bash
curl -X POST https://api.metis.ai/api/v2/engagements \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "How should we enter the AI market?",
    "project_id": "proj_123"
  }'
```

### 2. Monitor Progress

```bash
curl https://api.metis.ai/api/v2/engagements/{id}/trace
```

### 3. Retrieve Final Report

```bash
curl https://api.metis.ai/api/v2/engagements/{id}/report
```

## Resources

- [V5.3 Canonical Standard Specification](https://docs.metis.ai/v5.3)
- [Architecture Documentation](https://docs.metis.ai/architecture)
- [Migration Guide (V1 → V2)](https://docs.metis.ai/migration)
- [GitHub Repository](https://github.com/metis-ai/platform)

## Support

- **Documentation**: https://docs.metis.ai
- **Issues**: https://github.com/metis-ai/platform/issues
- **Email**: support@metis.ai
        """,
        "version": "5.3.0",
        "contact": {
            "name": "METIS Platform Engineering",
            "email": "engineering@metis.ai",
            "url": "https://metis.ai/contact"
        },
        "license_info": {
            "name": "Proprietary",
            "url": "https://metis.ai/license"
        },
        "terms_of_service": "https://metis.ai/terms",
    }


def get_openapi_tags() -> List[Dict[str, str]]:
    """Get OpenAPI tags for organizing endpoints."""
    return [
        {
            "name": "System",
            "description": "System health, status, and configuration endpoints"
        },
        {
            "name": "Engagements (V2)",
            "description": "V2 engagement management endpoints (recommended)"
        },
        {
            "name": "Engagements (V1)",
            "description": "Legacy engagement endpoints (deprecated)"
        },
        {
            "name": "Projects (V2)",
            "description": "V2 project management endpoints (recommended)"
        },
        {
            "name": "Projects (V1)",
            "description": "Legacy project endpoints (deprecated)"
        },
        {
            "name": "Cognitive Pipeline",
            "description": "Direct access to pipeline stages (Socratic, Analysis, Devils Advocate)"
        },
        {
            "name": "Research",
            "description": "Hybrid research endpoints (RAG + web search)"
        },
        {
            "name": "Progressive Questions",
            "description": "Multi-tier business analysis question generation"
        },
        {
            "name": "Quality & Calibration",
            "description": "Decision quality ribbon and calibration services"
        },
        {
            "name": "Specialized Workflows",
            "description": "IdeaFlow, Copywriter, Pitch, and other NWAY workflows"
        },
        {
            "name": "Documents",
            "description": "Document upload and ingestion for RAG"
        },
        {
            "name": "Admin",
            "description": "Administrative endpoints for evidence and diagnostics"
        }
    ]


def get_openapi_security_schemes() -> Dict[str, Any]:
    """Get OpenAPI security schemes."""
    return {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token for authenticated requests (production only)"
        },
        "apiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service-to-service authentication"
        }
    }


def get_openapi_servers() -> List[Dict[str, str]]:
    """Get OpenAPI server configurations."""
    return [
        {
            "url": "http://localhost:8000",
            "description": "Development server (local)"
        },
        {
            "url": "https://staging-api.metis.ai",
            "description": "Staging environment"
        },
        {
            "url": "https://api.metis.ai",
            "description": "Production environment"
        }
    ]


def get_openapi_examples() -> Dict[str, Any]:
    """Get reusable OpenAPI examples for common request/response bodies."""
    return {
        "CreateEngagementRequest": {
            "summary": "Strategic market entry analysis",
            "description": "Example engagement for market entry strategy",
            "value": {
                "query": "How should we enter the AI market with our SaaS product?",
                "project_id": "proj_abc123",
                "config": {
                    "enable_depth_enrichment": True,
                    "max_iterations": 3,
                    "consultant_team_size": 5
                }
            }
        },
        "EngagementResponse": {
            "summary": "Created engagement",
            "description": "Example engagement response after creation",
            "value": {
                "id": "eng_xyz789",
                "query": "How should we enter the AI market with our SaaS product?",
                "status": "processing",
                "created_at": "2025-10-09T10:30:00Z",
                "project_id": "proj_abc123"
            }
        },
        "EventTraceResponse": {
            "summary": "Glass-box event trace",
            "description": "Example event trace showing pipeline execution",
            "value": {
                "engagement_id": "eng_xyz789",
                "events": [
                    {
                        "event_type": "STAGE_START",
                        "timestamp": "2025-10-09T10:30:01Z",
                        "data": {"stage_name": "socratic_questions", "stage_number": 1}
                    },
                    {
                        "event_type": "LLM_CALL",
                        "timestamp": "2025-10-09T10:30:02Z",
                        "data": {
                            "provider": "deepseek",
                            "model": "deepseek-chat",
                            "prompt": "Generate 3 clarifying questions..."
                        },
                        "metadata": {"cost_usd": 0.000002, "latency_ms": 1250}
                    },
                    {
                        "event_type": "STAGE_COMPLETE",
                        "timestamp": "2025-10-09T10:30:03Z",
                        "data": {
                            "stage_name": "socratic_questions",
                            "output": {"questions": ["Q1", "Q2", "Q3"]}
                        }
                    }
                ],
                "total_events": 45,
                "pipeline_duration_ms": 45000
            }
        },
        "FinalReport": {
            "summary": "Strategic recommendation report",
            "description": "Example final report with recommendations",
            "value": {
                "engagement_id": "eng_xyz789",
                "query": "How should we enter the AI market with our SaaS product?",
                "executive_summary": "Based on multi-perspective analysis...",
                "recommendations": [
                    {
                        "title": "Focus on vertical AI solutions",
                        "confidence": 0.85,
                        "rationale": "Market analysis shows...",
                        "evidence": ["Source 1", "Source 2"]
                    }
                ],
                "consultant_team": [
                    {"name": "Strategic MBA", "perspective": "Business strategy"},
                    {"name": "Tech Futurist", "perspective": "Technology trends"}
                ],
                "quality_score": 0.92,
                "created_at": "2025-10-09T10:31:00Z"
            }
        }
    }


def customize_openapi_schema(openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Customize the OpenAPI schema with additional metadata and examples."""

    # Add metadata
    metadata = get_openapi_metadata()
    openapi_schema.update({
        "info": {
            **openapi_schema["info"],
            **metadata
        }
    })

    # Add tags
    openapi_schema["tags"] = get_openapi_tags()

    # Add security schemes
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}

    openapi_schema["components"]["securitySchemes"].update(
        get_openapi_security_schemes()
    )

    # Add servers
    openapi_schema["servers"] = get_openapi_servers()

    # Add global examples
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "examples" not in openapi_schema["components"]:
        openapi_schema["components"]["examples"] = {}

    openapi_schema["components"]["examples"].update(get_openapi_examples())

    # Add custom x-logo extension
    if "info" in openapi_schema:
        openapi_schema["info"]["x-logo"] = {
            "url": "https://metis.ai/logo.png",
            "altText": "METIS Platform Logo"
        }

    return openapi_schema
