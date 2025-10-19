"""Schema definitions for API endpoints."""

ANALYSIS_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1
        },
        "context": {
            "type": "object",
            "default": {}
        },
        "complexity": {
            "type": "string",
            "enum": ["auto", "simple", "strategic", "complex"],
            "default": "auto"
        }
    },
    "required": ["query"]
}

SERVICES_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["healthy", "unhealthy", "degraded"]
        },
        "version": {
            "type": "string"
        },
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["healthy", "unhealthy", "degraded"]
                    }
                },
                "required": ["name", "status"]
            }
        }
    },
    "required": ["status", "version", "components"]
}