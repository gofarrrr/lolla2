"""Schema validation framework with version hashing support."""

import hashlib
import json
from typing import Any, Dict, Optional
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

class SchemaVersion(BaseModel):
    """Schema version information."""
    name: str
    version: str
    hash: str
    schema: Dict[str, Any]

class SchemaRegistry:
    """Central registry for schema validation and versioning."""
    
    def __init__(self):
        self._schemas: Dict[str, SchemaVersion] = {}
        
    def register_schema(self, name: str, schema: Dict[str, Any], version: str) -> None:
        """Register a new schema version."""
        schema_hash = self._compute_schema_hash(schema)
        self._schemas[name] = SchemaVersion(
            name=name,
            version=version,
            hash=schema_hash,
            schema=schema
        )
    
    def validate(self, name: str, data: Dict[str, Any]) -> None:
        """Validate data against registered schema."""
        if name not in self._schemas:
            raise HTTPException(status_code=400, detail=f"Unknown schema: {name}")
            
        try:
            # Create a dynamic Pydantic model from the schema
            model = self._create_model(name)
            model.model_validate(data)
            
            # Enforce enum constraints from the JSON schema
            schema = self._schemas[name].schema
            for field_name, field_schema in schema.get("properties", {}).items():
                if "enum" in field_schema and field_name in data:
                    if data[field_name] not in field_schema["enum"]:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Field '{field_name}' must be one of {field_schema['enum']}"
                        )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    def _create_model(self, name: str) -> type[BaseModel]:
        """Create a Pydantic model from a schema definition."""
        schema = self._schemas[name].schema
        
        # Create model annotations
        annotations = {}
        required = schema.get("required", [])
        for field_name, field_schema in schema.get("properties", {}).items():
            field_type = self._get_field_type(field_schema)
            annotations[field_name] = field_type
            
        # Create model config
        namespace = {
            "__annotations__": annotations,
            "model_config": {
                "extra": "forbid"
            }
        }
        
        # Add default values for optional fields
        for field_name, field_schema in schema.get("properties", {}).items():
            if field_name not in required:
                namespace[field_name] = field_schema.get("default", None)
            
        return type(f"{name.title()}Model", (BaseModel,), namespace)
        
    def _get_field_type(self, field_schema: Dict[str, Any]) -> type:
        """Convert JSON Schema types to Python types."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        return type_mapping.get(field_schema.get("type"), Any)
            
    def get_schema_version(self, name: str) -> Optional[SchemaVersion]:
        """Get schema version info."""
        return self._schemas.get(name)
    
    def _compute_schema_hash(self, schema: Dict[str, Any]) -> str:
        """Compute deterministic hash of schema."""
        schema_json = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()

# Global schema registry instance
schema_registry = SchemaRegistry()