# CoreOps DSL Parser (Walking Skeleton)
# MVP: parse a linear sequence of core.create_argument ops from YAML

from __future__ import annotations

from typing import List, Dict, Any

import yaml
from src.config.models import CoreProgramModel, CoreOpModel

# Backwards-compatibility type aliases
CoreProgram = CoreProgramModel
CoreOp = CoreOpModel


def parse_coreops_yaml(yaml_text: str) -> CoreProgramModel:
    data = yaml.safe_load(yaml_text) or {}
    # Let Pydantic validate and coerce types
    # Ensure steps default to []
    data.setdefault("steps", [])
    # Build typed steps if they are dicts
    if isinstance(data.get("steps"), list):
        data["steps"] = [CoreOpModel(**s).model_dump() if isinstance(s, dict) else s for s in data["steps"]]
    return CoreProgramModel(**data)
