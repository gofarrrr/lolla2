from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import re


class Scope(BaseModel):
    job: str
    audience: str
    in_scope: List[str]
    out_of_scope: List[str]
    success: List[str]
    assumptions: List[str] = []
    guardrails: List[str] = []


class Inputs(BaseModel):
    required: List[str]
    optional: List[str] = []
    normalization: List[str] = []
    on_failure: str = Field(default="diagnostic-json")


class Rules(BaseModel):
    rounding: int = 1
    signed_numbers: bool = True
    tone: str = "plain_english_pm_designer"
    language: str = "uk_en"
    pii: str = "aggregate_only"


class OutputSpec(BaseModel):
    formats: List[str] = ["markdown", "json"]
    json_schema_version: Optional[str] = None


class Evaluation(BaseModel):
    rubric: List[str]
    auto_repair: bool = True


class SystemContract(BaseModel):
    contract_id: str
    scope: Scope
    inputs: Inputs
    process: List[str]
    rules: Rules
    output: OutputSpec
    evaluation: Evaluation

    @field_validator("contract_id")
    @classmethod
    def ensure_versioned_contract_id(cls, v: str) -> str:
        """
        Ensure the contract_id follows a versioned identifier pattern: NAME@X.Y[.Z]
        If no version is provided, default to @1.0.
        """
        if "@" not in v:
            # Auto-upgrade legacy IDs to versioned format
            return f"{v}@1.0"
        # Validate pattern (alphanumeric/underscore/dash name + semver)
        pattern = re.compile(r"^[A-Za-z0-9_\-]+@[0-9]+\.[0-9]+(\.[0-9]+)?$")
        if not pattern.match(v):
            raise ValueError(
                "contract_id must match NAME@MAJOR.MINOR[.PATCH], e.g., NWAY_STRATEGIC_ANALYSIS_CORE@1.0"
            )
        return v

    @property
    def name(self) -> str:
        return self.contract_id.split("@")[0]

    @property
    def version(self) -> str:
        parts = self.contract_id.split("@")
        return parts[1] if len(parts) > 1 else "1.0"
