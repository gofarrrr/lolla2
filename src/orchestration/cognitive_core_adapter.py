# Cognitive Core Adapter - bridges legacy orchestrators with CoreOps DSL + CognitiveCoreService
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.services.cognitive_core_service import CognitiveCoreService
from src.services.coreops_dsl import parse_coreops_yaml
from src.orchestration.coreops_executor import execute_core_program
from src.models.cognitive_core import Argument

# Optional import of framework types (used for Problem Structuring)
try:
    from src.orchestration.contracts import (
        StructuredAnalyticalFramework,
        AnalyticalDimension,
        FrameworkType,
    )
except Exception:
    StructuredAnalyticalFramework = None  # type: ignore
    AnalyticalDimension = None  # type: ignore
    FrameworkType = None  # type: ignore

logger = logging.getLogger(__name__)


class CognitiveCoreAdapter:
    """
    Adapter layer to run CoreOps programs and return results in legacy-friendly forms.

    Methods:
    - execute_analysis(system_contract_id, context): runs mapped DSL and returns argument graph
    - For problem_structuring@1.0, also maps claims into StructuredAnalyticalFramework
    """

    PROGRAM_MAP = {
        "problem_structuring@1.0": "examples/coreops/problem_structuring.yaml",
        "mckinsey_strategist@1.0": "examples/coreops/mckinsey_strategist.yaml",
        "risk_assessor@1.0": "examples/coreops/risk_assessor.yaml",
        "operations_expert@1.0": "examples/coreops/operations_expert.yaml",
        "market_researcher@1.0": "examples/coreops/market_researcher.yaml",
        "financial_analyst@1.0": "examples/coreops/financial_analyst.yaml",
        "strategic_analyst@1.0": "examples/coreops/strategic_analyst.yaml",
        "technology_advisor@1.0": "examples/coreops/technology_advisor.yaml",
        "implementation_specialist@1.0": "examples/coreops/implementation_specialist.yaml",
        "innovation_consultant@1.0": "examples/coreops/innovation_consultant.yaml",
        "crisis_manager@1.0": "examples/coreops/crisis_manager.yaml",
        "turnaround_specialist@1.0": "examples/coreops/turnaround_specialist.yaml",
        "competitive_analyst@1.0": "examples/coreops/competitive_analyst.yaml",
    }

    def __init__(self, core: Optional[CognitiveCoreService] = None):
        self.core = core or CognitiveCoreService(mock_mode=True)

    async def execute_analysis(
        self, system_contract_id: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        program_rel = self.PROGRAM_MAP.get(system_contract_id)
        if not program_rel:
            raise ValueError(
                f"No program mapped for system contract: {system_contract_id}"
            )

        program_path = Path(program_rel)
        if not program_path.is_absolute():
            # resolve relative to repo root (assumes CWD is repo root)
            program_path = Path.cwd() / program_path
        yaml_text = program_path.read_text()

        program = parse_coreops_yaml(yaml_text)
        trace_id = self.core.context_stream.trace_id

        # Record execution start time for evidence
        import time

        start_time = time.time()

        results = await execute_core_program(program, self.core, trace_id)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        arguments: List[Argument] = list(results.values())
        payload: Dict[str, Any] = {
            "trace_id": trace_id,
            "arguments": arguments,
        }

        # 🔍 GLASS-BOX EVIDENCE: Record CoreOps execution proof
        self._record_coreops_execution_evidence(
            system_contract_id=system_contract_id,
            program_path=str(program_path),
            step_count=len(program.steps),
            arguments=arguments,
            processing_time_ms=processing_time_ms,
            trace_id=trace_id,
        )

        # Special mapping for problem structuring
        if (
            system_contract_id == "problem_structuring@1.0"
            and StructuredAnalyticalFramework is not None
        ):
            framework = self._build_framework_from_arguments(arguments)
            payload["framework"] = framework

        return payload

    def _build_framework_from_arguments(self, arguments: List[Argument]):
        """Parse dimension claims of the form:
        "Dimension: NAME | Questions: q1; q2; q3 | Approach: TEXT | Priority: N"
        into AnalyticalDimension objects and assemble a StructuredAnalyticalFramework.
        """
        if AnalyticalDimension is None or StructuredAnalyticalFramework is None:
            return None

        dims: List[Any] = []
        for arg in arguments:
            text = arg.claim
            if not text.lower().startswith("dimension:"):
                continue
            # naive parse
            parts = [p.strip() for p in text.split("|")]
            name = parts[0].split(":", 1)[1].strip() if ":" in parts[0] else parts[0]
            questions: List[str] = []
            approach = ""
            priority = 3
            for p in parts[1:]:
                if p.lower().startswith("questions:"):
                    qstr = p.split(":", 1)[1]
                    questions = [q.strip() for q in qstr.split(";") if q.strip()]
                elif p.lower().startswith("approach:"):
                    approach = p.split(":", 1)[1].strip()
                elif p.lower().startswith("priority:"):
                    try:
                        priority = int(p.split(":", 1)[1].strip())
                    except Exception:
                        priority = 3
            dim = AnalyticalDimension(
                dimension_name=name,
                key_questions=questions,
                analysis_approach=approach,
                priority_level=priority,
            )
            dims.append(dim)

        framework_type = FrameworkType("strategic_analysis") if FrameworkType else None
        sequence = [d.dimension_name for d in dims]
        consultant_types = ["strategic_analyst", "market_researcher"]
        complexity = "MODERATE"

        framework = StructuredAnalyticalFramework(
            framework_type=framework_type,
            primary_dimensions=dims,
            secondary_considerations=[],
            analytical_sequence=sequence,
            complexity_assessment=complexity,
            recommended_consultant_types=consultant_types,
            processing_time_seconds=0.0,
            timestamp=None,
        )
        return framework

    def _record_coreops_execution_evidence(
        self,
        system_contract_id: str,
        program_path: str,
        step_count: int,
        arguments: List[Argument],
        processing_time_ms: int,
        trace_id: str,
    ) -> None:
        """Record glass-box evidence of V2 CoreOps execution"""

        # Extract sample claims (first 2 for evidence)
        sample_claims = [arg.claim for arg in arguments[:2]] if arguments else []

        # Evidence data structure
        evidence_data = {
            "system_contract_id": system_contract_id,
            "program_path": program_path,
            "step_count": step_count,
            "argument_count": len(arguments),
            "sample_claims": sample_claims,
            "rag_evidence_ids": [],  # TODO: Extract from RAG if available
            "processing_time_ms": processing_time_ms,
            "execution_mode": "v2_coreops",
            "v2_proof": True,
            "trace_id": trace_id,
        }

        # Log to UnifiedContextStream
        from src.core.unified_context_stream import ContextEventType

        self.core.context_stream.add_event(
            event_type=ContextEventType.COREOPS_RUN_SUMMARY,
            data=evidence_data,
            metadata={
                "evidence_type": "coreops_execution",
                "audit_level": "complete",
                "trace_id": trace_id,
                "contract_id": system_contract_id,
            },
        )

        logger.info(
            f"🔍 CoreOps Evidence: {system_contract_id} executed {step_count} steps, {len(arguments)} arguments in {processing_time_ms}ms"
        )
