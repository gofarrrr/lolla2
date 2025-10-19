# Senior Advisor V2 - WSN synthesis over Argument Graph
from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path

from src.models.cognitive_core import Argument
from src.services.cognitive_core_service import CognitiveCoreService
from src.services.coreops_dsl import parse_coreops_yaml
from src.orchestration.coreops_executor import execute_core_program


class SeniorAdvisorV2:
    def __init__(self, core: Optional[CognitiveCoreService] = None):
        self.core = core or CognitiveCoreService(mock_mode=True)
        self.dsl_template_path = Path("examples/coreops/senior_advisor_wsn.yaml")

    async def synthesize_report(
        self, trace_id: str, arguments: List[Argument]
    ) -> Dict[str, Any]:
        # Select bounded syntheses as primary inputs
        bounded = [
            a
            for a in arguments
            if getattr(a.status, "value", getattr(a.status, "name", ""))
            in ("bounded_validity",)
        ]
        if not bounded:
            bounded = arguments

        # Compose WSN strings
        what_items = [a.claim for a in bounded[:3]]
        WHAT = "; ".join(what_items) if what_items else "Consolidated synthesis"

        # Extract boundary conditions from claims (naive parse of 'under conditions')
        boundaries: List[str] = []
        for a in bounded:
            lower = a.claim.lower()
            if "under conditions" in lower:
                # include substring following 'under conditions'
                idx = lower.find("under conditions")
                boundaries.append(a.claim[idx:])
        boundaries_text = (
            "; ".join(boundaries)
            if boundaries
            else "Boundary conditions documented in synthesis nodes."
        )
        SO_WHAT = f"Impact and risks depend on {boundaries_text}"

        NOW_WHAT = "Run a timeboxed pilot under the boundary conditions; measure outcomes; expand if successful."

        # Load DSL template and substitute
        yaml_text = self.dsl_template_path.read_text()
        yaml_text = (
            yaml_text.replace("${WHAT}", WHAT)
            .replace("${SO_WHAT}", SO_WHAT)
            .replace("${NOW_WHAT}", NOW_WHAT)
        )
        program = parse_coreops_yaml(yaml_text)

        results = await execute_core_program(program, self.core, trace_id)
        wsn_args = list(results.values())
        return {
            "what": WHAT,
            "so_what": SO_WHAT,
            "now_what": NOW_WHAT,
            "wsn_arguments": wsn_args,
        }
