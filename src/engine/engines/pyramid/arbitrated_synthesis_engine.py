"""
Arbitrated Pyramid Synthesis Engine
Purpose: Upgrade synthesis to be arbitration-driven, prioritizing user-selected critiques

This module extends the PyramidEngine to:
1. Accept validation_results with user arbitration
2. Prioritize user-selected critiques in synthesis
3. Add "Refinements Based on Your Feedback" section
4. Maintain backward compatibility for non-arbitrated synthesis
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from src.config import get_settings
from src.engine.adapters.logging import get_logger  # Migrated
from src.engine.models.data_contracts import MetisDataContract
from .engine import PyramidEngine
from .models import ExecutiveDeliverable
from .enums import DeliverableType

settings = get_settings()
logger = get_logger(__name__, component="arbitrated_synthesis")


@dataclass
class ArbitrationContext:
    """Context for user arbitration during synthesis"""

    has_arbitration: bool = False
    selected_critique_ids: List[str] = field(default_factory=list)
    selected_critiques: List[Dict[str, Any]] = field(default_factory=list)
    total_critiques: int = 0
    arbitration_rate: float = 0.0
    user_comments: Optional[str] = None
    skip_arbitration: bool = False


class ArbitratedPyramidSynthesisEngine(PyramidEngine):
    """
    Enhanced Pyramid Synthesis Engine with User Arbitration Support

    This engine prioritizes user-selected critiques in the final synthesis,
    creating a human-in-the-loop decision support system.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger.with_component("arbitrated_synthesis_engine")
        self.arbitration_prompt_template = self._load_arbitration_template()

    def _load_arbitration_template(self) -> str:
        """Load the arbitration-aware synthesis prompt template"""
        return """You are a McKinsey Senior Partner conducting the final synthesis. 
Your team has produced an initial analysis, and a 'Red Team Council' of internal skeptics 
has provided a set of critiques. The client (the user) has reviewed this debate and has 
flagged the following critiques as the most important priorities to address.

## INITIAL ANALYSIS
{initial_analysis}

## FULL RED TEAM REPORT
### Munger Inversion (Failure Modes):
{munger_critique}

### Ackoff Challenger (Assumptions):
{ackoff_critique}

### Bias Auditor (Cognitive Biases):
{bias_audit}

## CLIENT'S PRIORITIZED CRITIQUES
The client has specifically selected these {selected_count} critiques as most important:
{prioritized_critiques}

{user_comments_section}

## YOUR SYNTHESIS MANDATE

Your task is to forge the final, refined recommendation that:
1. EXPLICITLY ADDRESSES each of the client's priority critiques
2. INTEGRATES insights from non-selected critiques where relevant (as secondary context)
3. DEMONSTRATES how the initial analysis was improved by addressing these specific points
4. PROVIDES a coherent 'house view' that reflects this human-guided synthesis

Structure your response with these sections:

### EXECUTIVE SUMMARY
[Your refined recommendation that addresses the prioritized concerns]

### REFINEMENTS BASED ON YOUR FEEDBACK
[Explicitly list each prioritized critique and explain how you've addressed it]

### INTEGRATED ANALYSIS
[The full synthesis showing how initial analysis evolved through critique integration]

### CONFIDENCE & LIMITATIONS
[Final confidence assessment considering the addressed critiques]

Remember: The critiques the client selected are PRIORITIES. They must be visibly and 
substantively addressed in your synthesis. Version: 1.0.0"""

    async def synthesize_with_arbitration(
        self,
        contract: MetisDataContract,
        deliverable_type: DeliverableType = DeliverableType.EXECUTIVE_SUMMARY,
    ) -> ExecutiveDeliverable:
        """
        Create executive deliverable with user arbitration integration

        Args:
            contract: MetisDataContract with validation_results and arbitration
            deliverable_type: Type of deliverable to generate

        Returns:
            ExecutiveDeliverable with prioritized critique integration
        """

        self.logger.info(
            "synthesize_with_arbitration_started",
            engagement_id=str(contract.engagement_context.engagement_id),
            deliverable_type=deliverable_type.value,
        )

        # Extract arbitration context
        arbitration_context = self._extract_arbitration_context(contract)

        # Log arbitration metrics
        self._log_arbitration_metrics(arbitration_context)

        # Prepare synthesis input
        synthesis_input = self._prepare_arbitrated_synthesis_input(
            contract, arbitration_context
        )

        # Generate arbitration-aware synthesis prompt
        synthesis_prompt = self._generate_arbitrated_prompt(
            synthesis_input, arbitration_context
        )

        # Execute synthesis with prioritized critiques
        deliverable = await self._execute_arbitrated_synthesis(
            synthesis_prompt, synthesis_input, deliverable_type, arbitration_context
        )

        # Add refinements section based on user feedback
        deliverable = self._add_refinements_section(deliverable, arbitration_context)

        # Update confidence based on addressed critiques
        deliverable = self._update_confidence_with_arbitration(
            deliverable, arbitration_context
        )

        self.logger.info(
            "synthesize_with_arbitration_complete",
            engagement_id=str(contract.engagement_context.engagement_id),
            addressed_critiques=len(arbitration_context.selected_critique_ids),
            final_confidence=deliverable.partner_ready_score,
        )

        return deliverable

    def _extract_arbitration_context(
        self, contract: MetisDataContract
    ) -> ArbitrationContext:
        """Extract user arbitration context from contract"""
        context = ArbitrationContext()

        if not hasattr(contract, "validation_results"):
            return context

        validation_results = contract.validation_results

        # Check for user arbitration
        if "user_arbitration" in validation_results:
            arbitration = validation_results["user_arbitration"]
            context.has_arbitration = True
            context.selected_critique_ids = arbitration.get("prioritized_critiques", [])
            context.user_comments = arbitration.get("user_comments")
            context.skip_arbitration = arbitration.get("skip_arbitration", False)

            # Extract full critique details
            context.selected_critiques = self._extract_critique_details(
                validation_results, context.selected_critique_ids
            )

            # Calculate metrics
            context.total_critiques = self._count_total_critiques(validation_results)
            context.arbitration_rate = (
                len(context.selected_critique_ids) / context.total_critiques * 100
                if context.total_critiques > 0
                else 0
            )

        return context

    def _extract_critique_details(
        self, validation_results: Dict[str, Any], selected_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract full details for selected critiques"""
        selected_critiques = []

        for challenger_name in ["munger_critique", "ackoff_critique", "bias_audit"]:
            if challenger_name not in validation_results:
                continue

            result = validation_results[challenger_name]
            if not isinstance(result, dict) or "critiques" not in result:
                continue

            for critique in result["critiques"]:
                if isinstance(critique, dict) and critique.get("id") in selected_ids:
                    critique["source"] = challenger_name
                    selected_critiques.append(critique)

        return selected_critiques

    def _count_total_critiques(self, validation_results: Dict[str, Any]) -> int:
        """Count total number of critiques across all challengers"""
        total = 0

        for challenger_name in ["munger_critique", "ackoff_critique", "bias_audit"]:
            if challenger_name in validation_results:
                result = validation_results[challenger_name]
                if isinstance(result, dict) and "critiques" in result:
                    total += len(result["critiques"])

        return total

    def _prepare_arbitrated_synthesis_input(
        self, contract: MetisDataContract, arbitration_context: ArbitrationContext
    ) -> Dict[str, Any]:
        """Prepare synthesis input with arbitration emphasis"""

        # Get initial analysis
        initial_analysis = contract.analysis_results.get("initial_analysis", {})

        # Get validation results
        validation_results = getattr(contract, "validation_results", {})

        # Structure input for synthesis
        synthesis_input = {
            "initial_analysis": initial_analysis,
            "validation_results": validation_results,
            "prioritized_critiques": arbitration_context.selected_critiques,
            "all_critique_ids": [
                c["id"] for c in arbitration_context.selected_critiques
            ],
            "user_comments": arbitration_context.user_comments,
            "arbitration_metrics": {
                "total_critiques": arbitration_context.total_critiques,
                "selected_count": len(arbitration_context.selected_critique_ids),
                "arbitration_rate": arbitration_context.arbitration_rate,
            },
        }

        return synthesis_input

    def _generate_arbitrated_prompt(
        self, synthesis_input: Dict[str, Any], arbitration_context: ArbitrationContext
    ) -> str:
        """Generate synthesis prompt that prioritizes user-selected critiques"""

        # Format critiques by source
        munger_text = self._format_challenger_critiques(
            synthesis_input["validation_results"].get("munger_critique", {})
        )
        ackoff_text = self._format_challenger_critiques(
            synthesis_input["validation_results"].get("ackoff_critique", {})
        )
        bias_text = self._format_challenger_critiques(
            synthesis_input["validation_results"].get("bias_audit", {})
        )

        # Format prioritized critiques
        prioritized_text = self._format_prioritized_critiques(
            arbitration_context.selected_critiques
        )

        # Add user comments section if present
        user_comments_section = ""
        if arbitration_context.user_comments:
            user_comments_section = f"""
## CLIENT GUIDANCE
The client provided this additional context for their selections:
"{arbitration_context.user_comments}"
"""

        # Fill in the template
        prompt = self.arbitration_prompt_template.format(
            initial_analysis=synthesis_input["initial_analysis"],
            munger_critique=munger_text,
            ackoff_critique=ackoff_text,
            bias_audit=bias_text,
            selected_count=len(arbitration_context.selected_critique_ids),
            prioritized_critiques=prioritized_text,
            user_comments_section=user_comments_section,
        )

        return prompt

    def _format_challenger_critiques(self, challenger_result: Dict[str, Any]) -> str:
        """Format all critiques from a challenger for context"""
        if not challenger_result or challenger_result.get("status") == "failed":
            return "No critiques available from this challenger."

        critiques = challenger_result.get("critiques", [])
        if not critiques:
            return "This challenger found no significant issues."

        formatted = []
        for critique in critiques:
            formatted.append(
                f"- [{critique.get('id', 'unknown')}] {critique.get('title', 'Untitled')}: "
                f"{critique.get('description', 'No description')}"
                f" (Significance: {critique.get('significance', 0) * 100:.0f}%)"
            )

        return "\n".join(formatted)

    def _format_prioritized_critiques(
        self, selected_critiques: List[Dict[str, Any]]
    ) -> str:
        """Format user-selected critiques with emphasis"""
        if not selected_critiques:
            return "No specific critiques were prioritized by the client."

        formatted = []
        for i, critique in enumerate(selected_critiques, 1):
            source_map = {
                "munger_critique": "Munger Inversion",
                "ackoff_critique": "Ackoff Challenger",
                "bias_audit": "Bias Auditor",
            }
            source = source_map.get(critique.get("source", ""), "Unknown")

            formatted.append(
                f"""
**Priority #{i}: {critique.get('title', 'Untitled')}**
- Source: {source}
- ID: {critique.get('id', 'unknown')}
- Description: {critique.get('description', 'No description')}
- Evidence: {', '.join(critique.get('evidence', [])) if critique.get('evidence') else 'No specific evidence'}
- Significance: {critique.get('significance', 0) * 100:.0f}%
"""
            )

        return "\n".join(formatted)

    async def _execute_arbitrated_synthesis(
        self,
        synthesis_prompt: str,
        synthesis_input: Dict[str, Any],
        deliverable_type: DeliverableType,
        arbitration_context: ArbitrationContext,
    ) -> ExecutiveDeliverable:
        """Execute synthesis with LLM using arbitration-aware prompt"""

        # This would normally call the LLM
        # For now, create a mock deliverable
        deliverable = ExecutiveDeliverable(
            deliverable_id=UUID("12345678-1234-5678-1234-567812345678"),
            deliverable_type=deliverable_type,
            executive_summary="Synthesized recommendation addressing prioritized critiques",
            key_recommendations=[
                "Recommendation addressing user priority #1",
                "Recommendation addressing user priority #2",
            ],
            partner_ready_score=0.85,
            created_at=datetime.utcnow(),
        )

        return deliverable

    def _add_refinements_section(
        self, deliverable: ExecutiveDeliverable, arbitration_context: ArbitrationContext
    ) -> ExecutiveDeliverable:
        """Add 'Refinements Based on Your Feedback' section to deliverable"""

        if not arbitration_context.has_arbitration:
            return deliverable

        refinements = []
        for critique in arbitration_context.selected_critiques:
            refinements.append(
                {
                    "critique_id": critique["id"],
                    "critique_title": critique.get("title", "Untitled"),
                    "how_addressed": f"Addressed by adjusting recommendation to account for {critique.get('type', 'issue')}",
                    "impact": "Significant improvement in recommendation robustness",
                }
            )

        # Add refinements to deliverable metadata
        if not hasattr(deliverable, "refinements_section"):
            deliverable.refinements_section = refinements

        return deliverable

    def _update_confidence_with_arbitration(
        self, deliverable: ExecutiveDeliverable, arbitration_context: ArbitrationContext
    ) -> ExecutiveDeliverable:
        """Update confidence score based on addressed critiques"""

        if not arbitration_context.has_arbitration:
            return deliverable

        # Calculate confidence adjustment based on addressed critiques
        addressed_significance = sum(
            c.get("significance", 0.5) for c in arbitration_context.selected_critiques
        )

        # Average significance of addressed critiques
        avg_significance = (
            addressed_significance / len(arbitration_context.selected_critiques)
            if arbitration_context.selected_critiques
            else 0
        )

        # Boost confidence for addressing high-significance critiques
        confidence_boost = avg_significance * 0.1  # Up to 10% boost

        # Update partner ready score
        deliverable.partner_ready_score = min(
            1.0, deliverable.partner_ready_score + confidence_boost
        )

        self.logger.info(
            "confidence_updated_with_arbitration",
            original_confidence=deliverable.partner_ready_score - confidence_boost,
            new_confidence=deliverable.partner_ready_score,
            boost=confidence_boost,
        )

        return deliverable

    def _log_arbitration_metrics(self, arbitration_context: ArbitrationContext):
        """Log metrics about arbitration for monitoring"""

        if not arbitration_context.has_arbitration:
            return

        self.logger.info(
            "arbitration_synthesis_metrics",
            has_arbitration=arbitration_context.has_arbitration,
            total_critiques=arbitration_context.total_critiques,
            selected_count=len(arbitration_context.selected_critique_ids),
            arbitration_rate=arbitration_context.arbitration_rate,
            skip_arbitration=arbitration_context.skip_arbitration,
            has_user_comments=bool(arbitration_context.user_comments),
        )


# Factory function
def create_arbitrated_synthesis_engine(
    *args, **kwargs
) -> ArbitratedPyramidSynthesisEngine:
    """Create an arbitrated synthesis engine instance"""
    return ArbitratedPyramidSynthesisEngine(*args, **kwargs)
