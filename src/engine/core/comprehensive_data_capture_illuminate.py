"""
Comprehensive Data Capture System - OPERATION ILLUMINATE Enhanced
Phase 3-4: Complete system flow capture with glass-box transparency

Enhanced with LLMInteraction and ResearchInteraction capture
Includes encryption for prompt security and devil's advocate reasoning
"""

import json
import logging
import time
import uuid
import os
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Optional encryption support
try:
    from cryptography.fernet import Fernet

    ENCRYPTION_AVAILABLE = True
except ImportError:
    Fernet = None
    ENCRYPTION_AVAILABLE = False

# Import Operation Illuminate data models
try:
    from src.core.enhanced_llm_manager import LLMInteraction
    from src.engine.integrations.perplexity_client_illuminate import ResearchInteraction
except ImportError:
    # Fallback in case imports fail
    pass

logger = logging.getLogger(__name__)


@dataclass
class DevilsAdvocateCapture:
    """Captures Devil's Advocate reasoning and debate process"""

    debate_id: str
    timestamp: str
    challenger_name: str
    challenge_type: str  # assumption, logic, evidence, conclusion
    original_claim: str
    challenge_reasoning: str
    evidence_presented: List[str]
    counter_arguments: List[str]
    resolution_method: str
    confidence_before: float
    confidence_after: float
    outcome: str  # accepted, rejected, modified, deferred
    duration_ms: int


@dataclass
class CognitiveProcessingIlluminate:
    """Enhanced cognitive processing capture with complete reasoning chains"""

    processing_id: str
    timestamp: str
    phase: str
    input_query: str
    mental_models_applied: List[str]
    reasoning_steps: List[Dict[str, Any]]
    assumptions_made: List[str]
    evidence_considered: List[str]
    conflicts_detected: List[Dict[str, Any]]
    confidence_evolution: List[Dict[str, float]]
    final_output: Dict[str, Any]
    duration_ms: int


@dataclass
class SystemFlowIlluminate:
    """Enhanced system flow capture for Operation Illuminate"""

    flow_id: str
    timestamp: str
    user_query: str
    engagement_id: str

    # OPERATION ILLUMINATE: Enhanced capture arrays
    llm_interactions: List[Dict[str, Any]]  # LLMInteraction objects
    research_interactions: List[Dict[str, Any]]  # ResearchInteraction objects
    devils_advocate_debates: List[DevilsAdvocateCapture]
    cognitive_processing: List[CognitiveProcessingIlluminate]

    # Traditional captures (for compatibility)
    database_operations: List[Dict[str, Any]]

    # Final outputs
    final_response: Dict[str, Any]
    total_duration_ms: int
    total_cost_usd: float

    # OPERATION ILLUMINATE: Transparency metrics
    total_llm_calls: int = 0
    total_research_queries: int = 0
    total_contradictions_found: int = 0
    total_assumptions_challenged: int = 0
    glass_box_completeness_score: float = 0.0


class ComprehensiveDataCaptureIlluminate:
    """
    OPERATION ILLUMINATE Enhanced Data Capture System
    Captures complete cognitive journey with glass-box transparency
    """

    def __init__(self, output_dir: str = "data_captures", contract: Any = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # OPERATION ILLUMINATE: Link to MetisDataContract
        self.contract = contract

        # Active captures
        self.current_flow: Optional[SystemFlowIlluminate] = None
        self.capture_active = False

        # OPERATION ILLUMINATE: Memory safety
        self.MAX_CAPTURE_SIZE_MB = float(os.getenv("MAX_CAPTURE_SIZE_MB", "10"))
        self.current_capture_size = 0

        # OPERATION ILLUMINATE: Encryption for prompt security
        self._setup_encryption()

        logger.info(
            f"🎬 ILLUMINATE: Comprehensive Data Capture initialized - output: {self.output_dir}"
        )

    def _setup_encryption(self):
        """Setup encryption for prompt security"""
        if not ENCRYPTION_AVAILABLE:
            logger.warning(
                "🔐 ILLUMINATE: Cryptography not available - using plaintext fallback"
            )
            self.cipher = None
            return

        encryption_key = os.getenv("TRANSPARENCY_ENCRYPTION_KEY")
        if not encryption_key:
            # Generate a key for this session (in production, should be persistent)
            encryption_key = Fernet.generate_key().decode()
            os.environ["TRANSPARENCY_ENCRYPTION_KEY"] = encryption_key

        try:
            self.cipher = Fernet(
                encryption_key.encode()
                if isinstance(encryption_key, str)
                else encryption_key
            )
            logger.info("🔐 ILLUMINATE: Encryption system initialized")
        except Exception as e:
            logger.warning(
                f"🔐 ILLUMINATE: Encryption setup failed: {e} - using plaintext fallback"
            )
            self.cipher = None

    def _encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data like prompts"""
        if not self.cipher or not data:
            return data

        try:
            encrypted = self.cipher.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.warning(f"🔐 Encryption failed: {e} - using plaintext")
            return data

    def _estimate_capture_size(self) -> int:
        """Estimate current capture size in bytes"""
        if not self.current_flow:
            return 0

        # Rough estimate of flow size
        flow_dict = asdict(self.current_flow)
        return len(json.dumps(flow_dict, default=str))

    @contextmanager
    def capture_system_flow(self, user_query: str, engagement_id: str = ""):
        """OPERATION ILLUMINATE: Context manager for capturing complete system flow"""
        flow_id = str(uuid.uuid4())
        start_time = time.time()

        self.current_flow = SystemFlowIlluminate(
            flow_id=flow_id,
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            engagement_id=engagement_id,
            llm_interactions=[],
            research_interactions=[],
            devils_advocate_debates=[],
            cognitive_processing=[],
            database_operations=[],
            final_response={},
            total_duration_ms=0,
            total_cost_usd=0.0,
        )

        self.capture_active = True
        logger.info(
            f"🎬 ILLUMINATE: Started comprehensive capture for flow: {flow_id[:8]}"
        )

        try:
            yield self
        finally:
            # Finalize capture
            end_time = time.time()
            self.current_flow.total_duration_ms = int((end_time - start_time) * 1000)

            # Calculate total costs
            llm_cost = sum(
                interaction.get("cost_usd", 0.0)
                for interaction in self.current_flow.llm_interactions
            )
            research_cost = sum(
                interaction.get("cost_usd", 0.0)
                for interaction in self.current_flow.research_interactions
            )
            self.current_flow.total_cost_usd = llm_cost + research_cost

            # Calculate transparency metrics
            self._calculate_transparency_metrics()

            # Store in contract if available
            if self.contract:
                self._store_in_contract()

            # Save complete flow
            self._save_flow_data()
            self.capture_active = False

            logger.info(
                f"🎬 ILLUMINATE: Completed comprehensive capture - "
                f"LLM: {len(self.current_flow.llm_interactions)}, "
                f"Research: {len(self.current_flow.research_interactions)}, "
                f"Debates: {len(self.current_flow.devils_advocate_debates)}, "
                f"Cost: ${self.current_flow.total_cost_usd:.4f}"
            )

    def capture_llm_interaction_illuminate(self, interaction: "LLMInteraction"):
        """OPERATION ILLUMINATE: Capture complete LLM interaction with memory safety"""
        if not self.capture_active:
            return

        # Memory safety check
        if self._estimate_capture_size() > self.MAX_CAPTURE_SIZE_MB * 1024 * 1024:
            logger.warning(
                "🎬 ILLUMINATE: Capture size limit reached, truncating response"
            )
            interaction.raw_response = f"[TRUNCATED: Response too large, size: {len(interaction.raw_response)} chars]"

        # Convert to dict and encrypt sensitive data
        interaction_dict = asdict(interaction)

        # Encrypt prompts for at-rest storage while keeping available for debugging
        interaction_dict["system_prompt_encrypted"] = self._encrypt_sensitive_data(
            interaction_dict["system_prompt"]
        )
        interaction_dict["user_prompt_encrypted"] = self._encrypt_sensitive_data(
            interaction_dict["user_prompt"]
        )

        self.current_flow.llm_interactions.append(interaction_dict)
        self.current_flow.total_llm_calls += 1

        logger.info(
            f"📝 ILLUMINATE: Captured LLM interaction {interaction.interaction_id[:8]} - "
            f"{interaction.tokens_used} tokens, ${interaction.cost_usd:.4f}"
        )

    def capture_research_interaction_illuminate(
        self, interaction: "ResearchInteraction"
    ):
        """OPERATION ILLUMINATE: Capture complete research interaction"""
        if not self.capture_active:
            return

        # Convert to dict
        interaction_dict = asdict(interaction)

        # Track contradictions
        if interaction_dict.get("contradiction_detection_result", {}).get(
            "detected", False
        ):
            self.current_flow.total_contradictions_found += 1

        self.current_flow.research_interactions.append(interaction_dict)
        self.current_flow.total_research_queries += 1

        logger.info(
            f"🔍 ILLUMINATE: Captured research interaction {interaction.research_id[:8]} - "
            f"{interaction.sources_consulted_count} sources, ${interaction.cost_usd:.4f}"
        )

    def capture_devils_advocate_debate(
        self,
        challenger_name: str,
        challenge_type: str,
        original_claim: str,
        challenge_reasoning: str,
        evidence_presented: List[str],
        counter_arguments: List[str],
        resolution_method: str,
        confidence_before: float,
        confidence_after: float,
        outcome: str,
        duration_ms: int = 0,
    ):
        """OPERATION ILLUMINATE: Capture Devil's Advocate debate process"""
        if not self.capture_active:
            return

        debate = DevilsAdvocateCapture(
            debate_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            challenger_name=challenger_name,
            challenge_type=challenge_type,
            original_claim=original_claim,
            challenge_reasoning=challenge_reasoning,
            evidence_presented=evidence_presented,
            counter_arguments=counter_arguments,
            resolution_method=resolution_method,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            outcome=outcome,
            duration_ms=duration_ms,
        )

        self.current_flow.devils_advocate_debates.append(debate)
        self.current_flow.total_assumptions_challenged += 1

        logger.info(
            f"⚔️ ILLUMINATE: Captured debate {debate.debate_id[:8]} - "
            f"{challenger_name} {outcome} claim: {original_claim[:50]}..."
        )

    def capture_cognitive_processing_illuminate(
        self,
        phase: str,
        input_query: str,
        mental_models_applied: List[str],
        reasoning_steps: List[Dict[str, Any]],
        assumptions_made: List[str],
        evidence_considered: List[str],
        conflicts_detected: List[Dict[str, Any]],
        confidence_evolution: List[Dict[str, float]],
        final_output: Dict[str, Any],
        duration_ms: int = 0,
    ):
        """OPERATION ILLUMINATE: Capture cognitive processing with complete reasoning chains"""
        if not self.capture_active:
            return

        processing = CognitiveProcessingIlluminate(
            processing_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            phase=phase,
            input_query=input_query,
            mental_models_applied=mental_models_applied,
            reasoning_steps=reasoning_steps,
            assumptions_made=assumptions_made,
            evidence_considered=evidence_considered,
            conflicts_detected=conflicts_detected,
            confidence_evolution=confidence_evolution,
            final_output=final_output,
            duration_ms=duration_ms,
        )

        self.current_flow.cognitive_processing.append(processing)

        logger.info(
            f"🧠 ILLUMINATE: Captured cognitive processing {processing.processing_id[:8]} - "
            f"Phase: {phase}, Models: {len(mental_models_applied)}, Steps: {len(reasoning_steps)}"
        )

    def set_final_response(self, response: Dict[str, Any]):
        """Set the final response for the current flow"""
        if self.capture_active and self.current_flow:
            self.current_flow.final_response = response
            logger.info("📤 ILLUMINATE: Captured final response")

    def _calculate_transparency_metrics(self):
        """Calculate glass-box transparency completeness score"""
        if not self.current_flow:
            return

        score = 0.0
        max_score = 100.0

        # LLM transparency (25 points)
        if self.current_flow.llm_interactions:
            score += 25.0

        # Research transparency (25 points)
        if self.current_flow.research_interactions:
            score += 25.0

        # Debate transparency (25 points)
        if self.current_flow.devils_advocate_debates:
            score += 25.0

        # Cognitive processing transparency (25 points)
        if self.current_flow.cognitive_processing:
            score += 25.0

        self.current_flow.glass_box_completeness_score = score / max_score

        logger.info(
            f"📊 ILLUMINATE: Transparency score: {self.current_flow.glass_box_completeness_score:.1%}"
        )

    def _store_in_contract(self):
        """Store captured data in MetisDataContract for immediate access"""
        if not self.contract or not self.current_flow:
            return

        # Store raw outputs if not already stored
        if not hasattr(self.contract, "raw_outputs"):
            self.contract.raw_outputs = []

        # Store integration calls if not already stored
        if not hasattr(self.contract, "integration_calls"):
            self.contract.integration_calls = []

        # Add any new interactions that weren't already stored
        existing_llm_ids = {
            interaction.get("interaction_id")
            for interaction in self.contract.raw_outputs
            if isinstance(interaction, dict)
        }

        for interaction in self.current_flow.llm_interactions:
            if interaction.get("interaction_id") not in existing_llm_ids:
                self.contract.raw_outputs.append(interaction)

        existing_research_ids = {
            interaction.get("research_id")
            for interaction in self.contract.integration_calls
            if isinstance(interaction, dict)
        }

        for interaction in self.current_flow.research_interactions:
            if interaction.get("research_id") not in existing_research_ids:
                self.contract.integration_calls.append(interaction)

        logger.info(
            f"📄 ILLUMINATE: Stored {len(self.current_flow.llm_interactions)} LLM + "
            f"{len(self.current_flow.research_interactions)} research interactions in contract"
        )

    def _save_flow_data(self):
        """Save complete flow data to file"""
        if not self.current_flow:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"illuminate_flow_{self.current_flow.flow_id[:8]}_{timestamp}.json"
        filepath = self.output_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump(asdict(self.current_flow), f, indent=2, default=str)

            logger.info(f"💾 ILLUMINATE: Saved complete flow data: {filepath}")

            # Create summary
            self._create_flow_summary(filepath.stem)

        except Exception as e:
            logger.error(f"❌ ILLUMINATE: Failed to save flow data: {e}")

    def _create_flow_summary(self, flow_name: str):
        """Create human-readable flow summary with transparency metrics"""
        if not self.current_flow:
            return

        summary_file = self.output_dir / f"{flow_name}_TRANSPARENCY_REPORT.md"

        summary_content = f"""# OPERATION ILLUMINATE - Glass Box Transparency Report

## Flow Analysis
- **Flow ID**: {self.current_flow.flow_id}
- **Engagement ID**: {self.current_flow.engagement_id}
- **Timestamp**: {self.current_flow.timestamp}
- **Duration**: {self.current_flow.total_duration_ms}ms ({self.current_flow.total_duration_ms/1000:.2f}s)

## User Query
```
{self.current_flow.user_query}
```

## Transparency Metrics
- **Glass Box Completeness**: {self.current_flow.glass_box_completeness_score:.1%}
- **Total Cost**: ${self.current_flow.total_cost_usd:.4f}
- **LLM Interactions**: {len(self.current_flow.llm_interactions)}
- **Research Queries**: {len(self.current_flow.research_interactions)}
- **Contradictions Found**: {self.current_flow.total_contradictions_found}
- **Assumptions Challenged**: {self.current_flow.total_assumptions_challenged}

## LLM Interactions ({len(self.current_flow.llm_interactions)})
"""

        for i, interaction in enumerate(self.current_flow.llm_interactions, 1):
            summary_content += f"""
### LLM Call {i}: {interaction.get('provider', 'unknown')}/{interaction.get('model', 'unknown')}
- **Operation**: {interaction.get('operation_context', 'N/A')}
- **Tokens**: {interaction.get('tokens_used', 0)}
- **Cost**: ${interaction.get('cost_usd', 0.0):.4f}
- **Duration**: {interaction.get('duration_ms', 0)}ms
- **Confidence**: {interaction.get('confidence_score', 0.0):.2f}
- **Response Length**: {len(str(interaction.get('raw_response', '')))} chars
"""

        summary_content += f"""

## Research Interactions ({len(self.current_flow.research_interactions)})
"""

        for i, research in enumerate(self.current_flow.research_interactions, 1):
            contradictions = research.get("contradiction_detection_result", {})
            summary_content += f"""
### Research Query {i}
- **Query**: {research.get('query_sent', 'N/A')[:100]}...
- **Sources Found**: {research.get('sources_consulted_count', 0)}
- **Confidence**: {research.get('confidence_score', 0.0):.2f}
- **Cost**: ${research.get('cost_usd', 0.0):.4f}
- **Duration**: {research.get('time_taken_ms', 0)}ms
- **Contradictions**: {"Yes" if contradictions.get('detected', False) else "No"}
- **Search Mode**: {research.get('search_mode', 'N/A')}
"""

        if self.current_flow.devils_advocate_debates:
            summary_content += f"""

## Devil's Advocate Debates ({len(self.current_flow.devils_advocate_debates)})
"""
            for i, debate in enumerate(self.current_flow.devils_advocate_debates, 1):
                summary_content += f"""
### Debate {i}: {debate.challenger_name}
- **Challenge Type**: {debate.challenge_type}
- **Original Claim**: {debate.original_claim[:100]}...
- **Outcome**: {debate.outcome}
- **Confidence Change**: {debate.confidence_before:.2f} → {debate.confidence_after:.2f}
- **Evidence Presented**: {len(debate.evidence_presented)} items
"""

        summary_content += f"""

## Final Response
```json
{json.dumps(self.current_flow.final_response, indent=2)[:1000]}...
```

---
*Generated by Operation Illuminate - Glass Box Transparency System*
"""

        try:
            with open(summary_file, "w") as f:
                f.write(summary_content)
            logger.info(f"📄 ILLUMINATE: Created transparency report: {summary_file}")
        except Exception as e:
            logger.error(f"❌ Failed to create transparency report: {e}")


# Global instance and factory
_data_capture_instance: Optional[ComprehensiveDataCaptureIlluminate] = None


def get_comprehensive_data_capture_illuminate(
    contract: Any = None,
) -> ComprehensiveDataCaptureIlluminate:
    """Get or create Operation Illuminate data capture instance"""
    global _data_capture_instance
    if _data_capture_instance is None or contract is not None:
        _data_capture_instance = ComprehensiveDataCaptureIlluminate(contract=contract)
    return _data_capture_instance
