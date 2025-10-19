"""
Comprehensive Data Capture System
Captures all data flowing through METIS pipeline for analysis and verification
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class LLMInteraction:
    """Captures LLM request/response data"""

    interaction_id: str
    timestamp: str
    provider: str
    model: str
    request_data: Dict[str, Any]
    response_data: Dict[str, Any]
    tokens_used: int
    cost_usd: float
    duration_ms: int


@dataclass
class ResearchCapture:
    """Captures research data and findings"""

    research_id: str
    timestamp: str
    query: str
    context: Dict[str, Any]
    mode: str
    sources: List[Dict[str, Any]]
    confidence: float
    summary: str
    bullets: List[str]
    facts_extracted: List[str]
    contradictions: List[str]
    cost_usd: float
    duration_ms: int


@dataclass
class DatabaseOperation:
    """Captures database query/result data"""

    operation_id: str
    timestamp: str
    operation_type: str  # query, insert, update, delete
    table_name: str
    query_content: str
    parameters: Dict[str, Any]
    result_data: Any
    rows_affected: int
    duration_ms: int


@dataclass
class CognitiveProcessing:
    """Captures cognitive engine processing data"""

    processing_id: str
    timestamp: str
    input_query: str
    models_selected: List[Dict[str, Any]]
    model_applications: List[Dict[str, Any]]
    reasoning_chains: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    confidence_scores: Dict[str, float]
    final_output: Dict[str, Any]
    duration_ms: int


@dataclass
class SystemFlow:
    """Captures complete system data flow"""

    flow_id: str
    timestamp: str
    user_query: str
    llm_interactions: List[LLMInteraction]
    research_data: List[ResearchCapture]
    database_operations: List[DatabaseOperation]
    cognitive_processing: List[CognitiveProcessing]
    final_response: Dict[str, Any]
    total_duration_ms: int
    total_cost_usd: float


class ComprehensiveDataCapture:
    """
    Comprehensive data capture system for METIS pipeline
    Captures ALL data flowing through the system for analysis
    """

    def __init__(self, output_dir: str = "data_captures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Active captures
        self.current_flow: Optional[SystemFlow] = None
        self.capture_active = False

        # Storage
        self.llm_interactions: List[LLMInteraction] = []
        self.research_captures: List[ResearchCapture] = []
        self.database_operations: List[DatabaseOperation] = []
        self.cognitive_processing: List[CognitiveProcessing] = []

        logger.info(
            f"✅ Comprehensive Data Capture initialized - output: {self.output_dir}"
        )

    @contextmanager
    def capture_system_flow(self, user_query: str):
        """Context manager for capturing complete system flow"""
        flow_id = str(uuid.uuid4())
        start_time = time.time()

        self.current_flow = SystemFlow(
            flow_id=flow_id,
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            llm_interactions=[],
            research_data=[],
            database_operations=[],
            cognitive_processing=[],
            final_response={},
            total_duration_ms=0,
            total_cost_usd=0.0,
        )

        self.capture_active = True
        logger.info(f"🎬 Started data capture for flow: {flow_id}")

        try:
            yield self
        finally:
            # Finalize capture
            end_time = time.time()
            self.current_flow.total_duration_ms = int((end_time - start_time) * 1000)
            self.current_flow.total_cost_usd = sum(
                interaction.cost_usd
                for interaction in self.current_flow.llm_interactions
            ) + sum(research.cost_usd for research in self.current_flow.research_data)

            # Save complete flow
            self._save_flow_data()
            self.capture_active = False
            logger.info(f"🎬 Completed data capture for flow: {flow_id}")

    def capture_llm_interaction(
        self,
        provider: str,
        model: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
    ):
        """Capture LLM interaction data"""
        if not self.capture_active:
            return

        interaction = LLMInteraction(
            interaction_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            request_data=request_data,
            response_data=response_data,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )

        self.current_flow.llm_interactions.append(interaction)
        logger.info(
            f"📝 Captured LLM interaction: {provider}/{model} - {tokens_used} tokens, ${cost_usd:.4f}"
        )

    def capture_research_data(
        self,
        query: str,
        context: Dict[str, Any],
        mode: str,
        sources: List[Dict[str, Any]],
        confidence: float,
        summary: str,
        bullets: List[str],
        facts_extracted: List[str] = None,
        contradictions: List[str] = None,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
    ):
        """Capture research data and findings"""
        if not self.capture_active:
            return

        research = ResearchCapture(
            research_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            query=query,
            context=context,
            mode=mode,
            sources=sources,
            confidence=confidence,
            summary=summary,
            bullets=bullets,
            facts_extracted=facts_extracted or [],
            contradictions=contradictions or [],
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )

        self.current_flow.research_data.append(research)
        logger.info(
            f"🔍 Captured research data: {len(sources)} sources, confidence={confidence:.2f}"
        )

    def capture_database_operation(
        self,
        operation_type: str,
        table_name: str,
        query_content: str,
        parameters: Dict[str, Any],
        result_data: Any,
        rows_affected: int = 0,
        duration_ms: int = 0,
    ):
        """Capture database operation data"""
        if not self.capture_active:
            return

        operation = DatabaseOperation(
            operation_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            operation_type=operation_type,
            table_name=table_name,
            query_content=query_content,
            parameters=parameters,
            result_data=result_data,
            rows_affected=rows_affected,
            duration_ms=duration_ms,
        )

        self.current_flow.database_operations.append(operation)
        logger.info(
            f"💾 Captured DB operation: {operation_type} on {table_name} - {rows_affected} rows"
        )

    def capture_cognitive_processing(
        self,
        input_query: str,
        models_selected: List[Dict[str, Any]],
        model_applications: List[Dict[str, Any]],
        reasoning_chains: List[Dict[str, Any]],
        validation_results: Dict[str, Any],
        confidence_scores: Dict[str, float],
        final_output: Dict[str, Any],
        duration_ms: int = 0,
    ):
        """Capture cognitive processing data"""
        if not self.capture_active:
            return

        processing = CognitiveProcessing(
            processing_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            input_query=input_query,
            models_selected=models_selected,
            model_applications=model_applications,
            reasoning_chains=reasoning_chains,
            validation_results=validation_results,
            confidence_scores=confidence_scores,
            final_output=final_output,
            duration_ms=duration_ms,
        )

        self.current_flow.cognitive_processing.append(processing)
        logger.info(
            f"🧠 Captured cognitive processing: {len(models_selected)} models applied"
        )

    def set_final_response(self, response: Dict[str, Any]):
        """Set the final response for the current flow"""
        if self.capture_active and self.current_flow:
            self.current_flow.final_response = response
            logger.info("📤 Captured final response")

    def _save_flow_data(self):
        """Save complete flow data to file"""
        if not self.current_flow:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"system_flow_{self.current_flow.flow_id[:8]}_{timestamp}.json"
        filepath = self.output_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump(asdict(self.current_flow), f, indent=2, default=str)

            logger.info(f"💾 Saved complete flow data: {filepath}")

            # Also create summary file
            self._create_flow_summary(filepath.stem)

        except Exception as e:
            logger.error(f"❌ Failed to save flow data: {e}")

    def _create_flow_summary(self, flow_name: str):
        """Create human-readable flow summary"""
        if not self.current_flow:
            return

        summary_file = self.output_dir / f"{flow_name}_SUMMARY.md"

        summary_content = f"""# METIS System Flow Analysis
## Flow ID: {self.current_flow.flow_id}
## Timestamp: {self.current_flow.timestamp}

### User Query
```
{self.current_flow.user_query}
```

### LLM Interactions ({len(self.current_flow.llm_interactions)})
"""

        for i, interaction in enumerate(self.current_flow.llm_interactions, 1):
            summary_content += f"""
#### Interaction {i}: {interaction.provider}/{interaction.model}
- **Tokens**: {interaction.tokens_used}
- **Cost**: ${interaction.cost_usd:.4f}
- **Duration**: {interaction.duration_ms}ms
- **Request**: {json.dumps(interaction.request_data, indent=2)[:500]}...
- **Response**: {json.dumps(interaction.response_data, indent=2)[:500]}...
"""

        summary_content += f"""

### Research Data ({len(self.current_flow.research_data)})
"""

        for i, research in enumerate(self.current_flow.research_data, 1):
            summary_content += f"""
#### Research {i}: {research.mode} mode
- **Query**: {research.query}
- **Confidence**: {research.confidence:.2f}
- **Sources**: {len(research.sources)}
- **Cost**: ${research.cost_usd:.4f}
- **Duration**: {research.duration_ms}ms
- **Summary**: {research.summary}
- **Facts**: {research.facts_extracted}
"""

        summary_content += f"""

### Database Operations ({len(self.current_flow.database_operations)})
"""

        for i, operation in enumerate(self.current_flow.database_operations, 1):
            summary_content += f"""
#### Operation {i}: {operation.operation_type}
- **Table**: {operation.table_name}
- **Query**: {operation.query_content}
- **Rows**: {operation.rows_affected}
- **Duration**: {operation.duration_ms}ms
"""

        summary_content += f"""

### Cognitive Processing ({len(self.current_flow.cognitive_processing)})
"""

        for i, processing in enumerate(self.current_flow.cognitive_processing, 1):
            summary_content += f"""
#### Processing {i}
- **Models Selected**: {len(processing.models_selected)}
- **Model Applications**: {len(processing.model_applications)}
- **Confidence Scores**: {processing.confidence_scores}
- **Duration**: {processing.duration_ms}ms
"""

        summary_content += f"""

### Performance Summary
- **Total Duration**: {self.current_flow.total_duration_ms}ms ({self.current_flow.total_duration_ms/1000:.2f}s)
- **Total Cost**: ${self.current_flow.total_cost_usd:.4f}
- **LLM Calls**: {len(self.current_flow.llm_interactions)}
- **Research Queries**: {len(self.current_flow.research_data)}
- **DB Operations**: {len(self.current_flow.database_operations)}

### Final Response
```json
{json.dumps(self.current_flow.final_response, indent=2)}
```
"""

        try:
            with open(summary_file, "w") as f:
                f.write(summary_content)
            logger.info(f"📄 Created flow summary: {summary_file}")
        except Exception as e:
            logger.error(f"❌ Failed to create flow summary: {e}")


# Global instance
_data_capture_instance: Optional[ComprehensiveDataCapture] = None


def get_data_capture() -> ComprehensiveDataCapture:
    """Get or create global data capture instance"""
    global _data_capture_instance
    if _data_capture_instance is None:
        _data_capture_instance = ComprehensiveDataCapture()
    return _data_capture_instance
