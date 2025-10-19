"""
Glass Box Recorder - Complete Stage Transparency System
Captures all outputs from all stages for complete analysis transparency
Designed to generate comprehensive output similar to true_north_result structure
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field

from src.core.unified_context_stream import UnifiedContextStream, ContextEventType, ContextEvent


@dataclass
class StageResult:
    """Individual stage result with complete metadata"""
    stage_name: str
    trace_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None
    stage_output: Dict[str, Any] = field(default_factory=dict)
    stage_metadata: Dict[str, Any] = field(default_factory=dict)
    context_engineering: Dict[str, Any] = field(default_factory=dict)
    events: List[ContextEvent] = field(default_factory=list)
    
    def complete_stage(self, output: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Mark stage as complete with output"""
        self.completed_at = datetime.now(timezone.utc)
        self.processing_time_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
        self.stage_output = output
        if metadata:
            self.stage_metadata.update(metadata)


class GlassBoxRecorder:
    """
    Glass Box Recorder for complete transparency of analysis stages
    
    This class captures all intermediate outputs and processing stages to provide
    complete visibility into the analysis pipeline similar to true_north_result format.
    """
    
    def __init__(self, context_stream: UnifiedContextStream, initial_query: str):
        self.context_stream = context_stream
        self.trace_id = context_stream.trace_id
        self.initial_query = initial_query
        self.started_at = datetime.now(timezone.utc)
        self.last_updated = self.started_at
        
        # Stage tracking
        self.stages: Dict[str, StageResult] = {}
        self.current_stage: Optional[str] = None
        self.last_completed_stage: Optional[str] = None
        
        # Final result structure
        self.final_result = {
            "trace_id": self.trace_id,
            "initial_query": initial_query,
            "started_at": self.started_at.isoformat(),
            "user_id": None,
            "session_id": None,
            "analysis_type": "stateful_pipeline",
        }
        
        # Glass box metadata
        self.glass_box_metadata = {
            "status": "in_progress",
            "stages_completed": 0,
            "total_processing_time_ms": 0,
            "llm_provider": "grok-4-fast",
            "glass_box_enabled": True,
        }
        
    def start_stage(self, stage_name: str, context: Optional[Dict[str, Any]] = None) -> StageResult:
        """Start recording a new analysis stage"""
        stage_result = StageResult(
            stage_name=stage_name,
            trace_id=self.trace_id,
            started_at=datetime.now(timezone.utc),
        )
        
        if context:
            stage_result.stage_metadata.update(context)
            
        self.stages[stage_name] = stage_result
        self.current_stage = stage_name
        
        # Record stage start event
        self.context_stream.add_event(
            ContextEventType.PIPELINE_STAGE_STARTED,
            {
                "stage_name": stage_name,
                "context": context or {},
                "trace_id": self.trace_id,
            },
            {
                "glass_box_stage_start": True,
                "stage_sequence": len(self.stages),
            }
        )
        
        return stage_result
        
    def complete_stage(self, stage_name: str, output: Dict[str, Any], 
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Complete a stage with its output"""
        if stage_name not in self.stages:
            raise ValueError(f"Stage {stage_name} not found. Call start_stage first.")
            
        stage = self.stages[stage_name]
        stage.complete_stage(output, metadata)
        
        # Update tracking
        self.last_completed_stage = stage_name
        self.last_updated = datetime.now(timezone.utc)
        self.glass_box_metadata["stages_completed"] += 1
        self.glass_box_metadata["total_processing_time_ms"] += stage.processing_time_ms or 0
        
        # Add stage result to final result
        self.final_result[stage_name] = {
            **output,
            "_stage_metadata": {
                "stage": stage_name,
                "processing_time_ms": stage.processing_time_ms,
                "completed_at": stage.completed_at.isoformat(),
                "context_engineering": stage.context_engineering,
            }
        }
        
        # Record stage completion event
        self.context_stream.add_event(
            ContextEventType.PIPELINE_STAGE_COMPLETED,
            {
                "stage_name": stage_name,
                "output_size_chars": len(json.dumps(output, default=str)),
                "processing_time_ms": stage.processing_time_ms,
                "trace_id": self.trace_id,
            },
            {
                "glass_box_stage_complete": True,
                "stage_sequence": self.glass_box_metadata["stages_completed"],
                "output_captured": True,
            }
        )
        
        self.current_stage = None
        
    def add_stage_event(self, event_data: Dict[str, Any], event_type: ContextEventType = ContextEventType.REASONING_STEP) -> None:
        """Add an event to the current stage"""
        if not self.current_stage:
            return
            
        event = self.context_stream.add_event(
            event_type,
            {
                **event_data,
                "stage": self.current_stage,
                "trace_id": self.trace_id,
            },
            {
                "glass_box_event": True,
                "stage_name": self.current_stage,
            }
        )
        
        # Add to stage events
        if self.current_stage in self.stages:
            self.stages[self.current_stage].events.append(event)
            
    def capture_llm_interaction(self, provider: str, request: Dict[str, Any],
                              response: Dict[str, Any], stage: Optional[str] = None) -> None:
        """Capture LLM request/response for Glass Box transparency"""
        target_stage = stage or self.current_stage

        # Capture request
        self.context_stream.add_event(
            ContextEventType.LLM_PROVIDER_REQUEST,
            {
                "provider": provider,
                "request": request,
                "stage": target_stage,
                "trace_id": self.trace_id,
            },
            {
                "glass_box_llm_request": True,
                "stage_name": target_stage,
            }
        )

        # Capture response with required schema fields
        response_data = {
            "provider": provider,
            "response": response,
            "stage": target_stage,
            "trace_id": self.trace_id,
        }

        # Extract model, tokens_used, latency_ms if available in response
        if isinstance(response, dict):
            if "model" in response:
                response_data["model"] = response["model"]
            if "tokens_used" in response:
                response_data["tokens_used"] = response["tokens_used"]
            if "latency_ms" in response:
                response_data["latency_ms"] = response["latency_ms"]
            elif "response_time_ms" in response:
                response_data["latency_ms"] = response["response_time_ms"]

        self.context_stream.add_event(
            ContextEventType.LLM_PROVIDER_RESPONSE,
            response_data,
            {
                "glass_box_llm_response": True,
                "stage_name": target_stage,
            }
        )
        
    def capture_consultant_selection(self, consultants: List[Dict[str, Any]], 
                                   selection_rationale: str, confidence: float) -> None:
        """Capture consultant selection details"""
        selection_data = {
            "consultants": consultants,
            "selection_rationale": selection_rationale,
            "total_confidence": confidence,
            "consultant_count": len(consultants),
            "trace_id": self.trace_id,
        }
        
        self.context_stream.add_event(
            ContextEventType.CONSULTANT_SELECTION_COMPLETE,
            selection_data,
            {
                "glass_box_consultant_selection": True,
                "stage_name": self.current_stage,
            }
        )
        
    def capture_research_grounding(self, research_queries: List[str], 
                                 research_results: List[Dict[str, Any]]) -> None:
        """Capture research grounding process"""
        self.context_stream.add_event(
            ContextEventType.RESEARCH_GROUNDING_START,
            {
                "queries": research_queries,
                "query_count": len(research_queries),
                "trace_id": self.trace_id,
            },
            {
                "glass_box_research_start": True,
                "stage_name": self.current_stage,
            }
        )
        
        self.context_stream.add_event(
            ContextEventType.RESEARCH_GROUNDING_COMPLETE,
            {
                "results": research_results,
                "result_count": len(research_results),
                "trace_id": self.trace_id,
            },
            {
                "glass_box_research_complete": True,
                "stage_name": self.current_stage,
            }
        )
        
    def generate_complete_glass_box_result(self) -> Dict[str, Any]:
        """Generate complete Glass Box result similar to true_north_result format"""
        
        # Update final metadata
        self.glass_box_metadata.update({
            "status": "completed",
            "last_completed_stage": self.last_completed_stage,
            "last_updated": self.last_updated.isoformat(),
            "total_stages": len(self.stages),
        })
        
        # Build complete result structure
        complete_result = {
            "status": "completed",
            "trace_id": self.trace_id,
            "final_result": self.final_result,
            "glass_box_metadata": self.glass_box_metadata,
            "stages_detail": {},
            "event_timeline": [],
            "context_stream_summary": self.context_stream.get_evidence_summary(),
        }
        
        # Add detailed stage information
        for stage_name, stage in self.stages.items():
            complete_result["stages_detail"][stage_name] = {
                "trace_id": stage.trace_id,
                "started_at": stage.started_at.isoformat(),
                "completed_at": stage.completed_at.isoformat() if stage.completed_at else None,
                "processing_time_ms": stage.processing_time_ms,
                "stage_output": stage.stage_output,
                "stage_metadata": stage.stage_metadata,
                "context_engineering": stage.context_engineering,
                "events_count": len(stage.events),
            }
            
        # Add event timeline for Glass Box transparency
        all_events = self.context_stream.get_events()
        for event in all_events:
            if event.metadata.get("glass_box_event") or event.metadata.get("glass_box_stage_start") or event.metadata.get("glass_box_stage_complete"):
                complete_result["event_timeline"].append({
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "event_id": event.event_id,
                    "stage": event.data.get("stage"),
                    "summary": self._summarize_event(event),
                })
                
        return complete_result
        
    def _summarize_event(self, event: ContextEvent) -> str:
        """Create summary of event for timeline"""
        event_type = event.event_type.value
        stage = event.data.get("stage", "unknown")
        
        if event_type == "pipeline_stage_started":
            return f"Started stage: {stage}"
        elif event_type == "pipeline_stage_completed":
            processing_time = event.data.get("processing_time_ms", 0)
            return f"Completed stage: {stage} ({processing_time}ms)"
        elif event_type == "llm_provider_request":
            provider = event.data.get("provider", "unknown")
            return f"LLM request to {provider} in stage {stage}"
        elif event_type == "llm_provider_response":
            provider = event.data.get("provider", "unknown")
            return f"LLM response from {provider} in stage {stage}"
        elif event_type == "consultant_selection_complete":
            count = event.data.get("consultant_count", 0)
            confidence = event.data.get("total_confidence", 0)
            return f"Selected {count} consultants with {confidence:.1%} confidence"
        elif event_type == "research_grounding_start":
            count = event.data.get("query_count", 0)
            return f"Started research with {count} queries"
        elif event_type == "research_grounding_complete":
            count = event.data.get("result_count", 0)
            return f"Completed research with {count} results"
        else:
            return f"{event_type} in stage {stage}"
            
    def export_for_user(self, filename: Optional[str] = None) -> str:
        """Export Glass Box result to file for user verification"""
        complete_result = self.generate_complete_glass_box_result()
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/Users/marcin/lolla_v1_release/glass_box_result_{timestamp}.json"
            
        with open(filename, 'w') as f:
            json.dump(complete_result, f, indent=2, default=str)
            
        return filename
        
    def get_stage_output(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get output from a specific stage"""
        if stage_name in self.stages:
            return self.stages[stage_name].stage_output
        return None
        
    def get_current_stage_name(self) -> Optional[str]:
        """Get current stage name"""
        return self.current_stage
        
    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage is completed"""
        return stage_name in self.stages and self.stages[stage_name].completed_at is not None


def create_glass_box_recorder(context_stream: UnifiedContextStream, initial_query: str) -> GlassBoxRecorder:
    """Factory function to create Glass Box recorder"""
    return GlassBoxRecorder(context_stream, initial_query)