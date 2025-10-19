"""
Cognitive Tracer
Comprehensive decision audit trail system for cognitive orchestration
Provides full transparency and traceability of all cognitive decisions
"""

import uuid
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
import logging


class DecisionType(Enum):
    """Types of cognitive decisions that can be traced"""

    MODEL_SELECTION = "model_selection"
    STRATEGY_SELECTION = "strategy_selection"
    QUALITY_VALIDATION = "quality_validation"
    RESULT_SYNTHESIS = "result_synthesis"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RETRY_DECISION = "retry_decision"


class DecisionLevel(Enum):
    """Importance levels for cognitive decisions"""

    CRITICAL = "critical"  # Core orchestration decisions
    IMPORTANT = "important"  # Significant model/quality decisions
    INFORMATIONAL = "info"  # Supporting information and metrics
    DEBUG = "debug"  # Detailed technical information


@dataclass
class DecisionContext:
    """Context information for a cognitive decision"""

    orchestration_id: str
    problem_statement: str
    business_context: Dict[str, Any]
    quality_requirements: Dict[str, float]
    strategy: str
    timestamp: datetime
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class DecisionTrace:
    """Individual decision trace entry"""

    trace_id: str
    decision_type: DecisionType
    level: DecisionLevel
    timestamp: datetime
    decision_maker: str  # Component that made the decision
    decision_description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    reasoning: str
    confidence: float
    processing_time_ms: float
    context: DecisionContext
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization"""
        return {
            "trace_id": self.trace_id,
            "decision_type": self.decision_type.value,
            "level": self.level.value,
            "timestamp": self.timestamp.isoformat(),
            "decision_maker": self.decision_maker,
            "decision_description": self.decision_description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "context": {
                "orchestration_id": self.context.orchestration_id,
                "problem_statement": (
                    self.context.problem_statement[:200] + "..."
                    if len(self.context.problem_statement) > 200
                    else self.context.problem_statement
                ),
                "business_context": self.context.business_context,
                "quality_requirements": self.context.quality_requirements,
                "strategy": self.context.strategy,
                "timestamp": self.context.timestamp.isoformat(),
                "session_id": self.context.session_id,
                "user_id": self.context.user_id,
            },
            "metadata": self.metadata,
        }


@dataclass
class CognitiveAuditTrail:
    """Complete audit trail for a cognitive orchestration"""

    orchestration_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_processing_time_ms: float
    traces: List[DecisionTrace]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit trail to dictionary"""
        return {
            "orchestration_id": self.orchestration_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_processing_time_ms": self.total_processing_time_ms,
            "traces": [trace.to_dict() for trace in self.traces],
            "summary": self.summary,
        }


class CognitiveTracer:
    """
    Comprehensive cognitive decision tracer

    Tracks all decisions made during cognitive orchestration for full transparency
    and audit trail capabilities. Enables debugging, optimization, and compliance.
    """

    def __init__(self, enable_detailed_logging: bool = True):
        """
        Initialize cognitive tracer

        Args:
            enable_detailed_logging: Whether to enable detailed debug logging
        """
        self.enable_detailed_logging = enable_detailed_logging
        self.logger = logging.getLogger(__name__)

        # Active orchestration contexts
        self._active_contexts: Dict[str, DecisionContext] = {}

        # Active audit trails
        self._active_trails: Dict[str, List[DecisionTrace]] = {}

        # Completed audit trails for analysis
        self._completed_trails: List[CognitiveAuditTrail] = []

        # Performance metrics
        self._trace_count = 0
        self._total_decisions_traced = 0

        self.logger.info(
            "🔍 CognitiveTracer initialized with detailed logging: %s",
            enable_detailed_logging,
        )

    def start_orchestration(
        self,
        orchestration_id: str,
        problem_statement: str,
        business_context: Dict[str, Any],
        quality_requirements: Dict[str, float],
        strategy: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> DecisionContext:
        """
        Start tracing a new cognitive orchestration

        Args:
            orchestration_id: Unique identifier for this orchestration
            problem_statement: The problem being analyzed
            business_context: Business context for the analysis
            quality_requirements: Quality requirements and thresholds
            strategy: Orchestration strategy being used
            session_id: Optional session identifier
            user_id: Optional user identifier

        Returns:
            DecisionContext for this orchestration
        """
        context = DecisionContext(
            orchestration_id=orchestration_id,
            problem_statement=problem_statement,
            business_context=business_context,
            quality_requirements=quality_requirements,
            strategy=strategy,
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            user_id=user_id,
        )

        self._active_contexts[orchestration_id] = context
        self._active_trails[orchestration_id] = []

        self.logger.info(
            f"🚀 Started tracing orchestration {orchestration_id[:8]}... with strategy: {strategy}"
        )
        return context

    def trace_decision(
        self,
        orchestration_id: str,
        decision_type: DecisionType,
        level: DecisionLevel,
        decision_maker: str,
        decision_description: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reasoning: str,
        confidence: float,
        processing_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Trace a cognitive decision

        Args:
            orchestration_id: ID of the orchestration this decision belongs to
            decision_type: Type of decision being made
            level: Importance level of this decision
            decision_maker: Component or system making the decision
            decision_description: Human-readable description
            inputs: Input data that influenced the decision
            outputs: Results/outputs of the decision
            reasoning: Explanation of why this decision was made
            confidence: Confidence score (0.0-1.0) in this decision
            processing_time_ms: Time taken to make this decision
            metadata: Additional metadata

        Returns:
            Unique trace ID for this decision
        """
        if orchestration_id not in self._active_contexts:
            self.logger.warning(
                f"⚠️ Attempted to trace decision for unknown orchestration: {orchestration_id}"
            )
            return str(uuid.uuid4())

        trace_id = str(uuid.uuid4())
        context = self._active_contexts[orchestration_id]

        trace = DecisionTrace(
            trace_id=trace_id,
            decision_type=decision_type,
            level=level,
            timestamp=datetime.now(timezone.utc),
            decision_maker=decision_maker,
            decision_description=decision_description,
            inputs=self._sanitize_data(inputs),
            outputs=self._sanitize_data(outputs),
            reasoning=reasoning,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            context=context,
            metadata=metadata or {},
        )

        self._active_trails[orchestration_id].append(trace)
        self._trace_count += 1
        self._total_decisions_traced += 1

        # Log based on level
        if level == DecisionLevel.CRITICAL:
            self.logger.info(
                f"🎯 CRITICAL: {decision_maker} - {decision_description} (confidence: {confidence:.3f})"
            )
        elif level == DecisionLevel.IMPORTANT:
            self.logger.info(
                f"📋 IMPORTANT: {decision_maker} - {decision_description} (confidence: {confidence:.3f})"
            )
        elif self.enable_detailed_logging:
            if level == DecisionLevel.INFORMATIONAL:
                self.logger.debug(f"ℹ️ INFO: {decision_maker} - {decision_description}")
            elif level == DecisionLevel.DEBUG:
                self.logger.debug(
                    f"🔧 DEBUG: {decision_maker} - {decision_description}"
                )

        return trace_id

    def trace_model_selection(
        self,
        orchestration_id: str,
        available_models: List[str],
        selected_models: List[str],
        selection_criteria: Dict[str, Any],
        strategy_applied: str,
        confidence: float,
        processing_time_ms: float,
    ) -> str:
        """Convenience method for tracing model selection decisions"""

        return self.trace_decision(
            orchestration_id=orchestration_id,
            decision_type=DecisionType.MODEL_SELECTION,
            level=DecisionLevel.CRITICAL,
            decision_maker="CognitiveOrchestrator",
            decision_description=f"Selected {len(selected_models)} models from {len(available_models)} available",
            inputs={
                "available_models": available_models,
                "selection_criteria": selection_criteria,
                "strategy": strategy_applied,
            },
            outputs={
                "selected_models": selected_models,
                "selection_rationale": f"Applied {strategy_applied} strategy",
            },
            reasoning=f"Selected models based on {strategy_applied} strategy considering {', '.join(selection_criteria.keys())}",
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            metadata={"model_count": len(selected_models)},
        )

    def trace_quality_validation(
        self,
        orchestration_id: str,
        content: str,
        quality_scores: Dict[str, float],
        overall_score: float,
        passed: bool,
        issues_found: List[str],
        processing_time_ms: float,
    ) -> str:
        """Convenience method for tracing quality validation decisions"""

        return self.trace_decision(
            orchestration_id=orchestration_id,
            decision_type=DecisionType.QUALITY_VALIDATION,
            level=DecisionLevel.IMPORTANT,
            decision_maker="QualityValidator",
            decision_description=f"Quality validation {'PASSED' if passed else 'FAILED'} with score {overall_score:.3f}",
            inputs={
                "content_length": len(content),
                "validation_criteria": list(quality_scores.keys()),
            },
            outputs={
                "quality_scores": quality_scores,
                "overall_score": overall_score,
                "passed": passed,
                "issues_count": len(issues_found),
            },
            reasoning=f"Evaluated content across {len(quality_scores)} dimensions, {'passed' if passed else 'failed'} with {len(issues_found)} issues",
            confidence=overall_score,
            processing_time_ms=processing_time_ms,
            metadata={
                "content_length": len(content),
                "issues": issues_found[:3],
            },  # Limit issues in metadata
        )

    def trace_strategy_selection(
        self,
        orchestration_id: str,
        problem_complexity: str,
        available_strategies: List[str],
        selected_strategy: str,
        selection_reasoning: str,
        confidence: float,
    ) -> str:
        """Convenience method for tracing strategy selection decisions"""

        return self.trace_decision(
            orchestration_id=orchestration_id,
            decision_type=DecisionType.STRATEGY_SELECTION,
            level=DecisionLevel.CRITICAL,
            decision_maker="CognitiveOrchestrator",
            decision_description=f"Selected {selected_strategy} strategy for {problem_complexity} complexity problem",
            inputs={
                "problem_complexity": problem_complexity,
                "available_strategies": available_strategies,
            },
            outputs={
                "selected_strategy": selected_strategy,
                "reasoning": selection_reasoning,
            },
            reasoning=selection_reasoning,
            confidence=confidence,
            processing_time_ms=0.0,
            metadata={"complexity": problem_complexity},
        )

    def trace_error_handling(
        self,
        orchestration_id: str,
        error_type: str,
        error_message: str,
        component: str,
        recovery_action: str,
        recovery_successful: bool,
    ) -> str:
        """Convenience method for tracing error handling decisions"""

        return self.trace_decision(
            orchestration_id=orchestration_id,
            decision_type=DecisionType.ERROR_HANDLING,
            level=DecisionLevel.CRITICAL,
            decision_maker=component,
            decision_description=f"Error handled: {error_type} - Recovery {'successful' if recovery_successful else 'failed'}",
            inputs={
                "error_type": error_type,
                "error_message": error_message[:200],  # Truncate long error messages
                "component": component,
            },
            outputs={
                "recovery_action": recovery_action,
                "recovery_successful": recovery_successful,
            },
            reasoning=f"Encountered {error_type} in {component}, attempted {recovery_action}",
            confidence=1.0 if recovery_successful else 0.0,
            processing_time_ms=0.0,
            metadata={"error_type": error_type, "component": component},
        )

    def end_orchestration(
        self,
        orchestration_id: str,
        success: bool,
        final_confidence: float,
        total_processing_time_ms: float,
        models_applied: List[str],
        quality_passed: bool,
    ) -> CognitiveAuditTrail:
        """
        End tracing for an orchestration and create final audit trail

        Args:
            orchestration_id: ID of the orchestration to end
            success: Whether the orchestration was successful
            final_confidence: Final confidence score of the result
            total_processing_time_ms: Total time for the orchestration
            models_applied: List of models that were applied
            quality_passed: Whether quality validation passed

        Returns:
            Complete audit trail for the orchestration
        """
        if orchestration_id not in self._active_contexts:
            self.logger.warning(
                f"⚠️ Attempted to end unknown orchestration: {orchestration_id}"
            )
            return None

        context = self._active_contexts[orchestration_id]
        traces = self._active_trails[orchestration_id]

        # Create summary
        summary = self._create_audit_summary(
            traces, success, final_confidence, models_applied, quality_passed
        )

        # Create audit trail
        audit_trail = CognitiveAuditTrail(
            orchestration_id=orchestration_id,
            start_time=context.timestamp,
            end_time=datetime.now(timezone.utc),
            total_processing_time_ms=total_processing_time_ms,
            traces=traces,
            summary=summary,
        )

        # Store completed trail
        self._completed_trails.append(audit_trail)

        # Clean up active tracking
        del self._active_contexts[orchestration_id]
        del self._active_trails[orchestration_id]

        # Keep only last 100 completed trails for memory management
        if len(self._completed_trails) > 100:
            self._completed_trails = self._completed_trails[-100:]

        self.logger.info(
            f"✅ Orchestration {orchestration_id[:8]}... completed: "
            f"{'SUCCESS' if success else 'FAILED'}, {len(traces)} decisions traced, "
            f"{total_processing_time_ms:.1f}ms total"
        )

        return audit_trail

    def get_audit_trail(self, orchestration_id: str) -> Optional[CognitiveAuditTrail]:
        """Get completed audit trail by orchestration ID"""
        for trail in self._completed_trails:
            if trail.orchestration_id == orchestration_id:
                return trail
        return None

    def get_recent_audit_trails(self, limit: int = 10) -> List[CognitiveAuditTrail]:
        """Get most recent completed audit trails"""
        return self._completed_trails[-limit:]

    def export_audit_trail(
        self, orchestration_id: str, format: str = "json"
    ) -> Optional[str]:
        """
        Export audit trail in specified format

        Args:
            orchestration_id: ID of orchestration to export
            format: Export format ("json", "summary", "detailed")

        Returns:
            Formatted audit trail string
        """
        trail = self.get_audit_trail(orchestration_id)
        if not trail:
            return None

        if format == "json":
            return json.dumps(trail.to_dict(), indent=2)
        elif format == "summary":
            return self._create_summary_export(trail)
        elif format == "detailed":
            return self._create_detailed_export(trail)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_tracing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracing statistics"""

        decision_type_counts = {}
        level_counts = {}
        total_traces = 0

        for trail in self._completed_trails:
            for trace in trail.traces:
                total_traces += 1
                decision_type_counts[trace.decision_type.value] = (
                    decision_type_counts.get(trace.decision_type.value, 0) + 1
                )
                level_counts[trace.level.value] = (
                    level_counts.get(trace.level.value, 0) + 1
                )

        return {
            "total_orchestrations": len(self._completed_trails),
            "active_orchestrations": len(self._active_contexts),
            "total_decisions_traced": self._total_decisions_traced,
            "decisions_by_type": decision_type_counts,
            "decisions_by_level": level_counts,
            "average_decisions_per_orchestration": total_traces
            / max(len(self._completed_trails), 1),
            "tracing_enabled": True,
            "detailed_logging": self.enable_detailed_logging,
        }

    def _sanitize_data(self, data: Any) -> Any:
        """Sanitize data for tracing (remove sensitive info, limit size)"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(
                    sensitive in key.lower()
                    for sensitive in ["password", "token", "secret", "key"]
                ):
                    sanitized[key] = "[REDACTED]"
                elif isinstance(value, str) and len(value) > 1000:
                    sanitized[key] = value[:1000] + "... [TRUNCATED]"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        elif isinstance(data, list) and len(data) > 100:
            return data[:100] + ["... [TRUNCATED]"]
        elif isinstance(data, str) and len(data) > 1000:
            return data[:1000] + "... [TRUNCATED]"
        else:
            return data

    def _create_audit_summary(
        self,
        traces: List[DecisionTrace],
        success: bool,
        final_confidence: float,
        models_applied: List[str],
        quality_passed: bool,
    ) -> Dict[str, Any]:
        """Create summary of the audit trail"""

        decision_counts = {}
        level_counts = {}
        total_processing_time = 0.0
        confidence_scores = []

        for trace in traces:
            decision_counts[trace.decision_type.value] = (
                decision_counts.get(trace.decision_type.value, 0) + 1
            )
            level_counts[trace.level.value] = level_counts.get(trace.level.value, 0) + 1
            total_processing_time += trace.processing_time_ms
            confidence_scores.append(trace.confidence)

        return {
            "success": success,
            "final_confidence": final_confidence,
            "quality_passed": quality_passed,
            "models_applied": models_applied,
            "total_decisions": len(traces),
            "decisions_by_type": decision_counts,
            "decisions_by_level": level_counts,
            "average_decision_confidence": (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else 0.0
            ),
            "total_decision_processing_time_ms": total_processing_time,
            "critical_decisions": level_counts.get("critical", 0),
            "important_decisions": level_counts.get("important", 0),
        }

    def _create_summary_export(self, trail: CognitiveAuditTrail) -> str:
        """Create summary export format"""
        lines = [
            "COGNITIVE ORCHESTRATION AUDIT TRAIL",
            "=====================================",
            f"Orchestration ID: {trail.orchestration_id}",
            f"Start Time: {trail.start_time.isoformat()}",
            f"End Time: {trail.end_time.isoformat() if trail.end_time else 'In Progress'}",
            f"Total Time: {trail.total_processing_time_ms:.1f}ms",
            f"Success: {trail.summary['success']}",
            f"Final Confidence: {trail.summary['final_confidence']:.3f}",
            f"Quality Passed: {trail.summary['quality_passed']}",
            f"Models Applied: {', '.join(trail.summary['models_applied'])}",
            "",
            "DECISION SUMMARY:",
            f"Total Decisions: {trail.summary['total_decisions']}",
            f"Critical: {trail.summary['critical_decisions']}",
            f"Important: {trail.summary['important_decisions']}",
            f"Average Confidence: {trail.summary['average_decision_confidence']:.3f}",
            "",
            "DECISIONS BY TYPE:",
        ]

        for decision_type, count in trail.summary["decisions_by_type"].items():
            lines.append(f"  {decision_type}: {count}")

        return "\n".join(lines)

    def _create_detailed_export(self, trail: CognitiveAuditTrail) -> str:
        """Create detailed export format"""
        lines = [self._create_summary_export(trail), "", "DETAILED DECISION TRACE:", ""]

        for i, trace in enumerate(trail.traces, 1):
            lines.extend(
                [
                    f"{i}. [{trace.level.value.upper()}] {trace.decision_type.value}",
                    f"   Time: {trace.timestamp.isoformat()}",
                    f"   Maker: {trace.decision_maker}",
                    f"   Description: {trace.decision_description}",
                    f"   Confidence: {trace.confidence:.3f}",
                    f"   Processing: {trace.processing_time_ms:.1f}ms",
                    f"   Reasoning: {trace.reasoning}",
                    f"   Inputs: {json.dumps(trace.inputs, indent=4) if trace.inputs else 'None'}",
                    f"   Outputs: {json.dumps(trace.outputs, indent=4) if trace.outputs else 'None'}",
                    "",
                ]
            )

        return "\n".join(lines)


# Global tracer instance
_global_tracer: Optional[CognitiveTracer] = None


def get_cognitive_tracer() -> CognitiveTracer:
    """Get or create global cognitive tracer instance"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = CognitiveTracer()
    return _global_tracer


def create_cognitive_tracer(enable_detailed_logging: bool = True) -> CognitiveTracer:
    """Create a new cognitive tracer instance"""
    return CognitiveTracer(enable_detailed_logging=enable_detailed_logging)
