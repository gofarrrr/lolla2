#!/usr/bin/env python3
"""
Coordination Detector

Monitors Three Consultant system to ensure consultants operate independently
with no coordination or synthesis. Part of the Multi-Single-Agent paradigm validation.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class CoordinationEvent:
    """Detected coordination event between consultants"""

    event_type: str
    consultant_a: str
    consultant_b: str
    evidence: str
    severity: float  # 0.0 to 1.0
    timestamp: datetime


class CoordinationDetector:
    """
    Detects coordination/synthesis between consultants which violates
    the Multi-Single-Agent paradigm requirement for independence.
    """

    def __init__(self, enable_monitoring: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_monitoring = enable_monitoring
        self.coordination_events: List[CoordinationEvent] = []

        if self.enable_monitoring:
            self.logger.info(
                "👁️ CoordinationDetector initialized - monitoring consultant independence"
            )
        else:
            self.logger.info("👁️ CoordinationDetector initialized - monitoring disabled")

    def analyze_consultant_outputs(
        self, consultant_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze consultant outputs for coordination/synthesis violations.

        Args:
            consultant_outputs: Dictionary of consultant responses

        Returns:
            Analysis report with independence validation
        """
        if not self.enable_monitoring:
            return {"monitoring_disabled": True}

        analysis_report = {
            "independence_validated": True,
            "coordination_events": [],
            "cross_references_detected": 0,
            "synthesis_attempts_detected": 0,
            "overall_independence_score": 1.0,
        }

        if not consultant_outputs:
            return analysis_report

        # Check for cross-consultant references (violation)
        consultant_names = list(consultant_outputs.keys())
        cross_references = 0

        for consultant_a, output_a in consultant_outputs.items():
            if isinstance(output_a, dict) and "analysis" in output_a:
                analysis_text = str(output_a["analysis"]).lower()

                # Check if this consultant references other consultants
                for consultant_b in consultant_names:
                    if (
                        consultant_a != consultant_b
                        and consultant_b.lower() in analysis_text
                    ):
                        cross_references += 1

                        event = CoordinationEvent(
                            event_type="cross_reference",
                            consultant_a=consultant_a,
                            consultant_b=consultant_b,
                            evidence=f"{consultant_a} referenced {consultant_b} in analysis",
                            severity=0.8,
                            timestamp=datetime.utcnow(),
                        )

                        self.coordination_events.append(event)
                        analysis_report["coordination_events"].append(event.__dict__)

        analysis_report["cross_references_detected"] = cross_references

        # Check for synthesis language (violation)
        synthesis_keywords = [
            "combining",
            "synthesis",
            "together",
            "consensus",
            "agree",
            "disagree",
            "other consultant",
            "colleague",
        ]

        synthesis_count = 0
        for consultant, output in consultant_outputs.items():
            if isinstance(output, dict) and "analysis" in output:
                analysis_text = str(output["analysis"]).lower()

                for keyword in synthesis_keywords:
                    if keyword in analysis_text:
                        synthesis_count += 1

                        event = CoordinationEvent(
                            event_type="synthesis_attempt",
                            consultant_a=consultant,
                            consultant_b="unknown",
                            evidence=f"Synthesis language detected: '{keyword}'",
                            severity=0.9,
                            timestamp=datetime.utcnow(),
                        )

                        self.coordination_events.append(event)
                        analysis_report["coordination_events"].append(event.__dict__)

        analysis_report["synthesis_attempts_detected"] = synthesis_count

        # Calculate overall independence score
        total_violations = cross_references + synthesis_count
        max_possible_violations = len(consultant_names) * 2  # Rough estimate

        if max_possible_violations > 0:
            independence_score = max(
                0.0, 1.0 - (total_violations / max_possible_violations)
            )
        else:
            independence_score = 1.0

        analysis_report["overall_independence_score"] = independence_score
        analysis_report["independence_validated"] = independence_score >= 0.8

        if total_violations > 0:
            self.logger.warning(
                f"🚨 Coordination violations detected: {total_violations}"
            )
            self.logger.warning(f"📊 Independence score: {independence_score:.2f}")
        else:
            self.logger.info(
                f"✅ No coordination violations - Independence score: {independence_score:.2f}"
            )

        return analysis_report

    def get_coordination_events(self) -> List[CoordinationEvent]:
        """Get all detected coordination events"""
        return self.coordination_events.copy()

    def clear_events(self):
        """Clear coordination event history"""
        self.coordination_events.clear()
        self.logger.info("🗑️ Coordination event history cleared")

    def get_independence_summary(self) -> Dict[str, Any]:
        """Get summary of consultant independence monitoring"""
        return {
            "monitoring_enabled": self.enable_monitoring,
            "total_events_detected": len(self.coordination_events),
            "event_types": {
                "cross_references": len(
                    [
                        e
                        for e in self.coordination_events
                        if e.event_type == "cross_reference"
                    ]
                ),
                "synthesis_attempts": len(
                    [
                        e
                        for e in self.coordination_events
                        if e.event_type == "synthesis_attempt"
                    ]
                ),
            },
            "last_analysis": (
                self.coordination_events[-1].timestamp.isoformat()
                if self.coordination_events
                else None
            ),
        }
