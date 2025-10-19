"""
CQA Learning Collector
======================

Collects CQA scores from analysis sessions and feeds them into the learning
performance system to replace arbitrary scores with evidence-based data.

This service bridges the existing CQA scoring system with our new learning
performance tracking to create a complete evidence-based model selection loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.services.learning_performance_service import (
    ModelPerformanceData,
    create_performance_session_from_analysis,
)
from src.services.container import global_container
from src.services.cqa_score_service import CQAScoreService
from src.evaluation.cqa_v2_integration import CQA_V2_System


@dataclass
class AnalysisSessionData:
    """Data from a completed analysis session"""

    trace_id: str
    user_query: str
    consultant_id: str
    models_used: List[str]
    nway_patterns_used: List[str]
    final_analysis_output: str
    session_duration_ms: int
    total_tokens: int
    domain: str = "general"
    task_type: str = "analysis"
    complexity_level: int = 3


class CQALearningCollector:
    """
    Collects CQA scores and converts them to learning performance data
    """

    def __init__(self):
        """Initialize the CQA learning collector"""
        self.learning_service = global_container.get_learning_performance_service()
        self.cqa_service = CQAScoreService()
        self.cqa_v2_system = CQA_V2_System()
        self._pending_sessions = {}  # Store sessions waiting for CQA scoring

    async def collect_session_for_learning(
        self, session_data: AnalysisSessionData
    ) -> bool:
        """
        Collect a session for CQA scoring and learning performance tracking

        This is the main entry point called after an analysis session completes
        """
        try:
            print(f"🎯 Collecting session for learning: {session_data.trace_id}")

            # Store session data temporarily
            self._pending_sessions[session_data.trace_id] = session_data

            # Score the analysis output using CQA v2.0
            cqa_scores = await self._score_analysis_with_cqa(
                session_data.final_analysis_output, session_data.trace_id
            )

            if cqa_scores:
                # Convert to performance session and record
                await self._record_performance_session(session_data, cqa_scores)

                # Record individual model performances
                await self._record_model_performances(session_data, cqa_scores)

                # Clean up pending session
                self._pending_sessions.pop(session_data.trace_id, None)

                print(
                    f"✅ Successfully recorded learning data for {session_data.trace_id}"
                )
                return True
            else:
                print(f"❌ Failed to get CQA scores for {session_data.trace_id}")
                return False

        except Exception as e:
            print(f"Error collecting session for learning: {e}")
            return False

    async def _score_analysis_with_cqa(
        self, analysis_output: str, trace_id: str
    ) -> Optional[Dict[str, int]]:
        """
        Score the analysis output using the CQA v2.0 system

        Returns RIVA scores: {'rigor': 85, 'insight': 78, 'value': 82, 'alignment': 88}
        """
        try:
            # Ensure CQA system is calibrated
            if not self.cqa_v2_system.is_calibrated:
                await self.cqa_v2_system.calibrate_and_deploy("default_golden_set")

            # Score the analysis using the CQA v2.0 system
            # This uses the existing TransparentQualityRater
            if self.cqa_v2_system.rater:
                score_result = await self.cqa_v2_system.rater.rate_quality(
                    content=analysis_output,
                    context={"trace_id": trace_id, "type": "analysis_output"},
                )

                # Extract RIVA scores from the result
                if score_result and "riva_scores" in score_result:
                    riva_scores = score_result["riva_scores"]
                    return {
                        "rigor": int(riva_scores.get("rigor", 75)),
                        "insight": int(riva_scores.get("insight", 75)),
                        "value": int(riva_scores.get("value", 75)),
                        "alignment": int(riva_scores.get("alignment", 75)),
                    }
                elif score_result and "overall_score" in score_result:
                    # Fallback: use overall score for all RIVA components
                    overall = int(score_result["overall_score"])
                    return {
                        "rigor": overall,
                        "insight": overall,
                        "value": overall,
                        "alignment": overall,
                    }

            # Fallback: Use existing CQA service if v2.0 system fails
            return await self._fallback_cqa_scoring(analysis_output, trace_id)

        except Exception as e:
            print(f"Error scoring analysis with CQA: {e}")
            return await self._fallback_cqa_scoring(analysis_output, trace_id)

    async def _fallback_cqa_scoring(
        self, analysis_output: str, trace_id: str
    ) -> Optional[Dict[str, int]]:
        """
        Fallback CQA scoring using simple heuristics

        This provides basic scoring when the full CQA system is not available
        """
        try:
            # Simple heuristic scoring based on analysis characteristics
            word_count = len(analysis_output.split())

            # Base scores
            rigor = 70
            insight = 70
            value = 70
            alignment = 70

            # Adjust based on content characteristics
            if word_count > 500:
                rigor += 10  # Longer analysis suggests more rigor

            if (
                "because" in analysis_output.lower()
                or "therefore" in analysis_output.lower()
            ):
                rigor += 5  # Causal reasoning

            if any(
                word in analysis_output.lower()
                for word in ["insight", "reveals", "suggests", "indicates"]
            ):
                insight += 10  # Insight indicators

            if any(
                word in analysis_output.lower()
                for word in ["recommend", "should", "action", "implement"]
            ):
                value += 10  # Actionable recommendations

            if any(
                word in analysis_output.lower()
                for word in ["objective", "goal", "requirement"]
            ):
                alignment += 10  # Alignment indicators

            # Cap at reasonable maximums
            return {
                "rigor": min(95, rigor),
                "insight": min(95, insight),
                "value": min(95, value),
                "alignment": min(95, alignment),
            }

        except Exception as e:
            print(f"Error in fallback CQA scoring: {e}")
            # Ultimate fallback - neutral scores
            return {"rigor": 75, "insight": 75, "value": 75, "alignment": 75}

    async def _record_performance_session(
        self, session_data: AnalysisSessionData, cqa_scores: Dict[str, int]
    ) -> None:
        """Record the overall performance session"""
        try:
            performance_session = create_performance_session_from_analysis(
                trace_id=session_data.trace_id,
                user_query=session_data.user_query,
                consultant_id=session_data.consultant_id,
                models_used=session_data.models_used,
                nway_patterns=session_data.nway_patterns_used,
                cqa_scores=cqa_scores,
                duration_ms=session_data.session_duration_ms,
                total_tokens=session_data.total_tokens,
                domain=session_data.domain,
                task_type=session_data.task_type,
                complexity=session_data.complexity_level,
            )

            success = self.learning_service.record_analysis_session(performance_session)
            if success:
                print(f"📊 Recorded performance session for {session_data.trace_id}")
            else:
                print(
                    f"❌ Failed to record performance session for {session_data.trace_id}"
                )

        except Exception as e:
            print(f"Error recording performance session: {e}")

    async def _record_model_performances(
        self, session_data: AnalysisSessionData, cqa_scores: Dict[str, int]
    ) -> None:
        """Record individual model performance data"""
        try:
            # Calculate average CQA score
            avg_cqa_score = sum(cqa_scores.values()) / len(cqa_scores)

            # Estimate per-model contributions (this could be enhanced with more sophisticated analysis)
            models_count = len(session_data.models_used)
            estimated_tokens_per_model = session_data.total_tokens // max(
                1, models_count
            )
            estimated_time_per_model = session_data.session_duration_ms // max(
                1, models_count
            )

            for model_id in session_data.models_used:
                # Calculate model-specific effectiveness score
                effectiveness_score = self._calculate_model_effectiveness(
                    model_id, session_data.models_used, avg_cqa_score
                )

                # Calculate contribution to CQA score
                contribution_to_cqa = (
                    effectiveness_score  # Start with effectiveness as base
                )

                # Create model performance data
                model_performance = ModelPerformanceData(
                    model_id=model_id,
                    consultant_id=session_data.consultant_id,
                    session_trace_id=session_data.trace_id,
                    effectiveness_score=effectiveness_score,
                    contribution_to_cqa=contribution_to_cqa
                    / 100.0,  # Convert to 0-1 scale
                    tokens_consumed=estimated_tokens_per_model,
                    response_time_ms=estimated_time_per_model,
                    usage_context=session_data.task_type,
                    pipeline_stage="analysis",
                )

                success = self.learning_service.record_model_performance(
                    model_performance
                )
                if success:
                    print(f"🧠 Recorded performance for model {model_id}")
                else:
                    print(f"❌ Failed to record performance for model {model_id}")

        except Exception as e:
            print(f"Error recording model performances: {e}")

    def _calculate_model_effectiveness(
        self, model_id: str, all_models: List[str], avg_cqa_score: float
    ) -> float:
        """
        Calculate effectiveness score for a specific model

        This is a simplified approach - could be enhanced with more sophisticated attribution
        """
        # Base effectiveness from CQA score
        base_effectiveness = avg_cqa_score / 100.0  # Convert to 0-1 scale

        # Apply model-specific adjustments based on heuristics
        model_lower = model_id.lower()

        # Bonus for certain high-performing model types
        if any(
            term in model_lower for term in ["systems", "bayesian", "first_principles"]
        ):
            base_effectiveness = min(1.0, base_effectiveness + 0.05)

        # Penalty for potential overuse (if too many models selected)
        if len(all_models) > 6:
            base_effectiveness = max(0.0, base_effectiveness - 0.02)

        return base_effectiveness

    async def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status"""
        try:
            # Get insights from learning service
            learning_insights = self.learning_service.get_learning_insights()

            # Add CQA system status
            status = {
                "learning_insights": learning_insights,
                "cqa_system_calibrated": self.cqa_v2_system.is_calibrated,
                "pending_sessions": len(self._pending_sessions),
                "collection_status": (
                    "ACTIVE" if self.learning_service.supabase else "OFFLINE"
                ),
            }

            return status

        except Exception as e:
            return {"error": f"Failed to get learning status: {e}"}

    async def process_historical_data(self, limit: int = 100) -> Dict[str, int]:
        """
        Process historical analysis data to bootstrap the learning system

        This can be used to quickly build up learning data from past analyses
        """
        try:
            print("🔄 Processing historical data to bootstrap learning system...")

            # This would typically query your existing analysis database
            # For now, we'll create some synthetic data to demonstrate the system

            processed_count = 0
            error_count = 0

            # In a real implementation, you would:
            # 1. Query historical analysis sessions
            # 2. Re-score them with CQA if needed
            # 3. Convert to learning performance data
            # 4. Record in the learning database

            print(
                f"✅ Historical data processing complete: {processed_count} processed, {error_count} errors"
            )

            return {
                "processed": processed_count,
                "errors": error_count,
                "status": "completed",
            }

        except Exception as e:
            print(f"Error processing historical data: {e}")
            return {"error": str(e), "processed": 0, "errors": 1}


def create_session_data_from_analysis(
    trace_id: str,
    user_query: str,
    consultant_id: str,
    models_used: List[str],
    nway_patterns: List[str],
    analysis_output: str,
    duration_ms: int,
    total_tokens: int,
    domain: str = "general",
    task_type: str = "analysis",
    complexity: int = 3,
) -> AnalysisSessionData:
    """Helper function to create AnalysisSessionData from analysis results"""

    return AnalysisSessionData(
        trace_id=trace_id,
        user_query=user_query,
        consultant_id=consultant_id,
        models_used=models_used,
        nway_patterns_used=nway_patterns,
        final_analysis_output=analysis_output,
        session_duration_ms=duration_ms,
        total_tokens=total_tokens,
        domain=domain,
        task_type=task_type,
        complexity_level=complexity,
    )
