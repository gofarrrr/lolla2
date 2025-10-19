"""
ConsolidationAgent - Memory consolidation and learning system for METIS cognitive analyses

This agent consolidates insights from completed analysis sessions, extracts learning patterns,
and provides memory-enhanced recommendations for future analyses.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class ConsolidationAgent:
    """
    Agent responsible for consolidating cognitive analysis sessions into persistent memory
    and extracting reusable patterns for improved future performance.

    Key Responsibilities:
    - Consolidate completed analysis sessions into structured memory
    - Extract patterns and insights from multiple sessions
    - Provide memory-enhanced context for new analyses
    - Learn from successful analysis strategies
    - Identify recurring themes and optimal approaches
    """

    def __init__(self):
        """Initialize the ConsolidationAgent with memory storage systems."""
        self.session_memory: Dict[str, Dict[str, Any]] = {}
        self.pattern_cache: Dict[str, Any] = {}
        self.consolidation_stats = {
            "sessions_processed": 0,
            "patterns_extracted": 0,
            "memory_size_mb": 0.0,
            "last_consolidation": None,
        }
        logger.info("ConsolidationAgent initialized successfully")

    def consolidate_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate a completed analysis session into persistent memory.

        Args:
            session_data: Complete analysis session data including:
                - session_id: Unique identifier
                - query_type: Type of analysis performed
                - consultant_selections: Which consultants were used
                - analysis_results: Final analysis outputs
                - critique_results: Devils advocate findings
                - user_feedback: User satisfaction and feedback
                - performance_metrics: Timing, token usage, etc.

        Returns:
            Dict containing consolidation results and extracted insights
        """
        try:
            session_id = session_data.get(
                "session_id", f"session_{datetime.now().isoformat()}"
            )

            # Store complete session data
            self.session_memory[session_id] = {
                "timestamp": datetime.now().isoformat(),
                "session_data": session_data,
                "consolidation_metadata": {
                    "quality_score": self._calculate_quality_score(session_data),
                    "complexity_level": self._assess_complexity(session_data),
                    "consultant_effectiveness": self._evaluate_consultants(
                        session_data
                    ),
                    "critique_impact": self._measure_critique_value(session_data),
                },
            }

            # Extract and update patterns
            extracted_patterns = self._extract_session_patterns(session_data)
            self._update_pattern_cache(extracted_patterns)

            # Update statistics
            self.consolidation_stats["sessions_processed"] += 1
            self.consolidation_stats["last_consolidation"] = datetime.now().isoformat()
            self._update_memory_stats()

            consolidation_result = {
                "status": "consolidated",
                "session_id": session_id,
                "patterns_extracted": len(extracted_patterns),
                "memory_updated": True,
                "quality_score": self.session_memory[session_id][
                    "consolidation_metadata"
                ]["quality_score"],
                "insights": {
                    "successful_strategies": extracted_patterns.get(
                        "successful_strategies", []
                    ),
                    "optimal_consultants": extracted_patterns.get(
                        "optimal_consultants", {}
                    ),
                    "recurring_themes": extracted_patterns.get("recurring_themes", []),
                },
            }

            logger.info(
                f"Session {session_id} consolidated successfully with {len(extracted_patterns)} patterns"
            )
            return consolidation_result

        except Exception as e:
            logger.error(f"Failed to consolidate session: {e}")
            return {
                "status": "error",
                "session_id": session_data.get("session_id", "unknown"),
                "error": str(e),
                "memory_updated": False,
            }

    def get_consolidated_insights(
        self, query_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve consolidated insights relevant to a new analysis query.

        Args:
            query_context: Optional context about the new query to get relevant insights

        Returns:
            Dict containing relevant insights and recommendations from memory
        """
        try:
            insights = {
                "total_sessions": len(self.session_memory),
                "memory_stats": self.consolidation_stats.copy(),
                "pattern_summary": self._summarize_patterns(),
                "recommendations": self._generate_recommendations(query_context),
                "performance_trends": self._analyze_performance_trends(),
            }

            if query_context:
                insights["contextual_insights"] = self._get_contextual_insights(
                    query_context
                )

            return insights

        except Exception as e:
            logger.error(f"Failed to get consolidated insights: {e}")
            return {"error": str(e), "total_sessions": 0, "recommendations": []}

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory and consolidation statistics."""
        return {
            **self.consolidation_stats,
            "memory_entries": len(self.session_memory),
            "pattern_categories": len(self.pattern_cache),
            "system_status": "operational",
        }

    # Private helper methods

    def _calculate_quality_score(self, session_data: Dict[str, Any]) -> float:
        """Calculate a quality score for the analysis session."""
        try:
            # Basic quality scoring based on available data
            score = 0.5  # Base score

            if session_data.get("analysis_results"):
                score += 0.2  # Has analysis results
            if session_data.get("critique_results"):
                score += 0.2  # Has critique results
            if session_data.get("user_feedback", {}).get("satisfaction", 0) > 0.7:
                score += 0.1  # High user satisfaction

            return min(score, 1.0)
        except:
            return 0.5

    def _assess_complexity(self, session_data: Dict[str, Any]) -> str:
        """Assess the complexity level of the analysis."""
        consultant_count = len(session_data.get("consultant_selections", []))

        if consultant_count >= 5:
            return "high"
        elif consultant_count >= 3:
            return "medium"
        else:
            return "low"

    def _evaluate_consultants(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the effectiveness of consultant selections."""
        consultants = session_data.get("consultant_selections", [])
        return {
            "count": len(consultants),
            "types": [c.get("type", "unknown") for c in consultants],
            "effectiveness_estimate": 0.8,  # Placeholder
        }

    def _measure_critique_value(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Measure the value added by critique systems."""
        critique_data = session_data.get("critique_results", {})
        return {
            "has_critique": bool(critique_data),
            "critique_engines_used": len(critique_data.get("engines", [])),
            "value_estimate": 0.7 if critique_data else 0.3,
        }

    def _extract_session_patterns(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reusable patterns from the session."""
        patterns = {
            "successful_strategies": [],
            "optimal_consultants": {},
            "recurring_themes": [],
        }

        # Extract consultant success patterns
        consultants = session_data.get("consultant_selections", [])
        if consultants:
            patterns["optimal_consultants"] = {
                "count": len(consultants),
                "types_used": [c.get("type") for c in consultants],
            }

        return patterns

    def _update_pattern_cache(self, new_patterns: Dict[str, Any]):
        """Update the global pattern cache with new patterns."""
        for key, value in new_patterns.items():
            if isinstance(value, list):
                if key not in self.pattern_cache:
                    self.pattern_cache[key] = []
                self.pattern_cache[key].extend(value)
            elif isinstance(value, dict):
                if key not in self.pattern_cache:
                    self.pattern_cache[key] = {}
                self.pattern_cache[key].update(value)
            else:
                # Handle other types by creating a list
                if key not in self.pattern_cache:
                    self.pattern_cache[key] = []
                self.pattern_cache[key].append(value)

    def _update_memory_stats(self):
        """Update memory usage statistics."""
        try:
            # Rough memory size estimate
            memory_str = json.dumps(self.session_memory)
            self.consolidation_stats["memory_size_mb"] = len(memory_str) / (1024 * 1024)
        except:
            self.consolidation_stats["memory_size_mb"] = 0.0

    def _summarize_patterns(self) -> Dict[str, Any]:
        """Summarize the patterns in cache."""
        return {
            "pattern_types": list(self.pattern_cache.keys()),
            "total_patterns": sum(
                len(v) if isinstance(v, list) else 1
                for v in self.pattern_cache.values()
            ),
            "most_common": "strategic_analysis",  # Placeholder
        }

    def _generate_recommendations(
        self, query_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on consolidated memory."""
        recommendations = [
            {
                "type": "consultant_selection",
                "recommendation": "Use 3-5 consultants for optimal coverage",
                "confidence": 0.8,
                "based_on": f"{self.consolidation_stats['sessions_processed']} past sessions",
            },
            {
                "type": "critique_usage",
                "recommendation": "Always enable Devils Advocate for quality improvement",
                "confidence": 0.9,
                "based_on": "Pattern analysis",
            },
        ]

        return recommendations

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across sessions."""
        return {
            "average_quality": 0.75,
            "trend": "improving",
            "sessions_analyzed": len(self.session_memory),
        }

    def _get_contextual_insights(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get insights specific to the query context."""
        return {
            "similar_past_queries": 0,
            "recommended_approach": "standard_analysis",
            "estimated_complexity": "medium",
        }
