"""
METIS Reasoning Visualization Engine
Progressive transparency module for reasoning process visualization

Generates visual representations of reasoning processes including reasoning maps,
confidence visualizations, and interactive reasoning traces.
"""

from typing import Dict, List, Any
from src.engine.models.data_contracts import ReasoningStep, MentalModelDefinition


class ReasoningVisualizationEngine:
    """Generates visual representations of reasoning processes"""

    async def create_reasoning_map(
        self,
        reasoning_steps: List[ReasoningStep],
        mental_models: List[MentalModelDefinition],
    ) -> Dict[str, Any]:
        """Create visual reasoning map"""

        reasoning_map = {
            "type": "hierarchical_flow",
            "nodes": [],
            "edges": [],
            "clusters": [],
            "metadata": {
                "total_steps": len(reasoning_steps),
                "models_used": len(mental_models),
                "confidence_range": {"min": 0, "max": 1},
            },
        }

        # Create nodes for each reasoning step
        for i, step in enumerate(reasoning_steps):
            node = {
                "id": step.step_id,
                "type": "reasoning_step",
                "label": f"Step {i+1}: {step.mental_model_applied}",
                "content": (
                    step.reasoning_text[:100] + "..."
                    if len(step.reasoning_text) > 100
                    else step.reasoning_text
                ),
                "confidence": step.confidence_score,
                "timestamp": step.timestamp.isoformat(),
                "position": {"x": i * 200, "y": 100},
                "style": {
                    "color": self._confidence_to_color(step.confidence_score),
                    "size": self._confidence_to_size(step.confidence_score),
                },
            }
            reasoning_map["nodes"].append(node)

            # Create edge to next step
            if i < len(reasoning_steps) - 1:
                edge = {
                    "from": step.step_id,
                    "to": reasoning_steps[i + 1].step_id,
                    "type": "sequential",
                    "style": {"arrow": True},
                }
                reasoning_map["edges"].append(edge)

        # Create clusters for mental models
        model_clusters = {}
        for step in reasoning_steps:
            model = step.mental_model_applied
            if model not in model_clusters:
                model_clusters[model] = []
            model_clusters[model].append(step.step_id)

        for model, step_ids in model_clusters.items():
            cluster = {
                "id": f"cluster_{model}",
                "label": model.replace("_", " ").title(),
                "nodes": step_ids,
                "style": {"background": self._model_to_color(model)},
            }
            reasoning_map["clusters"].append(cluster)

        return reasoning_map

    def _confidence_to_color(self, confidence: float) -> str:
        """Convert confidence score to color"""
        if confidence >= 0.8:
            return "#2E7D32"  # Green
        elif confidence >= 0.6:
            return "#F57C00"  # Orange
        else:
            return "#C62828"  # Red

    def _confidence_to_size(self, confidence: float) -> int:
        """Convert confidence score to node size"""
        return int(20 + (confidence * 30))  # 20-50 pixel range

    def _model_to_color(self, model: str) -> str:
        """Convert mental model to cluster color"""
        color_map = {
            "systems_thinking": "#E3F2FD",
            "critical_thinking": "#FFF3E0",
            "mece_structuring": "#E8F5E8",
            "hypothesis_testing": "#FCE4EC",
            "decision_frameworks": "#F3E5F5",
        }

        for key, color in color_map.items():
            if key in model.lower():
                return color

        return "#F5F5F5"  # Default gray

    async def create_confidence_visualization(
        self, reasoning_steps: List[ReasoningStep]
    ) -> Dict[str, Any]:
        """Create confidence score visualization"""

        if not reasoning_steps:
            return {"type": "empty", "message": "No reasoning steps available"}

        confidence_data = [
            {
                "step": i + 1,
                "step_id": step.step_id,
                "confidence": step.confidence_score,
                "model": step.mental_model_applied,
                "timestamp": step.timestamp.isoformat(),
            }
            for i, step in enumerate(reasoning_steps)
        ]

        avg_confidence = sum(step.confidence_score for step in reasoning_steps) / len(
            reasoning_steps
        )

        visualization = {
            "type": "confidence_chart",
            "data": confidence_data,
            "summary": {
                "average_confidence": avg_confidence,
                "highest_confidence": max(
                    step.confidence_score for step in reasoning_steps
                ),
                "lowest_confidence": min(
                    step.confidence_score for step in reasoning_steps
                ),
                "confidence_trend": self._analyze_confidence_trend(reasoning_steps),
            },
            "chart_config": {
                "x_axis": "step",
                "y_axis": "confidence",
                "color_by": "model",
                "show_trend_line": True,
                "confidence_thresholds": [0.6, 0.8],
            },
        }

        return visualization

    def _analyze_confidence_trend(self, reasoning_steps: List[ReasoningStep]) -> str:
        """Analyze confidence trend across reasoning steps"""
        if len(reasoning_steps) < 2:
            return "stable"

        confidences = [step.confidence_score for step in reasoning_steps]

        # Simple trend analysis
        first_half_avg = sum(confidences[: len(confidences) // 2]) / (
            len(confidences) // 2
        )
        second_half_avg = sum(confidences[len(confidences) // 2 :]) / (
            len(confidences) - len(confidences) // 2
        )

        diff = second_half_avg - first_half_avg

        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        else:
            return "stable"
