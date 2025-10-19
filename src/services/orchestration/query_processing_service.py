# src/services/orchestration/query_processing_service.py
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.orchestration.contracts import StructuredAnalyticalFramework, FrameworkType
from src.core.next_gen_query_chunker import ChunkingMode

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQueryResult:
    enhanced_query: Optional[str]
    chunking_result: Optional[Any]
    task_classification: Dict[str, Any]
    enhancement_metadata: Dict[str, Any]
    complexity_level: str


class QueryProcessingService:
    """Encapsulates research-based enhancement and adaptive chunking/classification."""

    def __init__(
        self,
        enhancer: Any = None,
        chunker: Any = None,
        classifier: Any = None,
        query_enhancement_enabled: bool = True,
    ) -> None:
        self.enhancer = enhancer
        self.chunker = chunker
        self.classifier = classifier
        self.query_enhancement_enabled = query_enhancement_enabled
        # Session-level disable flag for chunker, similar to orchestrator semantics
        self._chunker_available = chunker is not None

    async def process_query(
        self, initial_query: Optional[str], framework: StructuredAnalyticalFramework
    ) -> ProcessedQueryResult:
        # Step 0A: Research-based enhancement
        enhanced_query = initial_query
        enhancement_result = None
        if initial_query and self.query_enhancement_enabled and self.enhancer:
            logger.info(
                "🔬 RESEARCH-BASED ENHANCEMENT: Analyzing and enhancing user query..."
            )
            try:
                enhancement_result = await self.enhancer.enhance_query_with_research(
                    user_query=initial_query,
                    conversation_style="strategic_partner",
                    max_questions=4,
                    user_context={
                        "framework_type": framework.framework_type.value,
                        "analysis_dimensions": len(framework.primary_dimensions),
                    },
                )
                enhanced_query = enhancement_result.enhanced_query
                logger.info(
                    "✅ Query enhanced: confidence=%.2f, completeness=%.2f",
                    getattr(enhancement_result, "enhancement_confidence", 0.0),
                    getattr(enhancement_result, "information_completeness", 0.0),
                )
                minority_signals = getattr(
                    enhancement_result, "minority_signals_captured", None
                )
                if minority_signals:
                    logger.info("🧭 Minority signals preserved: %d", len(minority_signals))
            except Exception as e:
                logger.warning(
                    "⚠️ Query enhancement failed, using original query: %s", e
                )
                enhanced_query = initial_query

        # Step 0B: Adaptive chunking with fallback classification
        task_classification: Optional[Dict[str, Any]] = None
        chunking_result: Optional[Any] = None

        if enhanced_query:
            logger.info(
                "🚀 ADAPTIVE ORCHESTRATION: Using Next-Gen Query Chunker for strategic decomposition..."
            )
            if self._chunker_available and self.chunker:
                try:
                    chunking_result = await self.chunker.chunk_query(
                        query=enhanced_query,
                        mode=ChunkingMode.ADAPTIVE,
                        user_context={
                            "framework_type": framework.framework_type.value,
                            "dimensions": len(framework.primary_dimensions),
                            "query_enhancement_applied": enhancement_result is not None,
                            "enhancement_confidence": getattr(
                                enhancement_result, "enhancement_confidence", None
                            ),
                        },
                    )
                    # Extract classification from chunking result for compatibility
                    if getattr(chunking_result, "strategic_decomposition", None):
                        sd = chunking_result.strategic_decomposition
                        complexity_score = getattr(sd, "complexity_score", 0.5)
                        if complexity_score <= 0.4:
                            complexity_level = "low"
                        elif complexity_score <= 0.7:
                            complexity_level = "medium"
                        else:
                            complexity_level = "high"
                        task_classification = {
                            "primary_domain": getattr(sd, "primary_domain", None)
                            or "strategy",
                            "task_type": getattr(sd, "decomposition_type", None)
                            or "analytical",
                            "confidence": getattr(
                                chunking_result, "confidence_score", 0.6
                            ),
                            "reasoning": f"Strategic MECE decomposition: {getattr(sd, 'core_question', '')[:100]}...",
                            "complexity_level": complexity_level,
                            "requires_creativity": getattr(
                                sd, "requires_creative_thinking", False
                            ),
                        }
                        logger.info(
                            "🎯 Strategic chunking complete: %s | %s (confidence: %.2f)",
                            task_classification["primary_domain"],
                            task_classification["task_type"],
                            task_classification["confidence"],
                        )
                    else:
                        raise Exception("No strategic decomposition available")
                except Exception as e:
                    logger.warning(
                        "⚠️ Next-Gen Query Chunker failed, falling back to task classifier: %s",
                        e,
                    )
                    self._chunker_available = False

            if task_classification is None:
                logger.info("🎯 Using fallback Task Classification Service...")
                if self.classifier:
                    task_classification = await self.classifier.classify_task(
                        enhanced_query
                    )
                    if task_classification and "complexity_level" not in task_classification:
                        task_classification["complexity_level"] = self._infer_complexity_from_query(
                            enhanced_query
                        )
                else:
                    task_classification = self._fallback_task_classification(framework)
                logger.info(
                    "✅ Task classified (fallback): %s | %s (confidence: %.2f)",
                    task_classification["primary_domain"],
                    task_classification["task_type"],
                    task_classification.get("confidence", 0.6),
                )
        else:
            # No query provided: framework-based fallback
            task_classification = self._fallback_task_classification(framework)

        enhancement_metadata = self._build_enhancement_metadata(enhancement_result)
        complexity_level = (
            task_classification.get("complexity_level")
            if task_classification
            else self._infer_complexity_from_query(enhanced_query)
        )

        return ProcessedQueryResult(
            enhanced_query=enhanced_query,
            chunking_result=chunking_result,
            task_classification=task_classification or {},
            enhancement_metadata=enhancement_metadata,
            complexity_level=complexity_level or "medium",
        )

    def _fallback_task_classification(
        self, framework: StructuredAnalyticalFramework
    ) -> Dict[str, Any]:
        mapping = {
            FrameworkType.STRATEGIC_ANALYSIS: {
                "primary_domain": "strategy",
                "task_type": "analytical",
                "requires_creativity": False,
            },
            FrameworkType.OPERATIONAL_OPTIMIZATION: {
                "primary_domain": "operations",
                "task_type": "analytical",
                "requires_creativity": False,
            },
            FrameworkType.INNOVATION_DISCOVERY: {
                "primary_domain": "creative",
                "task_type": "ideation",
                "requires_creativity": True,
            },
            FrameworkType.CRISIS_MANAGEMENT: {
                "primary_domain": "strategy",
                "task_type": "analytical",
                "requires_creativity": False,
            },
        }
        m = mapping.get(
            framework.framework_type,
            {
                "primary_domain": "strategy",
                "task_type": "analytical",
                "requires_creativity": False,
            },
        )
        return {
            "primary_domain": m["primary_domain"],
            "task_type": m["task_type"],
            "confidence": 0.6,
            "reasoning": f"Inferred from framework type: {framework.framework_type.value}",
            "complexity_level": "medium",
            "requires_creativity": m["requires_creativity"],
            "classification_metadata": {
                "method": "framework_based_fallback",
                "framework_type": framework.framework_type.value,
                "timestamp": "auto",
            },
        }

    def _build_enhancement_metadata(self, enhancement_result: Any) -> Dict[str, Any]:
        """Extract lightweight metadata from enhancement results for downstream stages."""

        if enhancement_result is None:
            return {}

        metadata: Dict[str, Any] = {}

        try:
            gaps = getattr(enhancement_result, "information_gaps", None)
            if gaps:
                metadata["priority_targets"] = {
                    "critical": getattr(gaps, "critical_gaps", []),
                    "important": getattr(gaps, "important_gaps", []),
                    "useful": getattr(gaps, "useful_gaps", []),
                }

                prioritized = getattr(gaps, "prioritized_questions", []) or []
                formatted_priorities = []
                for pq in prioritized[:5]:
                    if isinstance(pq, dict):
                        formatted_priorities.append(
                            {
                                "question": pq.get("question_text"),
                                "information_target": pq.get("information_target"),
                                "information_value": pq.get("information_value"),
                            }
                        )
                    else:
                        formatted_priorities.append(
                            {
                                "question": getattr(pq, "question_text", None),
                                "information_target": getattr(pq, "information_target", None),
                                "information_value": getattr(pq, "information_value", None),
                            }
                        )
                if formatted_priorities:
                    metadata["prioritized_questions"] = formatted_priorities

            minority_signals = getattr(
                enhancement_result, "minority_signals_captured", None
            ) or []
            metadata["minority_signal_count"] = len(minority_signals)
            if minority_signals:
                metadata["minority_signals"] = [
                    getattr(signal, "signal_content", "") for signal in minority_signals[:3]
                ]

            metadata["framing_invariance_tested"] = bool(
                getattr(enhancement_result, "framing_variants_tested", None)
            )

        except Exception as exc:
            logger.warning(f"⚠️ Failed to extract enhancement metadata: {exc}")

        return metadata

    def _infer_complexity_from_query(self, query: Optional[str]) -> str:
        """Fallback heuristic for determining complexity level from query text."""

        if not query:
            return "medium"

        length = len(query)
        question_marks = query.count("?")

        if length < 160 and question_marks <= 1:
            return "low"
        if length < 320 and question_marks <= 2:
            return "medium"
        return "high"
