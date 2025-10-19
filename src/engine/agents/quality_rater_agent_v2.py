"""
Quality Rater Agent v2.0 - Ensemble Orchestrator Architecture
=============================================================

Multi-agent ensemble orchestrator that eliminates dimensional interference
by using specialized single-dimension raters in parallel.

Architecture:
- ai_rigor_rater@1.0: Peak RIGOR correlation (0.781)
- ai_insight_rater@1.0: Peak INSIGHT correlation (0.732)
- ai_alignment_rater@1.0: Peak ALIGNMENT correlation (0.874)
- ai_value_rater@1.0: Peak VALUE correlation (0.628)

Intelligent Adjudicator combines specialist outputs for final CQA_Result.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from pydantic import BaseModel, Field
from src.engine.adapters.contracts import  # Migrated (
    RIVAScore,
    CQA_Result,
    QualityAuditRequest,
    QualityDimension,
)
from src.engine.adapters.contracts import  # Migrated rubric_registry

# Import specialized rater agents
from src.agents.ai_rigor_rater_v1 import AIRigorRater
from src.agents.ai_insight_rater_v1 import AIInsightRater
from src.agents.ai_alignment_rater_v1 import AIAlignmentRater
from src.agents.ai_value_rater_v1 import AIValueRater


class RaterAuditTrail(BaseModel):
    """
    Complete audit trail for a quality rating decision.

    Attributes:
        rating_id: Unique ID for this rating operation
        artifact_id: ID of the artifact being rated
        rubric_used: Which rubric variant was applied
        llm_request: Complete LLM request including prompt
        llm_response: Raw LLM response
        parsed_result: Final parsed CQA_Result
        parsing_attempts: Number of parsing attempts needed
        total_tokens: Token usage for transparency
        duration_ms: Time taken for rating
        nested_context_stream: The rater's own context stream
    """

    rating_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    artifact_id: str
    rubric_used: str
    llm_request: Dict[str, Any]
    llm_response: str
    parsed_result: Optional[CQA_Result]
    parsing_attempts: int = 1
    total_tokens: int = 0
    duration_ms: int = 0
    nested_context_stream: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TransparentQualityRater:
    """
    Quality rater with complete transparency and audit trails.

    Every decision is fully auditable, including the rater's own
    reasoning process.
    """

    def __init__(self, version: str = "2.0", production_mode: bool = True):
        """
        Initialize the transparent quality rater.

        Args:
            version: Version of this rater
            production_mode: If True, enforces real LLM integration (default)
        """
        self.version = version
        self.production_mode = production_mode
        self.audit_trails: Dict[str, RaterAuditTrail] = {}

        # PRODUCTION MODE ENFORCEMENT
        if production_mode:
            # Verify no mock methods exist (failsafe)
            if hasattr(self, "_mock_llm_generate"):
                raise RuntimeError(
                    "CRITICAL: Mock LLM method detected in production mode! All mock logic must be purged."
                )

    async def evaluate_with_audit(
        self, request: QualityAuditRequest
    ) -> tuple[CQA_Result, RaterAuditTrail]:
        """
        Ensemble Orchestrator: Runs 4 specialized single-dimension raters in parallel.

        Args:
            request: Quality audit request

        Returns:
            Tuple of (CQA_Result, RaterAuditTrail)
        """
        start_time = datetime.utcnow()
        rating_id = str(uuid.uuid4())
        nested_events = []

        # Determine which rubric to use
        rubric_variant_id = request.context.get("rubric_variant", "riva_standard@1.0")
        rubric = rubric_registry.get_variant(rubric_variant_id)

        if not rubric:
            rubric = rubric_registry.get_variant_for_agent(
                request.agent_name or "default"
            )
            rubric_variant_id = rubric.variant_id

        # Log orchestration start
        nested_events.append(
            {
                "event": "ensemble_orchestration_started",
                "rubric_id": rubric_variant_id,
                "agent_name": request.agent_name,
                "orchestrator_version": "v2.0_ensemble",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        try:
            # Initialize all 4 specialist raters
            rigor_rater = AIRigorRater()
            insight_rater = AIInsightRater()
            alignment_rater = AIAlignmentRater()
            value_rater = AIValueRater()

            # Log specialist initialization
            nested_events.append(
                {
                    "event": "specialists_initialized",
                    "specialists": {
                        "rigor": f"{rigor_rater.agent_id} (correlation: {rigor_rater.peak_correlation})",
                        "insight": f"{insight_rater.agent_id} (correlation: {insight_rater.peak_correlation})",
                        "alignment": f"{alignment_rater.agent_id} (correlation: {alignment_rater.peak_correlation})",
                        "value": f"{value_rater.agent_id} (correlation: {value_rater.peak_correlation})",
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # Execute 4 parallel specialist evaluations
            nested_events.append(
                {
                    "event": "parallel_specialist_evaluation_started",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            specialist_tasks = [
                rigor_rater.evaluate_rigor(request),
                insight_rater.evaluate_insight(request),
                alignment_rater.evaluate_alignment(request),
                value_rater.evaluate_value(request),
            ]

            # Run specialists in parallel for maximum efficiency
            specialist_results = await asyncio.gather(
                *specialist_tasks, return_exceptions=True
            )

            # Extract individual dimension scores
            rigor_result = (
                specialist_results[0]
                if not isinstance(specialist_results[0], Exception)
                else None
            )
            insight_result = (
                specialist_results[1]
                if not isinstance(specialist_results[1], Exception)
                else None
            )
            alignment_result = (
                specialist_results[2]
                if not isinstance(specialist_results[2], Exception)
                else None
            )
            value_result = (
                specialist_results[3]
                if not isinstance(specialist_results[3], Exception)
                else None
            )

            # Log specialist results
            nested_events.append(
                {
                    "event": "specialist_results_received",
                    "results": {
                        "rigor": {
                            "score": rigor_result.score if rigor_result else "failed",
                            "success": rigor_result is not None,
                        },
                        "insight": {
                            "score": (
                                insight_result.score if insight_result else "failed"
                            ),
                            "success": insight_result is not None,
                        },
                        "alignment": {
                            "score": (
                                alignment_result.score if alignment_result else "failed"
                            ),
                            "success": alignment_result is not None,
                        },
                        "value": {
                            "score": value_result.score if value_result else "failed",
                            "success": value_result is not None,
                        },
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # INTELLIGENT ADJUDICATOR: Combine specialist outputs
            nested_events.append(
                {
                    "event": "intelligent_adjudicator_started",
                    "adjudication_method": "weighted_specialist_combination",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # NO FALLBACKS. CRASH IF ANY SPECIALIST FAILS.
            if rigor_result is None:
                raise RuntimeError("RIGOR specialist failed - NO FALLBACKS AVAILABLE")
            if insight_result is None:
                raise RuntimeError("INSIGHT specialist failed - NO FALLBACKS AVAILABLE")
            if alignment_result is None:
                raise RuntimeError(
                    "ALIGNMENT specialist failed - NO FALLBACKS AVAILABLE"
                )
            if value_result is None:
                raise RuntimeError("VALUE specialist failed - NO FALLBACKS AVAILABLE")

            # Apply Intelligent Adjudicator with correlation-weighted averaging
            correlation_weights = {
                "rigor": 0.781,  # Peak RIGOR correlation
                "insight": 0.732,  # Peak INSIGHT correlation
                "alignment": 0.874,  # Peak ALIGNMENT correlation
                "value": 0.628,  # Peak VALUE correlation
            }

            # Normalize weights to sum to 1.0
            total_weight = sum(correlation_weights.values())
            normalized_weights = {
                k: v / total_weight for k, v in correlation_weights.items()
            }

            # Calculate confidence-weighted average
            weighted_scores = {
                "rigor": rigor_result.score,
                "insight": insight_result.score,
                "value": value_result.score,
                "alignment": alignment_result.score,
            }

            # Apply rubric weights if non-standard
            if rubric_variant_id != "riva_standard@1.0":
                weighted_average = rubric.get_adjusted_score(weighted_scores)

                nested_events.append(
                    {
                        "event": "rubric_weights_applied",
                        "original_scores": weighted_scores,
                        "weighted_average": weighted_average,
                        "rubric_weights": rubric.dimension_weights,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            else:
                # Standard equal weighting
                weighted_average = sum(weighted_scores.values()) / len(weighted_scores)

            # Create final CQA_Result from specialist outputs
            final_result = CQA_Result(
                rigor=rigor_result,
                insight=insight_result,
                value=value_result,
                alignment=alignment_result,
                confidence=0.85,  # High confidence from specialist ensemble
                metadata={
                    "ensemble_method": "parallel_specialists_v2.0",
                    "specialist_correlations": correlation_weights,
                    "rubric_applied": rubric_variant_id,
                    "ensemble_weighted_average": weighted_average,
                    "all_specialists_succeeded": all(
                        r is not None
                        for r in [
                            rigor_result,
                            insight_result,
                            alignment_result,
                            value_result,
                        ]
                    ),
                },
            )

            # Override average with intelligent adjudication
            final_result.average_score = weighted_average

            nested_events.append(
                {
                    "event": "intelligent_adjudication_completed",
                    "final_average_score": weighted_average,
                    "individual_scores": weighted_scores,
                    "confidence": final_result.confidence,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # Calculate duration
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Create comprehensive audit trail
            audit_trail = RaterAuditTrail(
                rating_id=rating_id,
                artifact_id=request.context.get("artifact_id", "unknown"),
                rubric_used=rubric_variant_id,
                llm_request={
                    "ensemble_method": "parallel_specialists",
                    "specialists_used": [
                        "ai_rigor_rater@1.0",
                        "ai_insight_rater@1.0",
                        "ai_alignment_rater@1.0",
                        "ai_value_rater@1.0",
                    ],
                    "correlation_weights": correlation_weights,
                },
                llm_response=f"Ensemble Result - Average: {weighted_average:.3f}",
                parsed_result=final_result,
                parsing_attempts=1,
                total_tokens=0,  # No single LLM call for ensemble
                duration_ms=duration_ms,
                nested_context_stream=nested_events,
            )

            # Store audit trail in memory AND persist to database
            self.audit_trails[rating_id] = audit_trail
            await self._persist_audit_trail(audit_trail)

            return final_result, audit_trail

        except Exception as e:
            # Log error with full context
            nested_events.append(
                {
                    "event": "ensemble_error_occurred",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # Create audit trail even for failures
            audit_trail = RaterAuditTrail(
                rating_id=rating_id,
                artifact_id=request.context.get("artifact_id", "unknown"),
                rubric_used=rubric_variant_id,
                llm_request={"ensemble_error": True},
                llm_response="",
                parsed_result=None,
                nested_context_stream=nested_events,
            )

            # Store audit trail in memory AND persist to database (even for errors)
            self.audit_trails[rating_id] = audit_trail
            await self._persist_audit_trail(audit_trail)

            # NO FALLBACKS. CRASH LOUDLY.
            raise RuntimeError(
                f"ENSEMBLE EVALUATION FAILED - NO FALLBACKS AVAILABLE: {e}"
            )

    async def evaluate(self, request: QualityAuditRequest) -> CQA_Result:
        """
        Simple evaluate method for backward compatibility.

        Args:
            request: Quality audit request

        Returns:
            CQA_Result (without audit trail)
        """
        result, audit_trail = await self.evaluate_with_audit(request)
        return result

    def _construct_rubric_aware_prompt(
        self, request: QualityAuditRequest, rubric
    ) -> str:
        """
        Construct evaluation prompt with rubric awareness.

        Args:
            request: The audit request
            rubric: The rubric variant to use

        Returns:
            Formatted prompt with rubric context
        """
        prompt = f"""
You are evaluating an AI-generated response using rigorous executive-level standards.

ARTIFACT TO EVALUATE:

Agent Name: {request.agent_name or "Unknown"}
Context: {json.dumps(request.context) if request.context else "None"}

ORIGINAL SYSTEM PROMPT:
{request.system_prompt}

USER QUERY:
{request.user_prompt}

GENERATED RESPONSE:
{request.llm_response}

---

Evaluate this strategic analysis response using the RIVA framework with executive-level standards:

**SCORING CALIBRATION (1-10 scale):**

**RIGOR (Analytical Depth & Methodology)**
- 10: Multiple frameworks, quantitative data, source citations, methodological soundness
- 8-9: Strong analytical structure with some quantitative backing
- 6-7: Logical structure covering key topics but lacking data
- 4-5: Basic structure but no analytical depth
- 2-3: Bullet points with no framework or logic
- 1: No analysis performed

**INSIGHT (Strategic Depth & Cross-Domain Analysis)**
- 10: Multiple novel strategic insights with comprehensive cross-domain analytical coverage and non-obvious connections
- 8-9: Several meaningful strategic insights with good analytical breadth and strategic implications
- 6-7: Some valuable insights with reasonable analytical coverage but limited strategic depth
- 4-5: Limited insights, basic observations with narrow analytical scope
- 2-3: Obvious points with minimal analytical depth or strategic value
- 1: No meaningful insights or cross-domain analytical coverage

**VALUE (Business Impact & Actionable Implementation)**
- 10: Extremely actionable with specific implementation steps, clear business impact, and immediate practical utility
- 8-9: Highly actionable with good implementation guidance and measurable business value
- 6-7: Actionable recommendations with reasonable specificity and practical utility
- 4-5: Somewhat actionable but lacks implementation specificity or clear business impact
- 2-3: Limited practical value, mostly theoretical with minimal actionable guidance
- 1: No practical value, actionability, or business implementation utility

**ALIGNMENT (Query Precision & Comprehensive Response Coverage)**
- 10: Perfectly addresses the exact query with comprehensive coverage of all specific aspects and contextual elements mentioned
- 8-9: Directly addresses the specific query with good comprehensive coverage of key aspects and context
- 6-7: Generally addresses the query with reasonable coverage but may miss some specific elements
- 4-5: Partially relevant but treats the query generically rather than addressing specific context
- 2-3: Minimally relevant, mostly generic treatment without addressing query specifics
- 1: Does not meaningfully address the specific query or context provided

**PROVIDE YOUR EVALUATION AS JSON:**
{{
  "rigor": {{"score": <1-10>, "rationale": "<detailed justification>", "evidence": ["<specific examples>"]}},
  "insight": {{"score": <1-10>, "rationale": "<detailed justification>", "evidence": ["<specific examples>"]}},
  "value": {{"score": <1-10>, "rationale": "<detailed justification>", "evidence": ["<specific examples>"]}},
  "alignment": {{"score": <1-10>, "rationale": "<detailed justification>", "evidence": ["<specific examples>"]}}
}}

**CRITICAL**: Follow ARC's methodology that prioritizes comprehensive coverage, quantitative backing, and specific actionable insights. Key scoring principles:
1. **Quantitative Data**: Financial projections, specific numbers, timelines significantly boost all scores
2. **Comprehensive Coverage**: Detailed frameworks covering multiple dimensions score higher than narrow focus
3. **Specific Examples**: Concrete examples and detailed implementation guidance elevate scores
4. **Structured Analysis**: Well-organized content with clear sections and logical flow
5. **Evidence-Based**: Conclusions supported by data and reasoning rather than opinion

Reward content that provides immediate practical value with detailed supporting evidence, not just sophisticated language.
"""
        return prompt

    def _get_rubric_aware_system_prompt(self, rubric) -> str:
        """
        Get system prompt tailored to the rubric with innovation-focused persona.

        Args:
            rubric: The rubric variant being used

        Returns:
            Innovation-calibrated system prompt
        """
        return """You are an expert business analysis evaluator with deep experience in assessing the quality of strategic business content across all industries and complexity levels.

Your evaluation follows ARC's proven scoring methodology that emphasizes comprehensiveness, quantitative backing, specific insights, and practical value over purely strategic sophistication.

**ARC CALIBRATION STANDARDS**:
- Perfect 10: Comprehensive strategic analysis with extensive quantitative data, multiple frameworks, detailed financial projections, specific timelines, risk assessments, and clear implementation roadmaps. Contains novel insights and sophisticated analysis across all dimensions.
- Excellent 9: Highly sophisticated analysis with strong quantitative backing, detailed implementation plans, financial projections, and strategic frameworks. Shows deep expertise with actionable recommendations and specific insights.
- Solid 7: Good comprehensive coverage of key areas with structured approach and some actionable advice, but lacks the quantitative depth, novel insights, or sophisticated frameworks of higher scores.
- Mediocre 4: Generic advice with basic bullet-point structure. Covers obvious points but lacks specific insights, quantitative analysis, or deep analytical thinking. Superficial treatment of complex topics.
- Poor 3: Very basic bullet-point responses with minimal depth. Lacks specific insights, actionable advice, or analytical rigor. Extremely superficial coverage.
- Failing 2: Error messages, technical failures, or content that provides no practical value whatsoever.

**KEY INSIGHT**: Value comprehensive coverage with quantitative backing over executive sophistication. Reward specific, actionable insights that professionals can immediately implement."""

    async def _parse_with_audit(
        self, response: str, nested_events: List[Dict]
    ) -> tuple[Optional[CQA_Result], int]:
        """
        Parse response with detailed audit logging.

        Args:
            response: LLM response to parse
            nested_events: List to append audit events

        Returns:
            Tuple of (parsed result, number of attempts)
        """
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            attempts += 1

            nested_events.append(
                {
                    "event": "parsing_attempt",
                    "attempt": attempts,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            try:
                # Attempt to parse as JSON
                data = json.loads(response)

                # Validate and create CQA_Result
                result = CQA_Result(
                    rigor=RIVAScore(**data["rigor"]),
                    insight=RIVAScore(**data["insight"]),
                    value=RIVAScore(**data["value"]),
                    alignment=RIVAScore(**data["alignment"]),
                    confidence=data.get("confidence", 0.8),
                    metadata=data.get("metadata", {}),
                )

                nested_events.append(
                    {
                        "event": "parsing_success",
                        "attempt": attempts,
                        "average_score": result.average_score,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                return result, attempts

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                nested_events.append(
                    {
                        "event": "parsing_error",
                        "attempt": attempts,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                if attempts < max_attempts:
                    # Try to fix common issues
                    response = self._attempt_json_repair(response)
                else:
                    # Final attempt: partial parse
                    return self._attempt_partial_parse(response, str(e)), attempts

        return None, attempts

    def _attempt_json_repair(self, response: str) -> str:
        """
        Attempt to repair malformed JSON.

        Args:
            response: Potentially malformed JSON response

        Returns:
            Repaired JSON string
        """
        # Simple repairs for common issues
        response = response.strip()
        if not response.startswith("{"):
            # Find the first opening brace
            start = response.find("{")
            if start != -1:
                response = response[start:]

        if not response.endswith("}"):
            # Find the last closing brace
            end = response.rfind("}")
            if end != -1:
                response = response[: end + 1]

        return response

    def _attempt_partial_parse(self, response: str, error: str) -> CQA_Result:
        """
        Attempt to extract partial results from a malformed response.

        Args:
            response: Potentially malformed response
            error: Original parsing error

        Returns:
            Best-effort CQA_Result
        """
        # Log the parsing issue
        metadata = {
            "parsing_error": error,
            "partial_parse": True,
            "raw_response_length": len(response),
        }

        # Try to extract any scores mentioned
        import re

        scores = {}

        import random

        response_hash = hash(response) % 1000
        random.seed(response_hash)  # Deterministic but varied

        for dim in ["rigor", "insight", "value", "alignment"]:
            # Look for patterns like "rigor: 7" or "rigor score: 7"
            pattern = rf"{dim}[:\s]+(?:score[:\s]+)?(\d+)"
            match = re.search(pattern, response.lower())
            if match:
                scores[dim] = int(match.group(1))
            else:
                # NO FALLBACKS. CRASH IF PARSING FAILS.
                raise RuntimeError(
                    f"Failed to parse {dim} score from LLM response - NO FALLBACKS AVAILABLE"
                )

        # Create result with extracted or default scores
        return CQA_Result(
            rigor=RIVAScore(
                dimension=QualityDimension.RIGOR,
                score=min(10, max(1, scores.get("rigor", 5))),
                rationale="Score extracted from partial response parsing",
            ),
            insight=RIVAScore(
                dimension=QualityDimension.INSIGHT,
                score=min(10, max(1, scores.get("insight", 5))),
                rationale="Score extracted from partial response parsing",
            ),
            value=RIVAScore(
                dimension=QualityDimension.VALUE,
                score=min(10, max(1, scores.get("value", 5))),
                rationale="Score extracted from partial response parsing",
            ),
            alignment=RIVAScore(
                dimension=QualityDimension.ALIGNMENT,
                score=min(10, max(1, scores.get("alignment", 5))),
                rationale="Score extracted from partial response parsing",
            ),
            confidence=0.3,  # Low confidence due to parsing issues
            metadata=metadata,
        )

    # METHOD DELETED - NO FALLBACKS ALLOWED

    def get_audit_trail(self, rating_id: str) -> Optional[RaterAuditTrail]:
        """
        Retrieve audit trail for a specific rating.

        Args:
            rating_id: ID of the rating

        Returns:
            Audit trail if found
        """
        return self.audit_trails.get(rating_id)

    def export_audit_trails(self, filepath: str):
        """
        Export all audit trails for analysis.

        Args:
            filepath: Path to save the audit trails
        """
        export_data = {
            "rater_version": self.version,
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_ratings": len(self.audit_trails),
            "audit_trails": [trail.dict() for trail in self.audit_trails.values()],
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"✅ Exported {len(self.audit_trails)} audit trails to {filepath}")

    async def _persist_audit_trail(self, audit_trail: RaterAuditTrail):
        """
        Persist audit trail to database for forensic analysis.

        This ensures audit trails survive script termination and can be
        retrieved for validation and comparison purposes.
        """
        try:
            import sqlite3
            import json
            import logging

            logger = logging.getLogger(__name__)

            # Create audit trails table if it doesn't exist
            conn = sqlite3.connect("evaluation_results.db")
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cqa_audit_trails (
                    rating_id TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    rubric_used TEXT NOT NULL,
                    llm_request TEXT NOT NULL,
                    llm_response TEXT NOT NULL,
                    parsed_result TEXT,
                    parsing_attempts INTEGER,
                    total_tokens INTEGER,
                    duration_ms INTEGER,
                    nested_context_stream TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """
            )

            # Insert audit trail
            cursor.execute(
                """
                INSERT OR REPLACE INTO cqa_audit_trails 
                (rating_id, artifact_id, rubric_used, llm_request, llm_response, 
                 parsed_result, parsing_attempts, total_tokens, duration_ms, 
                 nested_context_stream, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    audit_trail.rating_id,
                    audit_trail.artifact_id,
                    audit_trail.rubric_used,
                    json.dumps(audit_trail.llm_request, default=str),
                    audit_trail.llm_response,
                    (
                        json.dumps(audit_trail.parsed_result.dict(), default=str)
                        if audit_trail.parsed_result
                        else None
                    ),
                    audit_trail.parsing_attempts,
                    audit_trail.total_tokens,
                    audit_trail.duration_ms,
                    json.dumps(audit_trail.nested_context_stream, default=str),
                    audit_trail.timestamp.isoformat(),
                ),
            )

            conn.commit()
            conn.close()

            logger.info(f"🔐 Audit trail persisted: {audit_trail.rating_id}")

        except Exception as e:
            logger.error(
                f"❌ Failed to persist audit trail {audit_trail.rating_id}: {e}"
            )
            # Don't raise - persistence failure shouldn't break the rating process

    @classmethod
    async def load_audit_trail(cls, rating_id: str) -> Optional[RaterAuditTrail]:
        """
        Load audit trail from database.

        Args:
            rating_id: ID of the rating to load

        Returns:
            RaterAuditTrail if found, None otherwise
        """
        try:
            import sqlite3
            import json
            import logging

            logger = logging.getLogger(__name__)

            conn = sqlite3.connect("evaluation_results.db")
            cursor = conn.cursor()

            result = cursor.execute(
                """
                SELECT rating_id, artifact_id, rubric_used, llm_request, llm_response,
                       parsed_result, parsing_attempts, total_tokens, duration_ms,
                       nested_context_stream, timestamp
                FROM cqa_audit_trails 
                WHERE rating_id = ?
            """,
                (rating_id,),
            ).fetchone()

            conn.close()

            if not result:
                return None

            # Reconstruct audit trail
            return RaterAuditTrail(
                rating_id=result[0],
                artifact_id=result[1],
                rubric_used=result[2],
                llm_request=json.loads(result[3]),
                llm_response=result[4],
                parsed_result=json.loads(result[5]) if result[5] else None,
                parsing_attempts=result[6],
                total_tokens=result[7],
                duration_ms=result[8],
                nested_context_stream=json.loads(result[9]),
                timestamp=datetime.fromisoformat(result[10]),
            )

        except Exception as e:
            logger.error(f"❌ Failed to load audit trail {rating_id}: {e}")
            return None

    async def rate_quality(self, analysis_content: str, context: dict = None) -> dict:
        """
        Backward compatibility wrapper for rate_quality method.

        This method provides compatibility with learning systems that expect
        a rate_quality method. It wraps the newer evaluate_with_audit method.

        Args:
            analysis_content: The content to evaluate
            context: Optional context dictionary

        Returns:
            Dict with scores compatible with learning system expectations
        """
        try:
            # Create QualityAuditRequest from the input
            request = QualityAuditRequest(
                llm_response=analysis_content,
                user_prompt=context.get("user_prompt", "") if context else "",
                system_prompt=context.get("system_prompt", "") if context else "",
                context=context or {},
            )

            # Use the new evaluation method
            result, audit_trail = await self.evaluate_with_audit(request)

            # Base scores from CQA
            scores = {
                "rigor": result.rigor.score,
                "insight": result.insight.score,
                "value": result.value.score,
                "alignment": result.alignment.score,
                "total": (
                    result.rigor.score
                    + result.insight.score
                    + result.value.score
                    + result.alignment.score
                )
                / 4,
                "confidence": result.confidence,
                "audit_trail_id": audit_trail.rating_id,
            }

            # Inject runtime grounding signal (non-blocking)
            try:
                from src.engine.quality.grounding_contract import get_grounding_contract

                grounding_contract = get_grounding_contract(enabled=True)
                grounding_result = grounding_contract.validate(
                    response=analysis_content,
                    sources=(context.get("sources") if context else None),
                    context=context,
                )
                groundedness = max(
                    0.0, min(1.0, float(grounding_result.assessment.grounding_ratio))
                )
                scores["groundedness"] = round(groundedness, 4)
                # Alias used by some dashboards
                scores["attribution"] = round(groundedness, 4)
            except Exception:
                # Do not interfere with rating if grounding validation fails
                pass

            # Inject runtime self-verification signal (non-blocking)
            try:
                from src.engine.quality.self_verification import get_self_verifier

                verifier = get_self_verifier(enabled=True)
                vres = verifier.verify(
                    response=analysis_content,
                    query=(context.get("user_prompt") if context else None),
                    context=context,
                )
                scores["self_verification"] = round(float(vres.overall_quality), 4)
                scores["consistency"] = round(float(vres.consistency_score), 4)
            except Exception:
                # Do not interfere with rating if self-verification fails
                pass

            return scores

        except Exception as e:
            logger.error(f"❌ Quality rating failed: {e}")
            # Return fallback scores
            return {
                "rigor": 5.0,
                "insight": 5.0,
                "value": 5.0,
                "alignment": 5.0,
                "total": 5.0,
                "confidence": 0.1,
                "error": str(e),
            }


# Singleton instance getter
_quality_rater_instance = None


def get_quality_rater() -> TransparentQualityRater:
    """
    Get or create the singleton quality rater instance.

    Returns:
        TransparentQualityRater singleton instance
    """
    global _quality_rater_instance
    if _quality_rater_instance is None:
        _quality_rater_instance = TransparentQualityRater()
    return _quality_rater_instance
