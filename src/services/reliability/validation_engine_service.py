"""
METIS Validation Engine Service
Part of Reliability Services Cluster - Focused on multi-layer LLM output validation

Extracted from vulnerability_solutions.py ValidationEngine during Phase 5 decomposition.
Single Responsibility: Comprehensive validation and hallucination detection for LLM outputs.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.services.contracts.reliability_contracts import (
    ValidationResultContract,
    IValidationEngineService,
    ValidationLayer,
)


class ValidationEngineService(IValidationEngineService):
    """
    Focused service for multi-layer LLM output validation and hallucination detection
    Clean extraction from vulnerability_solutions.py ValidationEngine
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Validation layer configurations
        self.validation_layers = {
            ValidationLayer.FACTUAL_CONSISTENCY: {
                "weight": 0.25,
                "threshold": 0.7,
                "description": "Verify factual claims against known data",
            },
            ValidationLayer.LOGICAL_COHERENCE: {
                "weight": 0.20,
                "threshold": 0.8,
                "description": "Check logical consistency and reasoning flow",
            },
            ValidationLayer.RESEARCH_TRIANGULATION: {
                "weight": 0.25,
                "threshold": 0.75,
                "description": "Cross-validate against research sources",
            },
            ValidationLayer.CONFIDENCE_CALIBRATION: {
                "weight": 0.15,
                "threshold": 0.65,
                "description": "Assess confidence calibration accuracy",
            },
            ValidationLayer.CROSS_PROVIDER_VALIDATION: {
                "weight": 0.15,
                "threshold": 0.7,
                "description": "Compare across multiple LLM providers",
            },
        }

        # Hallucination detection patterns
        self.hallucination_patterns = [
            r"According to \[fabricated source\]",
            r"Recent studies show.*without citation",
            r"It is widely known that.*\(no source\)",
            r"Experts agree.*without attribution",
            r"\d{4} research indicates.*uncited",
        ]

        self.logger.info("🔍 ValidationEngineService initialized")

    async def validate_llm_output(
        self,
        llm_response: str,
        context: Dict[str, Any],
        research_base: List[Dict[str, Any]],
    ) -> ValidationResultContract:
        """
        Core service method: Comprehensive multi-layer validation of LLM output
        Clean, focused implementation with single responsibility
        """
        try:
            engagement_id = context.get("engagement_id", "unknown")

            # Execute all validation layers
            layer_results = []
            overall_passed = True
            all_issues = []
            all_evidence = []

            # Layer 1: Factual Consistency
            factual_result = await self._validate_factual_consistency(
                llm_response, research_base
            )
            layer_results.append(factual_result)
            if not factual_result["passed"]:
                overall_passed = False
            all_issues.extend(factual_result["issues"])
            all_evidence.extend(factual_result["evidence"])

            # Layer 2: Logical Coherence
            logical_result = await self._validate_logical_coherence(
                llm_response, context
            )
            layer_results.append(logical_result)
            if not logical_result["passed"]:
                overall_passed = False
            all_issues.extend(logical_result["issues"])
            all_evidence.extend(logical_result["evidence"])

            # Layer 3: Research Triangulation
            research_result = await self._validate_research_triangulation(
                llm_response, research_base
            )
            layer_results.append(research_result)
            if not research_result["passed"]:
                overall_passed = False
            all_issues.extend(research_result["issues"])
            all_evidence.extend(research_result["evidence"])

            # Layer 4: Confidence Calibration
            confidence_result = await self._validate_confidence_calibration(
                llm_response, context
            )
            layer_results.append(confidence_result)
            if not confidence_result["passed"]:
                overall_passed = False
            all_issues.extend(confidence_result["issues"])
            all_evidence.extend(confidence_result["evidence"])

            # Layer 5: Cross-Provider Validation (simplified in this implementation)
            cross_provider_result = await self._validate_cross_provider(
                llm_response, context
            )
            layer_results.append(cross_provider_result)
            if not cross_provider_result["passed"]:
                overall_passed = False
            all_issues.extend(cross_provider_result["issues"])
            all_evidence.extend(cross_provider_result["evidence"])

            # Create validation contract
            return ValidationResultContract(
                engagement_id=engagement_id,
                validation_layers=[layer["layer"] for layer in layer_results],
                overall_passed=overall_passed,
                layer_results=layer_results,
                issues_detected=all_issues,
                evidence_collected=all_evidence,
                validation_timestamp=datetime.utcnow(),
                service_version="v5_modular",
            )

        except Exception as e:
            self.logger.error(f"❌ LLM output validation failed: {e}")
            return self._create_fallback_validation(llm_response, context, str(e))

    async def _validate_factual_consistency(
        self, llm_response: str, research_base: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate factual claims against research base"""
        layer_config = self.validation_layers[ValidationLayer.FACTUAL_CONSISTENCY]
        issues = []
        evidence = []

        # Check for hallucination patterns
        for pattern in self.hallucination_patterns:
            if re.search(pattern, llm_response, re.IGNORECASE):
                issues.append(f"Potential hallucination pattern detected: {pattern}")

        # Verify specific claims against research base
        claims = self._extract_factual_claims(llm_response)
        for claim in claims:
            verification = self._verify_claim_against_research(claim, research_base)
            if verification["verified"]:
                evidence.append(f"Claim verified: {claim[:100]}...")
            else:
                issues.append(f"Unverified claim: {claim[:100]}...")

        # Calculate score based on verification rate
        total_claims = len(claims)
        verified_claims = len(evidence)
        score = verified_claims / total_claims if total_claims > 0 else 0.8

        passed = score >= layer_config["threshold"] and len(issues) == 0

        return {
            "layer": ValidationLayer.FACTUAL_CONSISTENCY.value,
            "passed": passed,
            "score": score,
            "issues": issues,
            "evidence": evidence,
            "description": layer_config["description"],
        }

    async def _validate_logical_coherence(
        self, llm_response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate logical consistency and reasoning flow"""
        layer_config = self.validation_layers[ValidationLayer.LOGICAL_COHERENCE]
        issues = []
        evidence = []

        # Check for logical consistency markers
        reasoning_steps = self._extract_reasoning_steps(llm_response)

        if len(reasoning_steps) >= 3:
            evidence.append("Clear reasoning structure with multiple steps")
        else:
            issues.append("Insufficient reasoning structure")

        # Check for logical contradictions (simplified)
        contradictions = self._detect_logical_contradictions(llm_response)
        if contradictions:
            issues.extend([f"Logical contradiction: {c}" for c in contradictions])
        else:
            evidence.append("No logical contradictions detected")

        # Check for proper conclusion support
        if "therefore" in llm_response.lower() or "thus" in llm_response.lower():
            evidence.append("Conclusions properly supported by reasoning")

        score = max(0.0, 1.0 - (len(issues) * 0.2))
        passed = score >= layer_config["threshold"]

        return {
            "layer": ValidationLayer.LOGICAL_COHERENCE.value,
            "passed": passed,
            "score": score,
            "issues": issues,
            "evidence": evidence,
            "description": layer_config["description"],
        }

    async def _validate_research_triangulation(
        self, llm_response: str, research_base: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Cross-validate against research sources"""
        layer_config = self.validation_layers[ValidationLayer.RESEARCH_TRIANGULATION]
        issues = []
        evidence = []

        if not research_base:
            issues.append("No research base provided for triangulation")
            score = 0.3
        else:
            # Check research integration
            research_citations = self._count_research_citations(llm_response)
            total_research_sources = len(research_base)

            citation_rate = research_citations / max(1, total_research_sources)

            if citation_rate >= 0.6:
                evidence.append(
                    f"Good research integration: {research_citations}/{total_research_sources} sources cited"
                )
            else:
                issues.append(
                    f"Low research integration: {research_citations}/{total_research_sources} sources cited"
                )

            # Check for research-supported claims
            supported_claims = self._count_research_supported_claims(
                llm_response, research_base
            )
            if supported_claims >= 3:
                evidence.append(
                    f"Multiple research-supported claims: {supported_claims}"
                )
            else:
                issues.append(f"Few research-supported claims: {supported_claims}")

            score = min(1.0, (citation_rate + (supported_claims / 5.0)) / 2.0)

        passed = score >= layer_config["threshold"]

        return {
            "layer": ValidationLayer.RESEARCH_TRIANGULATION.value,
            "passed": passed,
            "score": score,
            "issues": issues,
            "evidence": evidence,
            "description": layer_config["description"],
        }

    async def _validate_confidence_calibration(
        self, llm_response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess confidence calibration accuracy"""
        layer_config = self.validation_layers[ValidationLayer.CONFIDENCE_CALIBRATION]
        issues = []
        evidence = []

        # Check for appropriate confidence expressions
        confidence_expressions = self._extract_confidence_expressions(llm_response)

        if len(confidence_expressions) >= 2:
            evidence.append("Appropriate confidence calibration expressions found")
        else:
            issues.append("Insufficient confidence calibration")

        # Check for hedging and uncertainty acknowledgment
        uncertainty_markers = [
            "likely",
            "probably",
            "may",
            "might",
            "appears to",
            "suggests",
        ]
        uncertainty_count = sum(
            1 for marker in uncertainty_markers if marker in llm_response.lower()
        )

        if uncertainty_count >= 2:
            evidence.append("Appropriate uncertainty acknowledgment")
        else:
            issues.append("Overconfident tone without uncertainty acknowledgment")

        # Check confidence alignment with complexity
        problem_complexity = context.get("complexity_level", "medium")
        if problem_complexity == "high" and uncertainty_count < 3:
            issues.append(
                "High complexity problem requires more uncertainty acknowledgment"
            )

        score = max(0.0, 1.0 - (len(issues) * 0.25))
        passed = score >= layer_config["threshold"]

        return {
            "layer": ValidationLayer.CONFIDENCE_CALIBRATION.value,
            "passed": passed,
            "score": score,
            "issues": issues,
            "evidence": evidence,
            "description": layer_config["description"],
        }

    async def _validate_cross_provider(
        self, llm_response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cross-provider validation (simplified implementation)"""
        layer_config = self.validation_layers[ValidationLayer.CROSS_PROVIDER_VALIDATION]
        issues = []
        evidence = []

        # In full implementation, this would query alternative LLM providers
        # For now, use heuristics to simulate cross-provider validation

        provider_used = context.get("llm_provider", "unknown")
        if provider_used != "unknown":
            evidence.append(f"Response generated using {provider_used}")

        # Simulate cross-provider consistency check
        response_length = len(llm_response)
        if response_length > 500:
            evidence.append("Response length appropriate for cross-provider validation")
        else:
            issues.append("Response too brief for meaningful cross-provider validation")

        # Check for provider-specific patterns (simplified)
        if "I apologize" in llm_response or "I'm sorry" in llm_response:
            issues.append("Provider-specific apologetic language detected")

        score = max(0.0, 1.0 - (len(issues) * 0.3))
        passed = score >= layer_config["threshold"]

        return {
            "layer": ValidationLayer.CROSS_PROVIDER_VALIDATION.value,
            "passed": passed,
            "score": score,
            "issues": issues,
            "evidence": evidence,
            "description": layer_config["description"],
        }

    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Simplified implementation - in practice would use NLP
        sentences = text.split(".")
        claims = []

        for sentence in sentences:
            if any(
                word in sentence.lower()
                for word in [
                    "research shows",
                    "study found",
                    "data indicates",
                    "according to",
                ]
            ):
                claims.append(sentence.strip())

        return claims[:10]  # Limit to first 10 claims

    def _verify_claim_against_research(
        self, claim: str, research_base: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Verify a claim against research base"""
        # Simplified implementation - in practice would use semantic similarity
        for research in research_base:
            research_content = research.get("content", "").lower()
            claim_keywords = claim.lower().split()[:5]  # Use first 5 words

            if any(keyword in research_content for keyword in claim_keywords):
                return {"verified": True}

        return {"verified": False}

    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from text"""
        step_markers = [
            "first",
            "second",
            "third",
            "then",
            "next",
            "finally",
            "therefore",
        ]
        steps = []

        for marker in step_markers:
            if marker in text.lower():
                steps.append(marker)

        return steps

    def _detect_logical_contradictions(self, text: str) -> List[str]:
        """Detect logical contradictions in text"""
        # Simplified implementation
        contradictions = []

        if "always" in text.lower() and "never" in text.lower():
            contradictions.append("Absolute statements conflict")

        if "increase" in text.lower() and "decrease" in text.lower():
            # Check if they're about the same subject
            words_between = (
                text.lower().split("increase")[1].split("decrease")[0]
                if "increase" in text.lower() and "decrease" in text.lower()
                else ""
            )
            if len(words_between.split()) < 10:
                contradictions.append("Conflicting directional claims")

        return contradictions

    def _count_research_citations(self, text: str) -> int:
        """Count research citations in text"""
        citation_patterns = [
            r"\(.*\d{4}.*\)",  # Year in parentheses
            r"according to.*research",
            r"study.*found",
            r"research.*shows",
        ]

        total_citations = 0
        for pattern in citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            total_citations += len(matches)

        return total_citations

    def _count_research_supported_claims(
        self, text: str, research_base: List[Dict[str, Any]]
    ) -> int:
        """Count claims supported by research"""
        claims = self._extract_factual_claims(text)
        supported_count = 0

        for claim in claims:
            if self._verify_claim_against_research(claim, research_base)["verified"]:
                supported_count += 1

        return supported_count

    def _extract_confidence_expressions(self, text: str) -> List[str]:
        """Extract confidence expressions from text"""
        confidence_patterns = [
            r"confident that",
            r"certain that",
            r"likely that",
            r"probably",
            r"appears to",
            r"suggests that",
            r"indicates that",
        ]

        expressions = []
        for pattern in confidence_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            expressions.extend(matches)

        return expressions

    async def validate_research_findings(
        self, research_intelligence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate research findings for accuracy and reliability"""
        try:
            validation_score = 0.0
            issues = []
            strengths = []

            # Check source diversity
            sources = research_intelligence.get("sources", [])
            if len(sources) >= 5:
                strengths.append("Good source diversity")
                validation_score += 0.3
            else:
                issues.append(f"Limited source diversity: {len(sources)} sources")

            # Check confidence calibration
            overall_confidence = research_intelligence.get("overall_confidence", 0.0)
            if 0.6 <= overall_confidence <= 0.9:
                strengths.append("Well-calibrated confidence level")
                validation_score += 0.3
            else:
                issues.append(f"Confidence may be miscalibrated: {overall_confidence}")

            # Check for information gaps
            info_gaps = research_intelligence.get("information_gaps", [])
            if len(info_gaps) <= 3:
                strengths.append("Manageable information gaps")
                validation_score += 0.4
            else:
                issues.append(f"Many information gaps: {len(info_gaps)}")

            return {
                "validation_score": validation_score,
                "passed": validation_score >= 0.7,
                "issues": issues,
                "strengths": strengths,
            }

        except Exception as e:
            self.logger.error(f"❌ Research findings validation failed: {e}")
            return {
                "validation_score": 0.0,
                "passed": False,
                "issues": [f"Validation error: {str(e)}"],
                "strengths": [],
            }

    def _create_fallback_validation(
        self, llm_response: str, context: Dict[str, Any], error_msg: str
    ) -> ValidationResultContract:
        """Create fallback validation result when service fails"""
        return ValidationResultContract(
            engagement_id=context.get("engagement_id", "unknown"),
            validation_layers=["service_error"],
            overall_passed=False,
            layer_results=[
                {
                    "layer": "service_error",
                    "passed": False,
                    "score": 0.0,
                    "issues": [f"Validation service error: {error_msg}"],
                    "evidence": [],
                    "description": "Validation service encountered an error",
                }
            ],
            issues_detected=[f"Validation service error: {error_msg}"],
            evidence_collected=[],
            validation_timestamp=datetime.utcnow(),
            service_version="v5_modular_fallback",
        )

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "ValidationEngineService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "multi_layer_validation",
                "hallucination_detection",
                "factual_consistency_checking",
                "logical_coherence_validation",
                "research_triangulation",
                "confidence_calibration_assessment",
            ],
            "validation_layers": list(self.validation_layers.keys()),
            "layer_count": len(self.validation_layers),
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_validation_engine_service: Optional[ValidationEngineService] = None


def get_validation_engine_service() -> ValidationEngineService:
    """Get or create global validation engine service instance"""
    global _validation_engine_service

    if _validation_engine_service is None:
        _validation_engine_service = ValidationEngineService()

    return _validation_engine_service
