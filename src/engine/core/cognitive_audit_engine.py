#!/usr/bin/env python3
"""
Cognitive Audit Engine - Devils Advocate Engine #3
Implements cognitive auditing techniques including motivated reasoning detection and Ideological Turing Test
Based on "The Right AI Augmentation" methodology for systematic bias detection
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging


@dataclass
class MotivatedReasoningPattern:
    """Detected pattern of motivated reasoning"""

    pattern_type: str  # confirmation_bias, moving_goalposts, double_standards, cherry_picking, etc.
    pattern_name: str
    description: str
    evidence: List[str]
    severity: float  # 0.0-1.0
    ideological_turing_test_questions: List[str]
    mitigation_strategy: str
    detection_confidence: float


@dataclass
class CognitiveAuditResult:
    """Complete result from cognitive audit engine"""

    situation_analyzed: str
    motivated_reasoning_patterns: List[MotivatedReasoningPattern]
    overall_bias_score: float
    intellectual_honesty_assessment: float
    clarifying_questions: List[str]
    ideological_turing_test: Dict[str, List[str]]
    critical_thinking_recommendations: List[str]
    processing_time_ms: float
    audit_confidence: float


class CognitiveAuditEngine:
    """
    Cognitive auditing engine implementing advanced bias detection techniques

    Based on "The Right AI Augmentation" article methodology:
    1. Motivated reasoning pattern detection
    2. Ideological Turing Test implementation
    3. Systematic clarifying questions
    4. Double standard identification
    5. Emotional investment analysis
    6. Alternative perspective generation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Motivated reasoning pattern catalog
        self.reasoning_patterns = {
            "confirmation_bias": {
                "name": "Confirmation Bias",
                "description": "Seeking information that confirms preexisting beliefs",
                "indicators": ["proves", "confirms", "supports", "validates", "shows"],
                "turing_questions": [
                    "What evidence would convince you that your preferred approach is wrong?",
                    "How would someone who disagrees frame this differently?",
                ],
            },
            "moving_goalposts": {
                "name": "Moving Goalposts",
                "description": "Changing success criteria when original criteria aren't met",
                "indicators": [
                    "actually",
                    "really",
                    "what matters is",
                    "more important",
                    "focus on",
                ],
                "turing_questions": [
                    "How have your success criteria changed during this analysis?",
                    "What would you have said was most important before starting this process?",
                ],
            },
            "double_standards": {
                "name": "Double Standards Application",
                "description": "Applying different standards to similar situations",
                "indicators": [
                    "but this is different",
                    "special case",
                    "exception",
                    "unlike",
                ],
                "turing_questions": [
                    "Would you apply the same reasoning to a competitor in this situation?",
                    "How would you evaluate this if it were someone else's proposal?",
                ],
            },
            "cherry_picking": {
                "name": "Cherry-Picking Evidence",
                "description": "Selecting only supporting evidence while ignoring contradictory data",
                "indicators": [
                    "studies show",
                    "research indicates",
                    "data suggests",
                    "evidence points",
                ],
                "turing_questions": [
                    "What data or studies contradict this conclusion?",
                    "Have you actively sought disconfirming evidence?",
                ],
            },
            "anchoring_bias": {
                "name": "Anchoring Bias",
                "description": "Over-relying on first information received",
                "indicators": [
                    "initially",
                    "first",
                    "original",
                    "started with",
                    "began",
                ],
                "turing_questions": [
                    "How has your thinking evolved from your initial position?",
                    "What if you had encountered the information in reverse order?",
                ],
            },
            "loss_aversion": {
                "name": "Loss Aversion",
                "description": "Overweighting potential losses versus equivalent gains",
                "indicators": ["lose", "risk", "threat", "danger", "protect"],
                "turing_questions": [
                    "Are you overweighting the risks compared to potential benefits?",
                    "How would you frame this as an opportunity rather than a threat?",
                ],
            },
            "sunk_cost_fallacy": {
                "name": "Sunk Cost Fallacy",
                "description": "Continuing failing course due to past investment",
                "indicators": [
                    "already invested",
                    "spent so much",
                    "too far",
                    "committed",
                ],
                "turing_questions": [
                    "If you were starting fresh today, would you begin this initiative?",
                    "How is past investment influencing your current decision?",
                ],
            },
            "authority_bias": {
                "name": "Authority Bias",
                "description": "Deferring to authority without independent analysis",
                "indicators": [
                    "expert says",
                    "consultant recommends",
                    "leader believes",
                    "authority",
                ],
                "turing_questions": [
                    "What would your analysis conclude without the authority's input?",
                    "How might the authority figure be wrong or biased?",
                ],
            },
            "in_group_bias": {
                "name": "In-Group Bias",
                "description": "Favoring perspectives of similar others",
                "indicators": [
                    "our team",
                    "we believe",
                    "our experience",
                    "people like us",
                ],
                "turing_questions": [
                    "How would outsiders or different stakeholders view this?",
                    "What perspectives are missing from your analysis?",
                ],
            },
        }

        # Ideological Turing Test framework - can you accurately represent opposing views?
        self.turing_test_categories = {
            "stakeholder_perspectives": [
                "How would each major stakeholder describe this situation?",
                "What would their primary concerns and objections be?",
                "What solutions would they prefer and why?",
            ],
            "alternative_framings": [
                "How would a skeptic frame this problem differently?",
                "What alternative explanations exist for the current situation?",
                "How might this look from a completely different industry perspective?",
            ],
            "opposing_arguments": [
                "What are the strongest arguments against your preferred approach?",
                "Who would most strongly oppose this and what would they say?",
                "What are the best counterarguments to your reasoning?",
            ],
            "different_values": [
                "How would someone with different values approach this?",
                "What if the priorities were completely reversed?",
                "How would different cultures or contexts handle this?",
            ],
        }

        # Critical clarifying questions for motivated reasoning detection
        self.clarifying_questions = {
            "emotional_stakes": [
                "What personal or professional stakes do you have in this outcome?",
                "How would you feel emotionally if your preferred solution failed?",
                "What would success in this area mean for your career or reputation?",
            ],
            "information_processing": [
                "What information are you emphasizing vs. dismissing?",
                "How are you weighing conflicting pieces of evidence?",
                "What sources or perspectives are you not considering?",
            ],
            "assumption_examination": [
                "What assumptions are you treating as facts?",
                "Which of your beliefs about this situation could be wrong?",
                "What would need to be true for your preferred solution to fail?",
            ],
            "alternative_testing": [
                "What would convince you to change your mind?",
                "How would you test whether your preferred approach is actually optimal?",
                "What experiments could validate or invalidate your reasoning?",
            ],
        }

    async def audit_motivated_reasoning(
        self, recommendation: str, business_context: Dict[str, Any]
    ) -> CognitiveAuditResult:
        """Audit for motivated reasoning patterns using systematic methodology"""

        import time

        start_time = time.time()

        print("🔍 COGNITIVE AUDIT ENGINE")
        print("-" * 60)
        print(f"Auditing: {recommendation[:80]}...")

        # Step 1: Detect motivated reasoning patterns
        detected_patterns = []
        for pattern_key, pattern_config in self.reasoning_patterns.items():
            pattern_result = self._detect_reasoning_pattern(
                pattern_key, pattern_config, recommendation, business_context
            )
            if pattern_result and pattern_result.severity >= 0.3:
                detected_patterns.append(pattern_result)

        # Step 2: Generate clarifying questions
        clarifying_questions = self._generate_clarifying_questions(
            recommendation, business_context
        )

        # Step 3: Conduct Ideological Turing Test
        turing_test_results = self._conduct_ideological_turing_test(
            recommendation, business_context
        )

        # Step 4: Calculate bias and intellectual honesty scores
        overall_bias_score = self._calculate_overall_bias_score(detected_patterns)
        intellectual_honesty = self._assess_intellectual_honesty(
            detected_patterns, clarifying_questions, turing_test_results
        )

        # Step 5: Generate critical thinking recommendations
        thinking_recommendations = self._generate_critical_thinking_recommendations(
            detected_patterns, business_context
        )

        # Step 6: Calculate audit confidence
        audit_confidence = self._calculate_audit_confidence(
            detected_patterns, clarifying_questions
        )

        processing_time = (time.time() - start_time) * 1000

        print("📊 Cognitive Audit Results:")
        print(f"├─ Reasoning patterns: {len(detected_patterns)}")
        print(f"├─ Overall bias score: {overall_bias_score:.3f}")
        print(f"├─ Intellectual honesty: {intellectual_honesty:.3f}")
        print(f"└─ Processing time: {processing_time:.1f}ms")

        # Display detected patterns
        if detected_patterns:
            print("\n⚠️ Detected Reasoning Patterns:")
            sorted_patterns = sorted(
                detected_patterns, key=lambda x: x.severity, reverse=True
            )
            for i, pattern in enumerate(sorted_patterns[:3], 1):
                print(f"{i}. {pattern.pattern_name} (severity: {pattern.severity:.2f})")
                print(f"   → {pattern.description}")

        return CognitiveAuditResult(
            situation_analyzed=recommendation,
            motivated_reasoning_patterns=detected_patterns,
            overall_bias_score=overall_bias_score,
            intellectual_honesty_assessment=intellectual_honesty,
            clarifying_questions=clarifying_questions,
            ideological_turing_test=turing_test_results,
            critical_thinking_recommendations=thinking_recommendations,
            processing_time_ms=processing_time,
            audit_confidence=audit_confidence,
        )

    def _detect_reasoning_pattern(
        self,
        pattern_key: str,
        pattern_config: Dict[str, Any],
        recommendation: str,
        context: Dict[str, Any],
    ) -> Optional[MotivatedReasoningPattern]:
        """Detect specific motivated reasoning pattern"""

        # Check linguistic indicators
        rec_lower = recommendation.lower()
        indicator_matches = 0
        evidence_examples = []

        for indicator in pattern_config["indicators"]:
            if indicator in rec_lower:
                indicator_matches += 1
                # Find context around the indicator
                sentences = recommendation.split(".")
                matching_sentences = [
                    s.strip() for s in sentences if indicator in s.lower()
                ]
                if matching_sentences:
                    evidence_examples.append(
                        f"Uses '{indicator}': {matching_sentences[0][:100]}..."
                    )

        # Check contextual indicators
        contextual_evidence = self._detect_contextual_reasoning_patterns(
            pattern_key, recommendation, context
        )
        evidence_examples.extend(contextual_evidence)

        # Calculate severity
        linguistic_severity = min(0.6, indicator_matches * 0.2)
        contextual_severity = len(contextual_evidence) * 0.15
        total_severity = min(1.0, linguistic_severity + contextual_severity)

        if total_severity < 0.3:  # Not significant enough
            return None

        # Generate Ideological Turing Test questions
        turing_questions = pattern_config.get("turing_questions", [])

        # Generate specific mitigation strategy
        mitigation_strategy = self._generate_pattern_mitigation(
            pattern_key, evidence_examples
        )

        # Calculate detection confidence
        detection_confidence = min(
            1.0, (indicator_matches * 0.3) + (len(evidence_examples) * 0.2)
        )

        return MotivatedReasoningPattern(
            pattern_type=pattern_key,
            pattern_name=pattern_config["name"],
            description=pattern_config["description"],
            evidence=evidence_examples[:4],  # Top 4 pieces of evidence
            severity=total_severity,
            ideological_turing_test_questions=turing_questions,
            mitigation_strategy=mitigation_strategy,
            detection_confidence=detection_confidence,
        )

    def _detect_contextual_reasoning_patterns(
        self, pattern_key: str, recommendation: str, context: Dict[str, Any]
    ) -> List[str]:
        """Detect contextual indicators of motivated reasoning patterns"""

        contextual_evidence = []

        # Pattern-specific contextual analysis
        if pattern_key == "confirmation_bias":
            if context.get("stated_preferences"):
                contextual_evidence.append(
                    "Stated preferences may drive selective evidence gathering"
                )
            if "consultant" in str(context.get("stakeholders", [])).lower():
                contextual_evidence.append(
                    "External validation may substitute for independent analysis"
                )

        elif pattern_key == "authority_bias":
            stakeholders = context.get("stakeholders", [])
            if any(
                title in str(stakeholders).lower()
                for title in ["ceo", "cto", "consultant", "expert"]
            ):
                contextual_evidence.append(
                    "High-authority stakeholders present, risk of deference without analysis"
                )

        elif pattern_key == "loss_aversion":
            if context.get("financial_constraints") or context.get("timeline_pressure"):
                contextual_evidence.append(
                    "Constraint pressure may amplify loss aversion tendencies"
                )

        elif pattern_key == "in_group_bias":
            if context.get("company") and "team" in recommendation.lower():
                contextual_evidence.append(
                    "Internal team perspective may dominate external viewpoints"
                )

        elif pattern_key == "sunk_cost_fallacy":
            if (
                "investment" in recommendation.lower()
                or "spent" in recommendation.lower()
            ):
                contextual_evidence.append(
                    "Investment language suggests potential sunk cost influence"
                )

        elif pattern_key == "moving_goalposts":
            if context.get("timeline_pressure") or "deadline" in recommendation.lower():
                contextual_evidence.append(
                    "Time pressure may cause success criteria flexibility"
                )

        return contextual_evidence

    def _generate_clarifying_questions(
        self, recommendation: str, context: Dict[str, Any]
    ) -> List[str]:
        """Generate targeted clarifying questions to expose motivated reasoning"""

        questions = []

        # Always ask core clarifying questions
        questions.extend(
            [
                "What are your emotional stakes in this specific outcome?",
                "What evidence would convince you that your preferred approach is wrong?",
                "How might someone who strongly disagrees with you frame this situation?",
            ]
        )

        # Context-specific questions
        stakeholders = context.get("stakeholders", [])
        if "CEO" in stakeholders or "executive" in str(stakeholders).lower():
            questions.append(
                "How might your relationship with leadership influence your analysis?"
            )

        if context.get("financial_constraints"):
            questions.append(
                "How is financial pressure affecting your risk tolerance and analysis?"
            )

        if context.get("timeline_pressure"):
            questions.append(
                "Is time pressure causing you to settle for 'good enough' analysis?"
            )

        if context.get("stated_preferences"):
            questions.append(
                "How are your stated preferences biasing your evaluation of alternatives?"
            )

        # Recommendation-specific questions
        if "competitive" in recommendation.lower():
            questions.append(
                "Are competitive concerns driving decisions that may not optimize for your actual goals?"
            )

        if "must" in recommendation.lower() or "need to" in recommendation.lower():
            questions.append(
                "What assumptions are you treating as absolute requirements?"
            )

        return questions[:8]  # Limit to 8 most relevant questions

    def _conduct_ideological_turing_test(
        self, recommendation: str, context: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Conduct Ideological Turing Test to assess perspective-taking ability"""

        turing_results = {}

        # Generate questions for each category
        for category, base_questions in self.turing_test_categories.items():
            category_questions = base_questions.copy()

            # Customize questions based on context
            if category == "stakeholder_perspectives" and context.get("stakeholders"):
                stakeholders = context["stakeholders"]
                category_questions.append(
                    f"Specifically, how would {stakeholders[0]} vs. {stakeholders[-1]} view this differently?"
                )

            if (
                category == "opposing_arguments"
                and "competitive" in recommendation.lower()
            ):
                category_questions.append(
                    "How would your main competitor respond to this strategy?"
                )

            turing_results[category] = category_questions[
                :4
            ]  # Top 4 questions per category

        return turing_results

    def _generate_pattern_mitigation(
        self, pattern_key: str, evidence: List[str]
    ) -> str:
        """Generate specific mitigation strategy for detected reasoning pattern"""

        mitigation_strategies = {
            "confirmation_bias": "Actively seek disconfirming evidence and assign someone to argue the opposite position",
            "moving_goalposts": "Document success criteria upfront and require explicit justification for any changes",
            "double_standards": "Apply the same evaluation framework to all similar situations systematically",
            "cherry_picking": "Review all available evidence systematically, not just supportive data",
            "anchoring_bias": "Consider how your analysis might differ if you encountered information in reverse order",
            "loss_aversion": "Explicitly quantify both potential gains and losses using expected value framework",
            "sunk_cost_fallacy": "Evaluate future prospects independent of past investments using fresh-start analysis",
            "authority_bias": "Develop independent analysis before consulting authorities",
            "in_group_bias": "Explicitly seek perspectives from outsiders and potential critics",
        }

        base_strategy = mitigation_strategies.get(
            pattern_key, "Apply systematic critical thinking checklist"
        )

        # Add evidence-specific additions
        if evidence and "pressure" in " ".join(evidence):
            base_strategy += (
                " | Account for how pressure may be affecting judgment quality"
            )

        if evidence and "stakeholder" in " ".join(evidence):
            base_strategy += (
                " | Explicitly consider how stakeholder relationships may bias analysis"
            )

        return base_strategy

    def _calculate_overall_bias_score(
        self, patterns: List[MotivatedReasoningPattern]
    ) -> float:
        """Calculate overall cognitive bias score"""

        if not patterns:
            return 0.2  # Low baseline bias

        # Weighted average by severity and confidence
        total_weighted_bias = 0.0
        total_weights = 0.0

        for pattern in patterns:
            weight = pattern.detection_confidence
            weighted_bias = pattern.severity * weight
            total_weighted_bias += weighted_bias
            total_weights += weight

        average_bias = total_weighted_bias / total_weights if total_weights > 0 else 0.0

        # Penalty for multiple high-severity patterns (compound bias risk)
        high_severity_patterns = [p for p in patterns if p.severity >= 0.7]
        compound_penalty = len(high_severity_patterns) * 0.1

        total_bias_score = average_bias + compound_penalty

        return min(1.0, total_bias_score)

    def _assess_intellectual_honesty(
        self,
        patterns: List[MotivatedReasoningPattern],
        clarifying_questions: List[str],
        turing_test: Dict[str, List[str]],
    ) -> float:
        """Assess intellectual honesty based on audit results"""

        # Base intellectual honesty (inverted bias score)
        bias_penalty = self._calculate_overall_bias_score(patterns)
        base_honesty = 1.0 - bias_penalty

        # Bonus for systematic questioning capability
        question_quality_bonus = min(0.2, len(clarifying_questions) * 0.025)

        # Bonus for perspective-taking capability (Turing test)
        perspective_categories = len(turing_test)
        perspective_bonus = min(0.15, perspective_categories * 0.05)

        # Penalty for severe motivated reasoning patterns
        severe_patterns = [p for p in patterns if p.severity >= 0.8]
        severe_penalty = len(severe_patterns) * 0.15

        total_honesty = (
            base_honesty + question_quality_bonus + perspective_bonus - severe_penalty
        )

        return max(0.0, min(1.0, total_honesty))

    def _generate_critical_thinking_recommendations(
        self, patterns: List[MotivatedReasoningPattern], context: Dict[str, Any]
    ) -> List[str]:
        """Generate specific critical thinking improvement recommendations"""

        recommendations = []

        # Pattern-based recommendations
        if patterns:
            pattern_types = set(p.pattern_type for p in patterns)

            if "confirmation_bias" in pattern_types:
                recommendations.append(
                    "Implement systematic devil's advocate process with rotating roles"
                )

            if "authority_bias" in pattern_types:
                recommendations.append(
                    "Develop independent analysis framework before consulting experts"
                )

            if "cherry_picking" in pattern_types:
                recommendations.append(
                    "Create comprehensive evidence review protocol including contradictory data"
                )

            if len(patterns) >= 3:
                recommendations.append(
                    "Apply structured critical thinking checklist for all major decisions"
                )

        # Context-based recommendations
        if context.get("timeline_pressure"):
            recommendations.append(
                "Build explicit time allocation for critical thinking despite pressure"
            )

        if context.get("stakeholders") and len(context["stakeholders"]) >= 3:
            recommendations.append(
                "Implement multi-stakeholder perspective-taking exercise"
            )

        # Universal recommendations
        recommendations.extend(
            [
                "Practice Ideological Turing Test: accurately represent opposing viewpoints",
                "Establish pre-commitment to evidence that would change your mind",
            ]
        )

        return recommendations[:6]  # Top 6 recommendations

    def _calculate_audit_confidence(
        self, patterns: List[MotivatedReasoningPattern], clarifying_questions: List[str]
    ) -> float:
        """Calculate confidence in the audit results"""

        # Base confidence from pattern detection quality
        if patterns:
            avg_detection_confidence = sum(
                p.detection_confidence for p in patterns
            ) / len(patterns)
            pattern_confidence = avg_detection_confidence
        else:
            pattern_confidence = 0.7  # Moderate confidence when no patterns found

        # Question quality contribution
        question_confidence = min(
            1.0, len(clarifying_questions) / 6
        )  # Target 6 questions

        # Audit completeness
        completeness_confidence = (
            0.8 if len(patterns) >= 2 or len(clarifying_questions) >= 5 else 0.6
        )

        # Overall confidence (weighted average)
        overall_confidence = (
            pattern_confidence * 0.4
            + question_confidence * 0.3
            + completeness_confidence * 0.3
        )

        return min(1.0, overall_confidence)


async def demonstrate_cognitive_audit():
    """Demonstrate cognitive audit engine"""

    auditor = CognitiveAuditEngine()

    test_cases = [
        {
            "recommendation": "The data clearly shows that AI investment is the right move - our consultant confirms this, and everyone in the industry is doing it. We must act now or lose competitive advantage.",
            "context": {
                "stakeholders": ["CEO", "External Consultant", "CTO"],
                "stated_preferences": "Want to be seen as innovative",
                "timeline_pressure": True,
                "industry": "Financial Services",
            },
        },
        {
            "recommendation": "We've already invested $2M in this project and our team believes it will work. The research supports our approach and we can't give up now.",
            "context": {
                "stakeholders": ["Project Team", "CFO", "Board"],
                "financial_constraints": "Budget under scrutiny",
                "company": "TechCorp Inc",
            },
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*20} COGNITIVE AUDIT TEST {i} {'='*20}")

        result = await auditor.audit_motivated_reasoning(
            case["recommendation"], case["context"]
        )

        print("\n🎯 COGNITIVE AUDIT RESULTS:")
        print(f"Overall Bias Score: {result.overall_bias_score:.3f}")
        print(f"Intellectual Honesty: {result.intellectual_honesty_assessment:.3f}")
        print(f"Audit Confidence: {result.audit_confidence:.3f}")

        print("\n❓ KEY CLARIFYING QUESTIONS:")
        for question in result.clarifying_questions[:3]:
            print(f"• {question}")

        if result.critical_thinking_recommendations:
            print("\n💡 CRITICAL THINKING RECOMMENDATIONS:")
            for rec in result.critical_thinking_recommendations[:2]:
                print(f"• {rec}")

        if i < len(test_cases):
            print(f"\n{'NEXT TEST':=^80}\n")


if __name__ == "__main__":
    asyncio.run(demonstrate_cognitive_audit())
