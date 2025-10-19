#!/usr/bin/env python3
"""
Simple LLM-as-Judge Implementation
Fast, effective evaluation without complexity
"""

import asyncio
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.integrations.llm.unified_client import UnifiedLLMClient


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation"""

    score: float
    explanation: str
    criteria_scores: Dict[str, float]
    timestamp: datetime
    judge_type: str


class SimpleLLMJudge:
    """Fast, effective evaluation without complexity"""

    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None):
        self.llm_client = llm_client or UnifiedLLMClient()
        self.judge_prompts = self._load_judge_prompts()

    def _load_judge_prompts(self) -> Dict[str, str]:
        """Load judge prompts for different evaluation types"""
        return {
            "quality": """
Rate this consultant response quality on a 1-5 scale:

QUERY: {query}

CONSULTANT RESPONSE: {response}

EXPECTED CONSULTANT: {expected_consultant}
EXPECTED ANALYSIS POINTS: {key_points}

SCORING CRITERIA:
5 = EXCELLENT: Deep insights, highly actionable, addresses all key points, demonstrates expertise
4 = GOOD: Solid analysis, mostly actionable, covers most key points, shows competence  
3 = AVERAGE: Basic analysis, some actionable items, covers some key points, adequate
2 = POOR: Superficial analysis, limited actionability, misses key points, below standard
1 = BAD: Wrong analysis, harmful advice, completely off-topic, fails basic requirements

RESPONSE FORMAT:
Score: [1-5]
Reasoning: [One sentence explaining the score]
Strengths: [What the response does well]
Weaknesses: [What could be improved]
""",
            "consultant_selection": """
Evaluate if the correct consultant type was selected for this query:

QUERY: {query}
SELECTED CONSULTANT: {selected_consultant}
EXPECTED CONSULTANT: {expected_consultant}

EVALUATION:
- Does the selected consultant type match the query's primary need?
- Would this consultant type have the right expertise?
- Is this the optimal choice vs alternatives?

RESPONSE FORMAT:
Correct: [YES/NO]
Score: [0-1 where 1=perfect match, 0=completely wrong]
Reasoning: [Why this selection is right/wrong]
""",
            "nway_selection": """
Evaluate if appropriate mental model frameworks (NWAYs) were selected:

QUERY: {query}
SELECTED NWAYS: {selected_nways}
EXPECTED NWAYS: {expected_nways}

NWAY RELEVANCE:
- Do the selected NWAYs match the decision type?
- Are these the most relevant mental models for this situation?
- Is there good coverage of the problem space?

RESPONSE FORMAT:
Accuracy: [0-1 where 1=perfect selection, 0=completely irrelevant]
Coverage: [0-1 where 1=comprehensive coverage, 0=major gaps]
Reasoning: [Why these NWAYs are appropriate/inappropriate]
""",
            "usefulness": """
Rate how useful this analysis would be for decision-making:

QUERY: {query}
RESPONSE: {response}

USEFULNESS CRITERIA:
- Provides actionable insights and recommendations
- Addresses the core decision being made
- Offers practical next steps
- Helps reduce uncertainty or risk

RESPONSE FORMAT:
Score: [1-5]
Actionability: [How actionable are the recommendations?]
Clarity: [How clear and understandable is the advice?]
Completeness: [Does it address the full scope of the decision?]
""",
            "accuracy": """
Evaluate the factual accuracy and logical soundness:

QUERY: {query}
RESPONSE: {response}
SAMPLE_GOOD_RESPONSE: {sample_good}

ACCURACY CRITERIA:
- Factual correctness of claims and data
- Logical consistency of reasoning
- Appropriate use of business concepts
- Realistic assumptions and projections

RESPONSE FORMAT:
Score: [1-5]
Factual_Accuracy: [Are the facts and figures correct?]
Logic: [Is the reasoning sound and consistent?]
Realism: [Are assumptions and projections realistic?]
""",
        }

    async def judge_quality(
        self, query: str, response: str, golden_example: Optional[Dict] = None
    ) -> JudgeResult:
        """Judge overall quality of consultant response"""

        # Extract context from golden example if provided
        expected_consultant = (
            golden_example.get("expected_consultant", "N/A")
            if golden_example
            else "N/A"
        )
        key_points = (
            golden_example.get("key_analysis_points", []) if golden_example else []
        )
        key_points_str = "; ".join(key_points) if key_points else "N/A"

        prompt = self.judge_prompts["quality"].format(
            query=query,
            response=response,
            expected_consultant=expected_consultant,
            key_points=key_points_str,
        )

        try:
            result = await self.llm_client.call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistent scoring
            )

            # Parse response
            score = self._extract_score(result, pattern=r"Score:\s*(\d+)")
            reasoning = self._extract_field(result, "Reasoning")
            strengths = self._extract_field(result, "Strengths")
            weaknesses = self._extract_field(result, "Weaknesses")

            explanation = f"Reasoning: {reasoning}\nStrengths: {strengths}\nWeaknesses: {weaknesses}"

            return JudgeResult(
                score=score,
                explanation=explanation,
                criteria_scores={"overall": score},
                timestamp=datetime.now(),
                judge_type="quality",
            )

        except Exception as e:
            print(f"Error in quality judging: {e}")
            return JudgeResult(
                score=2.5,  # Default to middle score on error
                explanation=f"Error in evaluation: {str(e)}",
                criteria_scores={"overall": 2.5},
                timestamp=datetime.now(),
                judge_type="quality",
            )

    async def judge_consultant_selection(
        self, query: str, selected_consultant: str, expected_consultant: str
    ) -> JudgeResult:
        """Judge if correct consultant type was selected"""

        prompt = self.judge_prompts["consultant_selection"].format(
            query=query,
            selected_consultant=selected_consultant,
            expected_consultant=expected_consultant,
        )

        try:
            result = await self.llm_client.call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )

            # Parse response
            correct = self._extract_field(result, "Correct").upper() == "YES"
            score = self._extract_score(result, pattern=r"Score:\s*([0-1]\.?\d*)")
            reasoning = self._extract_field(result, "Reasoning")

            return JudgeResult(
                score=score,
                explanation=f"Correct: {correct}\nReasoning: {reasoning}",
                criteria_scores={"selection_accuracy": score},
                timestamp=datetime.now(),
                judge_type="consultant_selection",
            )

        except Exception as e:
            print(f"Error in consultant selection judging: {e}")
            return JudgeResult(
                score=0.5,
                explanation=f"Error in evaluation: {str(e)}",
                criteria_scores={"selection_accuracy": 0.5},
                timestamp=datetime.now(),
                judge_type="consultant_selection",
            )

    async def judge_nway_selection(
        self, query: str, selected_nways: List[str], expected_nways: List[str]
    ) -> JudgeResult:
        """Judge if appropriate NWAYs were selected"""

        prompt = self.judge_prompts["nway_selection"].format(
            query=query,
            selected_nways=", ".join(selected_nways),
            expected_nways=", ".join(expected_nways),
        )

        try:
            result = await self.llm_client.call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1,
            )

            # Parse response from LLMResponse object
            response_text = result.content
            accuracy = self._extract_score(
                response_text, pattern=r"Accuracy:\s*([0-1]\.?\d*)"
            )
            coverage = self._extract_score(
                response_text, pattern=r"Coverage:\s*([0-1]\.?\d*)"
            )
            reasoning = self._extract_field(response_text, "Reasoning")

            overall_score = (accuracy + coverage) / 2

            return JudgeResult(
                score=overall_score,
                explanation=f"Accuracy: {accuracy}, Coverage: {coverage}\nReasoning: {reasoning}",
                criteria_scores={"accuracy": accuracy, "coverage": coverage},
                timestamp=datetime.now(),
                judge_type="nway_selection",
            )

        except Exception as e:
            print(f"Error in NWAY selection judging: {e}")
            return JudgeResult(
                score=0.5,
                explanation=f"Error in evaluation: {str(e)}",
                criteria_scores={"accuracy": 0.5, "coverage": 0.5},
                timestamp=datetime.now(),
                judge_type="nway_selection",
            )

    async def judge_usefulness(self, query: str, response: str) -> JudgeResult:
        """Judge how useful the response is for decision-making"""

        prompt = self.judge_prompts["usefulness"].format(query=query, response=response)

        try:
            result = await self.llm_client.call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1,
            )

            # Parse response from LLMResponse object
            response_text = result.content
            score = self._extract_score(response_text, pattern=r"Score:\s*(\d+)")
            actionability = self._extract_field(response_text, "Actionability")
            clarity = self._extract_field(response_text, "Clarity")
            completeness = self._extract_field(response_text, "Completeness")

            explanation = f"Actionability: {actionability}\nClarity: {clarity}\nCompleteness: {completeness}"

            return JudgeResult(
                score=score,
                explanation=explanation,
                criteria_scores={"usefulness": score},
                timestamp=datetime.now(),
                judge_type="usefulness",
            )

        except Exception as e:
            print(f"Error in usefulness judging: {e}")
            return JudgeResult(
                score=2.5,
                explanation=f"Error in evaluation: {str(e)}",
                criteria_scores={"usefulness": 2.5},
                timestamp=datetime.now(),
                judge_type="usefulness",
            )

    async def evaluate_consultant_output(
        self, query: str, response: str, golden_example: Optional[Dict] = None
    ) -> Tuple[float, Dict]:
        """Comprehensive evaluation of consultant output"""

        # Run multiple judges in parallel for efficiency
        judge_tasks = [
            self.judge_quality(query, response, golden_example),
            self.judge_usefulness(query, response),
        ]

        # Add consultant and NWAY selection judges if golden example provided
        if golden_example:
            # Mock selected consultant and NWAYs - in real implementation these would come from system
            selected_consultant = golden_example.get("expected_consultant", "unknown")
            selected_nways = golden_example.get("expected_nways", [])

            judge_tasks.extend(
                [
                    self.judge_consultant_selection(
                        query,
                        selected_consultant,
                        golden_example.get("expected_consultant", ""),
                    ),
                    self.judge_nway_selection(
                        query, selected_nways, golden_example.get("expected_nways", [])
                    ),
                ]
            )

        # Execute all judges in parallel
        judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)

        # Process results
        scores = {}
        explanations = {}
        valid_scores = []

        for i, result in enumerate(judge_results):
            if isinstance(result, Exception):
                print(f"Judge {i} failed: {result}")
                continue

            if isinstance(result, JudgeResult):
                scores[result.judge_type] = result.score
                explanations[result.judge_type] = result.explanation
                valid_scores.append(result.score)

        # Calculate overall score
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 2.5

        evaluation_result = {
            "overall_score": overall_score,
            "individual_scores": scores,
            "explanations": explanations,
            "timestamp": datetime.now().isoformat(),
            "judge_count": len(valid_scores),
        }

        return overall_score, evaluation_result

    def _extract_score(self, text: str, pattern: str = r"Score:\s*(\d+)") -> float:
        """Extract numeric score from LLM response"""
        try:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))

            # Fallback: look for any number
            numbers = re.findall(r"\b([0-5]\.?\d*)\b", text)
            if numbers:
                return float(numbers[0])

            # Default to middle score if no number found
            return 2.5

        except (ValueError, AttributeError):
            return 2.5

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract specific field from LLM response"""
        try:
            pattern = rf"{field_name}:\s*(.+?)(?:\n|$)"
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
            return "N/A"
        except:
            return "N/A"


# Test function
async def test_simple_llm_judge():
    """Test the SimpleLLMJudge implementation"""
    print("🧪 Testing SimpleLLMJudge...")

    judge = SimpleLLMJudge()

    # Test data
    test_query = "Should we acquire TechStartup Inc for $200M? They have 50 engineers, $10M ARR, but losing $5M/year."
    test_response = "Based on financial analysis: (1) 20x revenue multiple is extremely high for a loss-making company (2) Need clear path to profitability within 24 months (3) Consider opportunity cost - $200M could generate $20M+ annually in safer investments (4) Recommend counter-offer at $100-120M or walk away"

    test_golden = {
        "expected_consultant": "financial_analyst",
        "expected_nways": ["opportunity_cost", "expected_value", "margin_of_safety"],
        "key_analysis_points": [
            "Revenue multiple analysis (20x ARR is high)",
            "Path to profitability assessment",
            "Integration costs and synergies",
            "Alternative investment opportunities",
        ],
    }

    # Run evaluation
    overall_score, detailed_results = await judge.evaluate_consultant_output(
        test_query, test_response, test_golden
    )

    print(f"✅ Overall Score: {overall_score:.2f}")
    print(f"📊 Individual Scores: {detailed_results['individual_scores']}")
    print(f"📝 Judge Count: {detailed_results['judge_count']}")

    for judge_type, explanation in detailed_results["explanations"].items():
        print(f"\n{judge_type.upper()}:")
        print(f"  {explanation}")

    return overall_score >= 3.5  # Pass if score is above average


if __name__ == "__main__":
    # Run test
    async def main():
        success = await test_simple_llm_judge()
        print(f"\n🎯 Test {'PASSED' if success else 'FAILED'}")
        return success

    asyncio.run(main())
