"""
Context Compiler - First-class context engineering discipline
Based on Manus principles: treat context as a compiled artifact
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Optional dependency: tiktoken. Provide a fallback if unavailable.
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # Fallback handled in class init

logger = logging.getLogger(__name__)


@dataclass
class CompiledContext:
    """
    A compiled context ready for LLM consumption.
    Tracks provenance, compression ratio, and relevance scores.
    """

    raw_size: int  # Original token count
    compiled_size: int  # After compilation
    compression_ratio: float
    relevance_score: float
    structured_data: Dict[str, Any]
    provenance: List[str]  # Where each piece came from
    tokens_saved: int = 0  # V5.4 compatibility - tokens saved by compression
    schema_version: str = "1.0"


class StageContextCompiler(ABC):
    """
    Base class for stage-specific context compilers.
    Each stage gets a custom compiler optimized for its needs.
    """

    def __init__(self, stage_name: str, token_limit: int = 4000):
        self.stage_name = stage_name
        self.token_limit = token_limit
        # Initialize tokenizer with graceful fallback
        if tiktoken is not None:
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            except Exception as e:  # pragma: no cover
                logger.warning(
                    f"Failed to load GPT-4 tokenizer: {e}, using cl100k_base"
                )
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                except Exception as e2:
                    logger.warning(
                        f"Failed to load cl100k_base tokenizer: {e2}, using naive tokenizer"
                    )
                    self.tokenizer = None
        else:
            self.tokenizer = None

    @abstractmethod
    def select(self, raw_context: Dict[str, Any]) -> Dict[str, Any]:
        """Select relevant information for this stage"""
        pass

    @abstractmethod
    def compress(self, selected: Dict[str, Any]) -> Dict[str, Any]:
        """Compress selected information"""
        pass

    @abstractmethod
    def structure(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Structure for optimal LLM consumption"""
        pass

    @abstractmethod
    def sequence(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """Sequence information by importance"""
        pass

    def compile(self, raw_context: Dict[str, Any]) -> CompiledContext:
        """
        Full compilation pipeline: select -> compress -> structure -> sequence
        This is the Manus principle in action.
        """
        raw_size = self._count_tokens(raw_context)

        # Four-stage compilation
        selected = self.select(raw_context)
        compressed = self.compress(selected)
        structured = self.structure(compressed)
        sequenced = self.sequence(structured)

        compiled_size = self._count_tokens(sequenced)

        tokens_saved = max(0, raw_size - compiled_size)
        return CompiledContext(
            raw_size=raw_size,
            compiled_size=compiled_size,
            compression_ratio=1 - (compiled_size / raw_size) if raw_size > 0 else 0,
            relevance_score=self._calculate_relevance(sequenced, raw_context),
            structured_data=sequenced,
            tokens_saved=tokens_saved,
            provenance=self._extract_provenance(sequenced),
        )

    def _count_tokens(self, data: Any) -> int:
        """Count tokens in data structure"""
        # Use tiktoken if available; otherwise approximate by word count
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(str(data)))
        # Fallback: crude token estimate by splitting on whitespace
        return max(1, len(str(data).split()))

    def _calculate_relevance(
        self, compiled: Dict[str, Any], original: Dict[str, Any]
    ) -> float:
        """Calculate relevance score of compiled context"""
        if not original:
            return 0.0

        # Heuristic: proportion of original salient values retained in compiled output
        compiled_str = str(compiled)
        total_items = 0
        retained = 0
        try:
            # Common salient fields
            for key in ["problem_statement", "constraints", "focus_areas"]:
                if key in original:
                    value = original[key]
                    if isinstance(value, str):
                        total_items += 1
                        if value and value in compiled_str:
                            retained += 1
                    elif isinstance(value, list):
                        for item in value:
                            total_items += 1
                            if str(item) in compiled_str:
                                retained += 1
            # Avoid division by zero
            if total_items == 0:
                return 1.0 if compiled_str else 0.0
            score = retained / total_items
            # Clamp and slightly favor retention
            return max(0.0, min(1.0, score))
        except Exception:  # pragma: no cover
            # Fallback to string overlap if something goes wrong
            original_tokens = set(str(original).lower().split())
            compiled_tokens = set(compiled_str.lower().split())
            if not original_tokens:
                return 1.0
            overlap = len(original_tokens & compiled_tokens)
            return min(1.0, overlap / len(original_tokens))

    def _extract_provenance(self, data: Dict[str, Any]) -> List[str]:
        """Extract provenance information from compiled context"""
        provenance = []

        def extract_sources(obj, path="root"):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}"
                    if key in ["source", "provenance", "origin"]:
                        provenance.append(f"{new_path}: {value}")
                    else:
                        extract_sources(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_sources(item, f"{path}[{i}]")

        extract_sources(data)
        return provenance


class SocraticEngineCompiler(StageContextCompiler):
    """Context compiler for Socratic Engine stage"""

    def select(self, raw_context: Dict[str, Any]) -> Dict[str, Any]:
        """Select only problem statement and constraints"""
        return {
            "problem": raw_context.get("problem_statement"),
            "constraints": raw_context.get("constraints", []),
            "focus_areas": raw_context.get("focus_areas", []),
            # Deliberately exclude accumulated_insights to prevent bias
        }

    def compress(self, selected: Dict[str, Any]) -> Dict[str, Any]:
        """Compress by removing redundancy"""
        compressed = selected.copy()

        # Remove duplicate constraints
        if "constraints" in compressed and isinstance(compressed["constraints"], list):
            compressed["constraints"] = list(set(compressed["constraints"]))

        # Remove duplicate focus areas
        if "focus_areas" in compressed and isinstance(compressed["focus_areas"], list):
            compressed["focus_areas"] = list(set(compressed["focus_areas"]))

        return compressed

    def structure(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Structure for Socratic questioning"""
        return {
            "CORE_PROBLEM": compressed.get("problem", ""),
            "HARD_CONSTRAINTS": compressed.get("constraints", []),
            "EXPLORATION_AREAS": compressed.get("focus_areas", []),
            "QUESTION_CATEGORIES": [
                "assumptions",
                "alternatives",
                "consequences",
                "evidence",
            ],
        }

    def sequence(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """Sequence by importance for question generation"""
        return {
            "PRIMARY": structured.get("CORE_PROBLEM", ""),
            "SECONDARY": {
                "constraints": structured.get("HARD_CONSTRAINTS", []),
                "areas": structured.get("EXPLORATION_AREAS", []),
            },
            "GUIDANCE": structured.get("QUESTION_CATEGORIES", []),
        }


class ProblemStructuringCompiler(StageContextCompiler):
    """Context compiler for Problem Structuring stage"""

    def select(self, raw_context: Dict[str, Any]) -> Dict[str, Any]:
        """Select problem statement, questions, and initial insights"""
        return {
            "problem": raw_context.get("problem_statement"),
            "socratic_questions": raw_context.get("socratic_questions", []),
            "initial_insights": raw_context.get("initial_insights", []),
            "constraints": raw_context.get("constraints", []),
        }

    def compress(self, selected: Dict[str, Any]) -> Dict[str, Any]:
        """Compress by summarizing questions and insights"""
        compressed = selected.copy()

        # Limit number of questions
        if (
            "socratic_questions" in compressed
            and len(compressed["socratic_questions"]) > 5
        ):
            compressed["socratic_questions"] = compressed["socratic_questions"][:5]

        # Summarize insights if too many
        if "initial_insights" in compressed and len(compressed["initial_insights"]) > 3:
            compressed["initial_insights"] = compressed["initial_insights"][:3]

        return compressed

    def structure(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Structure for problem decomposition"""
        return {
            "PROBLEM_DEFINITION": compressed.get("problem", ""),
            "KEY_QUESTIONS": compressed.get("socratic_questions", []),
            "EMERGING_INSIGHTS": compressed.get("initial_insights", []),
            "CONSTRAINTS_TO_CONSIDER": compressed.get("constraints", []),
            "STRUCTURING_FRAMEWORKS": ["MECE", "Issue Tree", "Hypothesis-Driven"],
        }

    def sequence(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """Sequence for systematic problem breakdown"""
        return {
            "FOUNDATION": {
                "problem": structured.get("PROBLEM_DEFINITION", ""),
                "constraints": structured.get("CONSTRAINTS_TO_CONSIDER", []),
            },
            "EXPLORATION": {
                "questions": structured.get("KEY_QUESTIONS", []),
                "insights": structured.get("EMERGING_INSIGHTS", []),
            },
            "METHODOLOGY": structured.get("STRUCTURING_FRAMEWORKS", []),
        }


class ConsultantSelectionCompiler(StageContextCompiler):
    """Context compiler for Consultant Selection stage"""

    def select(self, raw_context: Dict[str, Any]) -> Dict[str, Any]:
        """Select structured problem and relevant metadata"""
        return {
            "structured_problem": raw_context.get("structured_problem", {}),
            "problem_complexity": raw_context.get("problem_complexity", "medium"),
            "domains_identified": raw_context.get("domains_identified", []),
            "required_expertise": raw_context.get("required_expertise", []),
        }

    def compress(self, selected: Dict[str, Any]) -> Dict[str, Any]:
        """Compress by focusing on key selection criteria"""
        compressed = selected.copy()

        # Extract only essential problem aspects
        if "structured_problem" in compressed and isinstance(
            compressed["structured_problem"], dict
        ):
            compressed["structured_problem"] = {
                "core": compressed["structured_problem"].get("core_issue", ""),
                "type": compressed["structured_problem"].get(
                    "problem_type", "analysis"
                ),
            }

        return compressed

    def structure(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Structure for consultant matching"""
        return {
            "PROBLEM_PROFILE": compressed.get("structured_problem", {}),
            "COMPLEXITY_LEVEL": compressed.get("problem_complexity", "medium"),
            "DOMAIN_REQUIREMENTS": compressed.get("domains_identified", []),
            "EXPERTISE_NEEDED": compressed.get("required_expertise", []),
            "SELECTION_CRITERIA": ["relevance", "diversity", "complementarity"],
        }

    def sequence(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """Sequence for optimal consultant selection"""
        return {
            "REQUIREMENTS": {
                "complexity": structured.get("COMPLEXITY_LEVEL", "medium"),
                "domains": structured.get("DOMAIN_REQUIREMENTS", []),
                "expertise": structured.get("EXPERTISE_NEEDED", []),
            },
            "PROBLEM_CONTEXT": structured.get("PROBLEM_PROFILE", {}),
            "SELECTION_GUIDANCE": structured.get("SELECTION_CRITERIA", []),
        }


class ParallelAnalysisCompiler(StageContextCompiler):
    """Context compiler for Parallel Analysis stage"""

    def select(self, raw_context: Dict[str, Any]) -> Dict[str, Any]:
        """Select consultant-specific relevant context"""
        return {
            "problem_statement": raw_context.get("problem_statement"),
            "consultant_role": raw_context.get("consultant_role", "analyst"),
            "specific_focus": raw_context.get("specific_focus", []),
            "prior_insights": raw_context.get("relevant_prior_insights", []),
            "constraints": raw_context.get("constraints", []),
        }

    def compress(self, selected: Dict[str, Any]) -> Dict[str, Any]:
        """Compress by filtering to consultant's domain"""
        compressed = selected.copy()

        # Keep only insights relevant to this consultant's role
        if "prior_insights" in compressed and compressed.get("consultant_role"):
            role = compressed["consultant_role"].lower()
            filtered_insights = []
            for insight in compressed.get("prior_insights", []):
                if isinstance(insight, dict) and role in str(insight).lower():
                    filtered_insights.append(insight)
            compressed["prior_insights"] = filtered_insights[
                :2
            ]  # Limit to 2 most relevant

        return compressed

    def structure(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Structure for focused consultant analysis"""
        return {
            "ANALYSIS_MANDATE": {
                "problem": compressed.get("problem_statement", ""),
                "your_role": compressed.get("consultant_role", "analyst"),
                "focus_areas": compressed.get("specific_focus", []),
            },
            "CONTEXT": {
                "constraints": compressed.get("constraints", []),
                "prior_work": compressed.get("prior_insights", []),
            },
            "ANALYSIS_FRAMEWORK": {
                "approach": "systematic",
                "depth": "comprehensive",
                "perspective": compressed.get("consultant_role", "general"),
            },
        }

    def sequence(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """Sequence for consultant execution"""
        return {
            "DIRECTIVE": structured.get("ANALYSIS_MANDATE", {}),
            "BACKGROUND": structured.get("CONTEXT", {}),
            "METHODOLOGY": structured.get("ANALYSIS_FRAMEWORK", {}),
        }


class DevilsAdvocateCompiler(StageContextCompiler):
    """Context compiler for Devil's Advocate stage"""

    def select(self, raw_context: Dict[str, Any]) -> Dict[str, Any]:
        """Select all consultant analyses and key recommendations"""
        return {
            "consultant_analyses": raw_context.get("consultant_analyses", []),
            "key_recommendations": raw_context.get("key_recommendations", []),
            "consensus_points": raw_context.get("consensus_points", []),
            "divergent_views": raw_context.get("divergent_views", []),
            "original_problem": raw_context.get("problem_statement"),
        }

    def compress(self, selected: Dict[str, Any]) -> Dict[str, Any]:
        """Compress by extracting critical claims"""
        compressed = selected.copy()

        # Extract main claims from each analysis
        if "consultant_analyses" in compressed:
            compressed_analyses = []
            for analysis in compressed.get("consultant_analyses", []):
                if isinstance(analysis, dict):
                    compressed_analyses.append(
                        {
                            "consultant": analysis.get("consultant_id", "unknown"),
                            "main_claim": analysis.get("main_recommendation", ""),
                            "confidence": analysis.get("confidence_level", 0.5),
                        }
                    )
            compressed["consultant_analyses"] = compressed_analyses

        return compressed

    def structure(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Structure for critical examination"""
        return {
            "EXAMINATION_TARGET": {
                "problem": compressed.get("original_problem", ""),
                "proposed_solutions": compressed.get("key_recommendations", []),
                "analyses_to_challenge": compressed.get("consultant_analyses", []),
            },
            "IDENTIFIED_PATTERNS": {
                "consensus": compressed.get("consensus_points", []),
                "disagreements": compressed.get("divergent_views", []),
            },
            "CHALLENGE_FRAMEWORK": {
                "bias_types": [
                    "confirmation",
                    "anchoring",
                    "availability",
                    "groupthink",
                ],
                "assumption_levels": ["stated", "implicit", "paradigmatic"],
                "challenge_depth": "systematic",
            },
        }

    def sequence(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """Sequence for systematic challenge"""
        return {
            "TARGET": structured.get("EXAMINATION_TARGET", {}),
            "PATTERNS": structured.get("IDENTIFIED_PATTERNS", {}),
            "METHODOLOGY": structured.get("CHALLENGE_FRAMEWORK", {}),
        }


class SeniorAdvisorCompiler(StageContextCompiler):
    """Context compiler for Senior Advisor stage"""

    def select(self, raw_context: Dict[str, Any]) -> Dict[str, Any]:
        """Select synthesis-ready insights and critiques"""
        return {
            "problem_statement": raw_context.get("problem_statement"),
            "all_analyses": raw_context.get("consultant_analyses", []),
            "devils_advocate_critique": raw_context.get("devils_advocate_critique", {}),
            "key_recommendations": raw_context.get("key_recommendations", []),
            "implementation_considerations": raw_context.get(
                "implementation_considerations", []
            ),
        }

    def compress(self, selected: Dict[str, Any]) -> Dict[str, Any]:
        """Compress by extracting decision-critical information"""
        compressed = selected.copy()

        # Summarize analyses to key points
        if "all_analyses" in compressed:
            summary = []
            for analysis in compressed.get("all_analyses", []):
                if isinstance(analysis, dict):
                    summary.append(
                        {
                            "source": analysis.get("consultant_id", "unknown"),
                            "key_point": analysis.get("main_recommendation", ""),
                            "strength": analysis.get("evidence_strength", "medium"),
                        }
                    )
            compressed["all_analyses"] = summary[:5]  # Top 5 most important

        return compressed

    def structure(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Structure for executive synthesis"""
        return {
            "DECISION_CONTEXT": {
                "problem": compressed.get("problem_statement", ""),
                "stakes": "high",
                "timeframe": "strategic",
            },
            "SYNTHESIS_INPUTS": {
                "analyses": compressed.get("all_analyses", []),
                "critiques": compressed.get("devils_advocate_critique", {}),
                "recommendations": compressed.get("key_recommendations", []),
            },
            "INTEGRATION_FRAMEWORK": {
                "approach": "balanced synthesis",
                "priorities": ["feasibility", "impact", "risk"],
                "considerations": compressed.get("implementation_considerations", []),
            },
        }

    def sequence(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """Sequence for strategic recommendation"""
        return {
            "CONTEXT": structured.get("DECISION_CONTEXT", {}),
            "INPUTS": structured.get("SYNTHESIS_INPUTS", {}),
            "FRAMEWORK": structured.get("INTEGRATION_FRAMEWORK", {}),
        }


# Factory function to get appropriate compiler
def get_stage_compiler(
    stage_name: str, token_limit: int = 4000
) -> StageContextCompiler:
    """Factory function to get the appropriate compiler for a stage"""
    compilers = {
        "socratic": SocraticEngineCompiler,
        "problem_structuring": ProblemStructuringCompiler,
        "consultant_selection": ConsultantSelectionCompiler,
        "parallel_analysis": ParallelAnalysisCompiler,
        "devils_advocate": DevilsAdvocateCompiler,
        "senior_advisor": SeniorAdvisorCompiler,
    }

    compiler_class = compilers.get(stage_name.lower())
    if not compiler_class:
        raise ValueError(f"Unknown stage: {stage_name}")

    return compiler_class(stage_name=stage_name, token_limit=token_limit)
