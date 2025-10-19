"""
Enhancement Research Stage
==========================

Extracts enhancement research questions and answers.

Responsibility:
- Extract enhancement_context from senior_advisor stage
- Extract research_questions and answered_questions
- Merge questions with answers using question_id
- Generate answer previews (first 150 chars)
- Build frontend-ready format

Complexity: CC<5 (Simple extraction and merging)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.services.report_reconstruction.reconstruction_state import ReconstructionState
from src.services.report_reconstruction.reconstruction_stage import (
    ReconstructionStage,
    ReconstructionError,
)

logger = logging.getLogger(__name__)


class EnhancementResearchStage(ReconstructionStage):
    """
    Stage 4: Enhancement Research

    Extracts enhancement research questions and answers from enhancement_context.
    Merges them into a frontend-ready format.
    """

    @property
    def name(self) -> str:
        return "enhancement_research"

    @property
    def description(self) -> str:
        return "Extract enhancement research questions and answers"

    def process(self, state: ReconstructionState) -> ReconstructionState:
        """
        Extract enhancement research from senior_advisor.

        Args:
            state: Current reconstruction state with senior_advisor extracted

        Returns:
            Updated state with enhancement_research_answers populated
        """
        try:
            # Extract enhancement context from senior_advisor
            enhancement_context = self._extract_enhancement_context(state.senior_advisor)

            # Extract research questions and answers
            research_questions = enhancement_context.get("research_questions", [])
            answered_questions = enhancement_context.get("answered_questions", [])

            # Build lookup map from answered_questions
            answers_lookup = self._build_answers_lookup(answered_questions)

            # Merge questions with answers
            merged = self._merge_questions_with_answers(
                research_questions, answers_lookup
            )

            return state.with_research(enhancement_research_answers=merged)

        except Exception as e:
            raise ReconstructionError(
                self.name,
                f"Failed to extract enhancement research for trace_id={state.trace_id}",
                cause=e,
            )

    def _extract_enhancement_context(
        self, senior_advisor: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract enhancement_context from senior_advisor stage."""
        return senior_advisor.get("enhancement_context", {}) or {}

    def _build_answers_lookup(
        self, answered_questions: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Build lookup map from answered_questions.

        Args:
            answered_questions: List of answered questions with question_id and answer

        Returns:
            Dictionary mapping question_id -> answer text
        """
        answers_lookup = {}

        for aq in answered_questions:
            question_id = aq.get("question_id")
            if question_id:
                # V6: Try answer_summary first (shorter version), then full answer
                answer_text = aq.get("answer_summary") or aq.get("answer", "")
                answers_lookup[question_id] = answer_text

        return answers_lookup

    def _merge_questions_with_answers(
        self, research_questions: List[Dict[str, Any]], answers_lookup: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Merge research questions with their answers.

        Args:
            research_questions: List of research questions
            answers_lookup: Dictionary mapping question_id -> answer

        Returns:
            List of merged question/answer pairs in frontend format
        """
        merged = []

        for rq in research_questions:
            question_id = rq.get("question_id", "")
            question_text = rq.get("question_text", "")
            answer = answers_lookup.get(question_id, "Research in progress...")

            # Generate preview (first 150 chars of answer)
            answer_preview = answer[:150] + "..." if len(answer) > 150 else answer

            merged.append(
                {
                    "id": question_id,
                    "question": question_text,
                    "answer_preview": answer_preview,
                    "confidence": 0.85,  # TODO: Extract from research provider metadata
                    "citations_count": 0,  # TODO: Extract from research provider metadata
                }
            )

        return merged
