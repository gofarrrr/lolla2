"""
METIS V5 Consultant Profile Data Contracts
Defines public-facing consultant profile structures for the Faculty Showcase system.
"""

from pydantic import BaseModel, Field
from typing import List


class ConsultantProfile(BaseModel):
    """
    Public-facing consultant profile for the Faculty Showcase system.
    This model defines marketing-safe content without exposing internal IP.
    """

    agent_id: str = Field(
        ..., description="e.g., 'market_analyst', 'problem_structuring_agent'"
    )
    name: str = Field(..., description="e.g., 'The Market Analyst'")
    tagline: str = Field(..., description="A short, compelling one-liner.")
    matrix_position: str = Field(
        ..., description="e.g., 'Strategic Analysis' or 'A Priori'"
    )
    ikigai_purpose: str = Field(
        ...,
        description="A detailed, public-facing description of the agent's core function and value.",
    )
    key_strengths: List[str] = Field(
        ..., description="A list of 3-5 key skills or capabilities."
    )
    typical_questions_it_answers: List[str] = Field(
        ..., description="A list of 2-3 sample questions this agent excels at."
    )
    logo_concept: str = Field(..., description="A textual description for a designer.")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "strategic_analyst",
                "name": "The Strategic Analyst",
                "tagline": "Transforming market complexity into strategic clarity",
                "matrix_position": "Strategic Analysis",
                "ikigai_purpose": "The Strategic Analyst excels at market analysis, competitive intelligence, and strategic planning. This agent brings deep analytical rigor to business strategy questions, combining quantitative analysis with strategic frameworks to deliver actionable insights for executive decision-making.",
                "key_strengths": [
                    "Market Analysis & Competitive Intelligence",
                    "Strategic Planning & Framework Development",
                    "Business Model Innovation",
                    "Executive-Level Synthesis",
                    "Quantitative Business Analysis",
                ],
                "typical_questions_it_answers": [
                    "How should we position our product against key competitors?",
                    "What market opportunities should we prioritize for expansion?",
                    "How can we strengthen our competitive advantage in this sector?",
                ],
                "logo_concept": "A compass rose overlaying a market chart, representing strategic navigation through complex business landscapes",
            }
        }
