from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path


class PresentationType(Enum):
    """Supported presentation types"""

    STRATEGY = "strategy"
    PROBLEM_SOLVING = "problem_solving"
    RESEARCH_REPORT = "research_report"
    EXECUTIVE_BRIEFING = "executive_briefing"
    MENTAL_MODEL_GUIDE = "mental_model_guide"
    DECISION_ANALYSIS = "decision_analysis"


class TemplateEngine:
    """
    Manages presentation templates for different use cases
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[PresentationType, Dict[str, Any]]:
        """Load all presentation templates"""
        templates = {}

        # Strategy presentation template
        templates[PresentationType.STRATEGY] = {
            "name": "Strategic Analysis",
            "theme": "Night Sky",
            "text_mode": "generate",
            "format": "presentation",
            "card_dimensions": "16x9",
            "instructions": "Create a professional strategy presentation with clear sections for problem analysis, strategic options, and implementation roadmap. Use executive-friendly language.",
            "text_options": {
                "amount": "detailed",
                "tone": "professional, strategic, confident",
                "audience": "C-suite executives, senior management",
            },
            "image_options": {
                "source": "aiGenerated",
                "model": "flux-1-pro",
                "style": "professional business, modern corporate, abstract strategic visuals",
            },
        }

        # Problem solving template
        templates[PresentationType.PROBLEM_SOLVING] = {
            "name": "Problem Resolution Framework",
            "theme": "Oasis",
            "text_mode": "generate",
            "format": "presentation",
            "instructions": "Structure the presentation as a clear problem-solving journey from issue identification through root cause analysis to solution implementation.",
            "text_options": {
                "amount": "medium",
                "tone": "analytical, solution-oriented, practical",
                "audience": "project managers, technical teams",
            },
            "image_options": {
                "source": "aiGenerated",
                "model": "imagen-4-pro",
                "style": "diagrams, flowcharts, problem-solving visuals",
            },
        }

        # Research report template
        templates[PresentationType.RESEARCH_REPORT] = {
            "name": "Research Insights Report",
            "theme": "Clarity",
            "text_mode": "condense",
            "format": "document",
            "card_dimensions": "letter",
            "instructions": "Create a comprehensive research document with methodology, findings, and recommendations. Include data visualizations where appropriate.",
            "text_options": {
                "amount": "extensive",
                "tone": "academic, thorough, evidence-based",
                "audience": "researchers, analysts, subject matter experts",
            },
            "image_options": {
                "source": "aiGenerated",
                "model": "flux-1-pro",
                "style": "data visualization, charts, research graphics",
            },
        }

        # Executive briefing template
        templates[PresentationType.EXECUTIVE_BRIEFING] = {
            "name": "Executive Brief",
            "theme": "Bold",
            "text_mode": "condense",
            "format": "presentation",
            "num_cards": 5,
            "instructions": "Create a concise executive briefing with key points, decisions required, and clear recommendations. Maximum 5 slides.",
            "text_options": {
                "amount": "brief",
                "tone": "executive, decisive, action-oriented",
                "audience": "board members, C-suite",
            },
            "image_options": {
                "source": "pictographic",
            },
        }

        # Mental model guide template
        templates[PresentationType.MENTAL_MODEL_GUIDE] = {
            "name": "Mental Model Application Guide",
            "theme": "Minimal",
            "text_mode": "generate",
            "format": "document",
            "instructions": "Create an educational guide explaining mental models with practical examples and application scenarios.",
            "text_options": {
                "amount": "detailed",
                "tone": "educational, clear, practical",
                "audience": "business professionals, continuous learners",
            },
            "image_options": {
                "source": "aiGenerated",
                "model": "imagen-4-pro",
                "style": "conceptual diagrams, educational illustrations, minimal design",
            },
        }

        # Decision analysis template
        templates[PresentationType.DECISION_ANALYSIS] = {
            "name": "Decision Framework Analysis",
            "theme": "Focus",
            "text_mode": "generate",
            "format": "presentation",
            "instructions": "Present decision options with pros/cons, risk analysis, and recommended path forward. Include decision criteria and scoring.",
            "text_options": {
                "amount": "detailed",
                "tone": "analytical, balanced, decisive",
                "audience": "decision makers, stakeholders",
            },
            "image_options": {
                "source": "aiGenerated",
                "model": "flux-1-pro",
                "style": "decision trees, comparison charts, risk matrices",
            },
        }

        return templates

    def get_template(self, presentation_type: PresentationType) -> Dict[str, Any]:
        """Get template for specific presentation type"""
        return self.templates.get(
            presentation_type, self.templates[PresentationType.STRATEGY]
        )

    def create_custom_template(
        self, name: str, base_type: PresentationType, overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create custom template based on existing one"""
        base_template = self.get_template(base_type).copy()
        base_template.update(overrides)
        base_template["name"] = name
        return base_template

    def get_available_themes(self) -> list:
        """Get list of available Gamma themes"""
        return [
            "Night Sky",
            "Oasis",
            "Clarity",
            "Bold",
            "Minimal",
            "Focus",
            "Atlas",
            "Bauhaus",
            "Corporate",
            "Academic",
        ]

    def get_template_description(self, presentation_type: PresentationType) -> str:
        """Get template description"""
        descriptions = {
            PresentationType.STRATEGY: "Comprehensive strategic analysis with implementation roadmap",
            PresentationType.PROBLEM_SOLVING: "Structured problem resolution framework",
            PresentationType.RESEARCH_REPORT: "Detailed research findings and insights",
            PresentationType.EXECUTIVE_BRIEFING: "Concise executive-level summary",
            PresentationType.MENTAL_MODEL_GUIDE: "Educational guide on mental model application",
            PresentationType.DECISION_ANALYSIS: "Decision framework with options analysis",
        }
        return descriptions.get(presentation_type, "Professional presentation template")
