from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .templates import TemplateEngine, PresentationType

logger = logging.getLogger(__name__)


class PresentationBuilder:
    """
    Builds presentation content from METIS analysis outputs
    Handles formatting, structuring, and optimization for Gamma API
    """

    def __init__(self, template_engine: Optional[TemplateEngine] = None):
        self.template_engine = template_engine or TemplateEngine()

    def build_from_analysis(
        self,
        analysis_result: Dict[str, Any],
        presentation_type: PresentationType = PresentationType.STRATEGY,
    ) -> Dict[str, Any]:
        """
        Convert METIS analysis to Gamma-ready content

        Args:
            analysis_result: METIS cognitive analysis output
            presentation_type: Type of presentation to generate

        Returns:
            Formatted content ready for Gamma API
        """

        logger.info(f"📝 Building {presentation_type.value} presentation...")

        # Extract key components from analysis
        problem_statement = analysis_result.get("problem_statement", "")
        mental_models = analysis_result.get("recommended_mental_models", [])
        cognitive_framework = analysis_result.get("cognitive_framework", {})
        implementation = analysis_result.get("implementation_strategy", {})
        analysis_id = analysis_result.get("analysis_id", "unknown")

        # Handle different problem statement formats
        if isinstance(problem_statement, dict):
            problem_text = problem_statement.get(
                "problem_description", str(problem_statement)
            )
        else:
            problem_text = str(problem_statement)

        # Get appropriate template
        template = self.template_engine.get_template(presentation_type)

        # Build structured content
        content_structure = self._structure_content(
            problem_text,
            mental_models,
            cognitive_framework,
            implementation,
            template,
            analysis_id,
        )

        # Format for Gamma API
        gamma_content = self._format_for_gamma(content_structure, template)

        logger.info(f"✅ Built presentation with {len(content_structure)} sections")
        return gamma_content

    def _structure_content(
        self,
        problem: str,
        models: List[Dict],
        framework: Dict,
        implementation: Dict,
        template: Dict[str, Any],
        analysis_id: str,
    ) -> List[Dict[str, Any]]:
        """Structure content into presentation sections"""

        sections = []

        # Title slide
        title = self._extract_title_from_problem(problem)
        sections.append(
            {
                "type": "title",
                "content": f"# {title}\n\n*Strategic Analysis & Recommendations*\n\n{datetime.now().strftime('%B %Y')}\n\n*Powered by METIS Cognitive Intelligence*",
            }
        )

        # Executive summary
        executive_summary = self._build_executive_summary(
            framework, models, implementation
        )
        if executive_summary:
            sections.append(
                {
                    "type": "summary",
                    "content": f"# Executive Summary\n\n{executive_summary}",
                }
            )

        # Problem analysis
        sections.append(
            {
                "type": "problem",
                "content": self._format_problem_section(problem, framework),
            }
        )

        # Mental models section - Top 3 models
        for i, model in enumerate(models[:3]):
            sections.append(
                {"type": "model", "content": self._format_model_section(model, i + 1)}
            )

        # Cognitive insights and synergies
        if framework.get("synergies") or framework.get("key_insights"):
            sections.append(
                {
                    "type": "insights",
                    "content": self._format_insights_section(framework),
                }
            )

        # Implementation roadmap
        if implementation:
            sections.append(
                {
                    "type": "implementation",
                    "content": self._format_implementation_section(implementation),
                }
            )

        # Next steps and recommendations
        sections.append(
            {
                "type": "next_steps",
                "content": self._format_next_steps(implementation, analysis_id),
            }
        )

        return sections

    def _format_for_gamma(
        self, sections: List[Dict[str, Any]], template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format structured content for Gamma API"""

        # Join sections with card breaks
        content_blocks = []
        for section in sections:
            content_blocks.append(section["content"])

        gamma_input = "\n---\n".join(content_blocks)

        # Apply template settings
        gamma_config = {
            "input_text": gamma_input,
            "text_mode": template.get("text_mode", "generate"),
            "additional_instructions": template.get("instructions", ""),
            "theme_name": template.get("theme", "Oasis"),
            "num_cards": min(len(sections), template.get("num_cards", 10)),
            "text_options": template.get("text_options", {}),
            "image_options": template.get("image_options", {}),
        }

        return gamma_config

    def _extract_title_from_problem(self, problem: str) -> str:
        """Extract a concise title from the problem statement"""
        # Take the first sentence or first 60 characters
        sentences = problem.split(". ")
        if sentences and len(sentences[0]) < 80:
            return sentences[0].rstrip(".")

        # Fallback to truncated version
        title = problem[:60].strip()
        if len(problem) > 60:
            title += "..."

        return title or "Strategic Analysis"

    def _build_executive_summary(
        self, framework: Dict, models: List[Dict], implementation: Dict
    ) -> str:
        """Build executive summary from analysis components"""
        summary_parts = []

        # Add framework summary if available
        if framework.get("executive_summary"):
            summary_parts.append(framework["executive_summary"])

        # Add key insights
        if framework.get("key_insights"):
            insights = framework["key_insights"]
            if isinstance(insights, list):
                summary_parts.append("Key insights: " + "; ".join(insights[:3]))
            else:
                summary_parts.append(str(insights))

        # Add top mental model
        if models:
            top_model = models[0]
            summary_parts.append(
                f"Primary framework: {top_model.get('name', 'Mental Model')} - {top_model.get('description', '')[:100]}"
            )

        # Add implementation summary
        if implementation.get("immediate_actions"):
            actions = implementation["immediate_actions"]
            if isinstance(actions, list):
                summary_parts.append(f"Immediate actions: {', '.join(actions[:3])}")

        return "\n\n".join(summary_parts)

    def _format_problem_section(self, problem: str, framework: Dict) -> str:
        """Format problem analysis section"""
        content = f"# Problem Analysis\n\n## Challenge\n{problem}\n\n"

        # Add key considerations if available
        key_factors = framework.get("key_factors", [])
        if key_factors:
            content += f"## Key Considerations\n{self._bullet_list(key_factors)}\n\n"

        # Add complexity assessment
        complexity_info = []
        if framework.get("complexity"):
            complexity_info.append(f"**Level**: {framework['complexity']}")
        if framework.get("domain"):
            complexity_info.append(f"**Domain**: {framework['domain']}")
        if framework.get("impact"):
            complexity_info.append(f"**Impact**: {framework['impact']}")

        if complexity_info:
            content += "## Complexity Assessment\n" + "\n".join(complexity_info)

        return content

    def _format_model_section(self, model: Dict, number: int) -> str:
        """Format mental model section"""
        name = model.get("name", f"Mental Model {number}")
        description = model.get("description", "No description available")
        application = model.get("application", model.get("reasoning", ""))

        content = f"# {name}\n\n## Overview\n{description}\n\n"

        if application:
            content += f"## Application\n{application}\n\n"

        # Add benefits if available
        benefits = model.get("benefits", [])
        if benefits:
            content += f"## Key Benefits\n{self._bullet_list(benefits)}\n\n"

        # Add considerations
        considerations = model.get("considerations", model.get("limitations", []))
        if considerations:
            content += f"## Considerations\n{self._bullet_list(considerations)}"

        return content

    def _format_insights_section(self, framework: Dict) -> str:
        """Format cognitive insights section"""
        content = "# Cognitive Insights & Synergies\n\n"

        # Add synergies if available
        synergies = framework.get("synergies", [])
        if synergies:
            for i, synergy in enumerate(synergies[:3]):
                if isinstance(synergy, dict):
                    content += f"## {synergy.get('type', f'Insight {i+1}')}\n"
                    content += f"{synergy.get('description', '')}\n\n"
                    if synergy.get("impact"):
                        content += f"**Impact**: {synergy['impact']}\n\n"
                else:
                    content += f"## Insight {i+1}\n{str(synergy)}\n\n"

        # Add general insights
        insights = framework.get("key_insights", [])
        if insights and not synergies:
            content += "## Key Strategic Insights\n"
            content += self._bullet_list(
                insights if isinstance(insights, list) else [insights]
            )

        return content

    def _format_implementation_section(self, implementation: Dict) -> str:
        """Format implementation roadmap"""
        content = "# Implementation Roadmap\n\n"

        # Check for phase structure
        phases = ["phase1", "phase2", "phase3"]
        has_phases = any(implementation.get(phase) for phase in phases)

        if has_phases:
            for i, phase_key in enumerate(phases, 1):
                phase = implementation.get(phase_key, {})
                if phase:
                    content += f"## Phase {i}: {phase.get('focus', f'Phase {i}')}\n"
                    content += self._format_phase(phase) + "\n\n"
        else:
            # General implementation structure
            if implementation.get("timeline"):
                content += f"## Timeline\n{implementation['timeline']}\n\n"

            if implementation.get("key_activities"):
                content += f"## Key Activities\n{self._bullet_list(implementation['key_activities'])}\n\n"

        # Add success metrics
        metrics = implementation.get(
            "metrics", implementation.get("success_metrics", [])
        )
        if metrics:
            content += f"## Success Metrics\n{self._bullet_list(metrics)}"

        return content

    def _format_next_steps(self, implementation: Dict, analysis_id: str) -> str:
        """Format next steps section"""
        content = "# Next Steps\n\n"

        # Immediate actions
        immediate = implementation.get("immediate_actions", [])
        if immediate:
            content += f"## Immediate Actions\n{self._bullet_list(immediate)}\n\n"

        # Resources required
        resources = implementation.get(
            "resources", implementation.get("required_resources", [])
        )
        if resources:
            content += f"## Resources Required\n{self._bullet_list(resources)}\n\n"

        # Timeline information
        content += "## Timeline\n"
        quick_wins = implementation.get("quick_wins_timeline", "1-2 weeks")
        full_timeline = implementation.get("full_timeline", "2-3 months")
        content += f"- **Quick Wins**: {quick_wins}\n"
        content += f"- **Full Implementation**: {full_timeline}\n\n"

        # Footer
        content += "## About This Analysis\n"
        content += "*Generated by METIS Cognitive Platform*\n"
        content += f"*Analysis ID: {analysis_id}*\n"
        content += "*Powered by Advanced AI & Mental Models Framework*"

        return content

    def _format_phase(self, phase: Dict) -> str:
        """Format implementation phase"""
        content = f"**Duration**: {phase.get('duration', 'TBD')}\n"

        if phase.get("focus"):
            content += f"**Focus**: {phase['focus']}\n"

        activities = phase.get("activities", phase.get("key_activities", []))
        if activities:
            content += f"**Key Activities**:\n{self._bullet_list(activities)}"

        return content

    def _bullet_list(self, items: List) -> str:
        """Convert list to bullet points"""
        if not items:
            return "- To be determined"

        # Handle different item formats
        formatted_items = []
        for item in items:
            if isinstance(item, dict):
                # Extract meaningful content from dict
                if "name" in item and "description" in item:
                    formatted_items.append(f"{item['name']}: {item['description']}")
                else:
                    formatted_items.append(str(item))
            else:
                formatted_items.append(str(item))

        return "\n".join([f"- {item}" for item in formatted_items])

    def validate_analysis_result(self, analysis_result: Dict[str, Any]) -> bool:
        """Validate that analysis result contains minimum required data"""
        required_fields = ["problem_statement"]

        for field in required_fields:
            if field not in analysis_result:
                logger.warning(f"⚠️ Missing required field: {field}")
                return False

        return True
