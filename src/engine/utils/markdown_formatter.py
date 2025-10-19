"""
Markdown Formatter Module - "Intelligence as an Asset" Implementation

This module transforms METIS V5 analysis outputs into clean, human-readable, and portable
Markdown format, implementing the "Intelligence as an Asset" policy.

Key Features:
- Converts complete dossiers to comprehensive .md documents
- Formats context stream logs as readable narrative "Story Mode"
- Maintains professional formatting standards
- Preserves all analytical content and audit trails
- Creates portable, shareable intelligence assets

This ensures all METIS outputs are delivered in formats that maximize human readability
and knowledge retention.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass
import json


@dataclass
class MarkdownFormattingOptions:
    """
    Configuration options for markdown formatting
    """

    include_metadata: bool = True
    include_audit_trail: bool = True
    include_nway_clusters: bool = True
    include_processing_metrics: bool = True
    story_mode_context_stream: bool = True
    max_context_events: int = 100
    include_confidence_scores: bool = True
    include_timestamps: bool = True


class MarkdownFormatter:
    """
    Core utility for converting METIS outputs to professional Markdown format

    Implements "Intelligence as an Asset" by ensuring all cognitive intelligence
    is delivered in clean, portable, human-readable formats.
    """

    def __init__(self, options: Optional[MarkdownFormattingOptions] = None):
        self.options = options or MarkdownFormattingOptions()
        self.logger = logging.getLogger(__name__)

    def format_dossier_as_markdown(self, dossier_json: Dict[str, Any]) -> str:
        """
        Convert a complete METIS dossier JSON to comprehensive Markdown document

        Args:
            dossier_json: The complete final API payload from METIS analysis

        Returns:
            Professionally formatted Markdown document
        """
        try:
            timestamp = datetime.now(timezone.utc)

            # Build markdown document sections
            markdown_content = []

            # Header and metadata
            markdown_content.append(self._format_header(dossier_json, timestamp))

            # Executive summary
            if "executive_summary" in dossier_json:
                markdown_content.append(
                    self._format_executive_summary(dossier_json["executive_summary"])
                )

            # Query and context
            if "original_query" in dossier_json:
                markdown_content.append(self._format_query_section(dossier_json))

            # N-Way clusters (Level 3 Enhancement)
            if (
                self.options.include_nway_clusters
                and "selected_nway_clusters" in dossier_json
            ):
                markdown_content.append(
                    self._format_nway_clusters_section(dossier_json)
                )

            # Consultant analyses
            if "consultant_analyses" in dossier_json:
                markdown_content.append(
                    self._format_consultant_analyses(
                        dossier_json["consultant_analyses"]
                    )
                )

            # Devil's Advocate critiques
            if "devils_advocate_critiques" in dossier_json:
                markdown_content.append(
                    self._format_devils_advocate_section(
                        dossier_json["devils_advocate_critiques"]
                    )
                )

            # Senior Advisor meta-analysis
            if "senior_advisor_meta_analysis" in dossier_json:
                markdown_content.append(
                    self._format_senior_advisor_section(
                        dossier_json["senior_advisor_meta_analysis"]
                    )
                )

            # Processing metrics
            if (
                self.options.include_processing_metrics
                and "processing_metadata" in dossier_json
            ):
                markdown_content.append(
                    self._format_processing_metrics(dossier_json["processing_metadata"])
                )

            # Audit trail
            if self.options.include_audit_trail and "audit_trail" in dossier_json:
                markdown_content.append(
                    self._format_audit_trail(dossier_json["audit_trail"])
                )

            # Footer
            markdown_content.append(self._format_footer(timestamp))

            # Join all sections
            full_document = "\n\n".join(filter(None, markdown_content))

            self.logger.info(
                f"✅ Generated Markdown document ({len(full_document)} characters)"
            )
            return full_document

        except Exception as e:
            self.logger.error(f"❌ Error formatting dossier as Markdown: {e}")
            return f"# METIS Analysis Report\n\n**Error:** Could not format dossier - {e}\n\n{json.dumps(dossier_json, indent=2)}"

    def format_context_stream_as_markdown(
        self, context_stream_log: List[Dict[str, Any]]
    ) -> str:
        """
        Convert context stream log to readable narrative "Story Mode"

        Args:
            context_stream_log: List of context stream events

        Returns:
            Narrative Markdown document telling the "story" of the analysis
        """
        try:
            if not context_stream_log:
                return "# METIS Analysis Story\n\n**No context events available**"

            # Limit events if specified
            events_to_process = (
                context_stream_log[-self.options.max_context_events :]
                if self.options.max_context_events
                else context_stream_log
            )

            markdown_content = []

            # Story header
            markdown_content.append("# 📖 METIS Analysis Story")
            markdown_content.append(
                "*A narrative view of the cognitive intelligence process*"
            )
            markdown_content.append("---")

            # Group events by phase
            phase_events = self._group_events_by_phase(events_to_process)

            for phase_name, phase_events_list in phase_events.items():
                if phase_events_list:
                    phase_section = self._format_phase_narrative(
                        phase_name, phase_events_list
                    )
                    markdown_content.append(phase_section)

            # Story footer
            total_events = len(context_stream_log)
            events_shown = len(events_to_process)

            footer = f"""
---

## 📊 Story Metrics

- **Total Events Captured:** {total_events}
- **Events in Story:** {events_shown}
- **Story Generation:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Powered by:** METIS V5 Cognitive Intelligence Platform
"""
            markdown_content.append(footer)

            full_story = "\n\n".join(filter(None, markdown_content))

            self.logger.info(
                f"✅ Generated Context Stream story ({len(full_story)} characters, {events_shown} events)"
            )
            return full_story

        except Exception as e:
            self.logger.error(f"❌ Error formatting context stream as Markdown: {e}")
            return f"# METIS Analysis Story\n\n**Error:** Could not format story - {e}"

    def _format_header(self, dossier_json: Dict[str, Any], timestamp: datetime) -> str:
        """Format the document header"""
        engagement_id = dossier_json.get("engagement_id", "Unknown")

        header = f"""# 🧠 METIS Cognitive Intelligence Report

**Engagement ID:** `{engagement_id}`  
**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Platform:** METIS V5 - Level 3 Intelligence Enhancement  
**Architecture:** Multi-Single-Agent Paradigm with N-Way Cognitive Infusion"""

        if self.options.include_metadata:
            metadata = dossier_json.get("metadata", {})
            if metadata:
                header += f"\n**System Version:** {metadata.get('version', 'Unknown')}"
                header += f"  \n**Analysis Type:** {metadata.get('analysis_type', 'Comprehensive Cognitive Analysis')}"

        header += "\n\n---"
        return header

    def _format_executive_summary(self, executive_summary: str) -> str:
        """Format executive summary section"""
        return f"""## 📋 Executive Summary

{executive_summary}"""

    def _format_query_section(self, dossier_json: Dict[str, Any]) -> str:
        """Format the original query and context section"""
        query = dossier_json.get("original_query", "Query not available")
        enhanced_query = dossier_json.get("enhanced_query")

        section = f"""## 🎯 Original Query

```
{query}
```"""

        if enhanced_query and enhanced_query != query:
            section += f"""

### Enhanced Query (Post-Socratic)
```
{enhanced_query}
```"""

        # Add context if available
        context = dossier_json.get("analysis_context", {})
        if context:
            section += "\n\n### Context Parameters\n"
            for key, value in context.items():
                section += f"- **{key.replace('_', ' ').title()}:** {value}\n"

        return section

    def _format_nway_clusters_section(self, dossier_json: Dict[str, Any]) -> str:
        """Format N-Way cognitive clusters section (Level 3 Enhancement)"""
        selected_clusters = dossier_json.get("selected_nway_clusters", [])
        nway_metadata = dossier_json.get("nway_cluster_metadata", {})

        if not selected_clusters:
            return "## 🧩 N-Way Cognitive Directives\n\n*No N-Way clusters were applied to this analysis.*"

        section = f"""## 🧩 N-Way Cognitive Directives

**Applied Clusters:** {len(selected_clusters)}

This analysis leveraged proprietary N-Way interaction patterns to enhance cognitive reasoning:"""

        for i, cluster_id in enumerate(selected_clusters, 1):
            cluster_data = nway_metadata.get(cluster_id, {})

            section += f"""

### {i}. {cluster_id}

- **Type:** {cluster_data.get('type', 'Unknown')}
- **Models Involved:** {', '.join(cluster_data.get('models_involved', []))}
- **Emergent Effect:** {cluster_data.get('emergent_effect_summary', 'Not available')}
- **Application:** {cluster_data.get('instructional_cue_apce', 'Not available')}"""

        return section

    def _format_consultant_analyses(self, consultant_analyses: Dict[str, Any]) -> str:
        """Format individual consultant analyses"""
        section = "## 👥 Independent Consultant Analyses\n\n*Each consultant analyzed the problem independently using their specialized expertise and the mandated N-Way cognitive directives.*"

        for consultant_id, analysis in consultant_analyses.items():
            consultant_name = analysis.get(
                "consultant_name", consultant_id.replace("_", " ").title()
            )

            section += f"""

### 🎭 {consultant_name}

**Role:** {analysis.get('specialization', 'Strategic Consultant')}"""

            if self.options.include_confidence_scores:
                section += f"  \n**Confidence Level:** {analysis.get('confidence_level', 'Not specified')}"

            if analysis.get("executive_summary"):
                section += f"""

#### Executive Summary
{analysis['executive_summary']}"""

            if analysis.get("key_insights"):
                section += "\n\n#### Key Insights"
                for i, insight in enumerate(analysis["key_insights"], 1):
                    section += f"\n{i}. {insight}"

            if analysis.get("recommendations"):
                section += "\n\n#### Recommendations"
                for i, rec in enumerate(analysis["recommendations"], 1):
                    section += f"\n{i}. {rec}"

            if analysis.get("frameworks_used"):
                section += f"\n\n**Frameworks Applied:** {', '.join(analysis['frameworks_used'])}"

        return section

    def _format_devils_advocate_section(
        self, devils_advocate_critiques: Dict[str, Any]
    ) -> str:
        """Format Devil's Advocate critiques"""
        section = "## 🛡️ Devil's Advocate Quality Control\n\n*Independent critique of each consultant's analysis to identify potential blind spots, biases, and weaknesses.*"

        for consultant_id, critique in devils_advocate_critiques.items():
            consultant_name = consultant_id.replace("_", " ").title()

            section += f"""

### Critique of {consultant_name}

#### Strengths Identified
{critique.get('strengths', 'Not specified')}

#### Weaknesses & Blind Spots
{critique.get('weaknesses', 'Not specified')}

#### Bias Assessment
{critique.get('bias_analysis', 'Not specified')}"""

            if critique.get("nway_application_audit"):
                section += f"""

#### N-Way Application Audit
**Score:** {critique['nway_application_audit'].get('application_score', 'Not scored')}  
**Assessment:** {critique['nway_application_audit'].get('assessment', 'Not available')}"""

        return section

    def _format_senior_advisor_section(
        self, senior_advisor_analysis: Dict[str, Any]
    ) -> str:
        """Format Senior Advisor meta-analysis"""
        section = "## 🎩 Senior Advisor Meta-Analysis\n\n*Comparative analysis of consultant perspectives without synthesis, preserving all viewpoints for human decision-making.*"

        if senior_advisor_analysis.get("consultant_theses"):
            section += "\n\n### Core Theses by Consultant"
            for consultant, thesis in senior_advisor_analysis[
                "consultant_theses"
            ].items():
                section += f"\n- **{consultant.title()}:** {thesis}"

        if senior_advisor_analysis.get("key_tensions"):
            section += "\n\n### Key Decision Tensions"
            for i, tension in enumerate(senior_advisor_analysis["key_tensions"], 1):
                section += f"""

#### {i}. {tension.get('tension_topic', 'Unknown Tension')}

**Consultant Positions:**"""
                for consultant, position in tension.get(
                    "consultant_positions", {}
                ).items():
                    section += f"\n- **{consultant.title()}:** {position}"

                section += f"\n\n**Decision Implication:** {tension.get('decision_implication', 'Not specified')}"

        if senior_advisor_analysis.get("unexpected_synergies"):
            section += "\n\n### Unexpected Synergies"
            for i, synergy in enumerate(
                senior_advisor_analysis["unexpected_synergies"], 1
            ):
                section += f"""

#### {i}. {synergy.get('synergy_topic', 'Unknown Synergy')}

**Supporting Consultants:** {', '.join(synergy.get('supporting_consultants', []))}  
**Strategic Significance:** {synergy.get('strategic_significance', 'Not specified')}"""

        # Level 3 Enhancement: N-Way Application Audit
        if senior_advisor_analysis.get("nway_application_audit"):
            nway_audit = senior_advisor_analysis["nway_application_audit"]
            section += f"""

### N-Way Cognitive Model Application Audit

**Overall Compliance Score:** {nway_audit.get('overall_nway_compliance', 'Not scored')}

**Individual Consultant Scores:**"""

            for consultant, audit_data in nway_audit.items():
                if (
                    isinstance(audit_data, dict)
                    and "nway_application_score" in audit_data
                ):
                    score = audit_data.get("nway_application_score", "Not scored")
                    assessment = audit_data.get(
                        "application_quality_assessment", "Not available"
                    )
                    section += f"\n- **{consultant.title()}:** {score} - {assessment}"

            if nway_audit.get("proprietary_ip_utilization"):
                section += f"\n\n**Proprietary IP Utilization:** {nway_audit['proprietary_ip_utilization']}"

        return section

    def _format_processing_metrics(self, processing_metadata: Dict[str, Any]) -> str:
        """Format processing metrics and performance data"""
        section = "## ⚙️ Processing Metrics"

        total_time = processing_metadata.get("total_processing_time_seconds", 0)
        section += f"\n\n**Total Processing Time:** {total_time:.2f} seconds"

        if processing_metadata.get("consultant_count"):
            section += f"  \n**Consultants Engaged:** {processing_metadata['consultant_count']}"

        if processing_metadata.get("nway_cluster_count"):
            section += f"  \n**N-Way Clusters Applied:** {processing_metadata['nway_cluster_count']}"

        if processing_metadata.get("llm_calls_made"):
            section += (
                f"  \n**LLM Calls Made:** {processing_metadata['llm_calls_made']}"
            )

        if processing_metadata.get("total_tokens_used"):
            section += f"  \n**Total Tokens Used:** {processing_metadata['total_tokens_used']:,}"

        if processing_metadata.get("estimated_cost_usd"):
            section += f"  \n**Estimated Cost:** ${processing_metadata['estimated_cost_usd']:.4f}"

        # Financial Transparency Enhancement - Additional metrics
        if processing_metadata.get("perplexity_calls_made"):
            section += f"  \n**Perplexity Calls Made:** {processing_metadata['perplexity_calls_made']}"

        if processing_metadata.get("characters_generated"):
            section += f"  \n**Characters Generated:** {processing_metadata['characters_generated']:,}"

        if processing_metadata.get("reasoning_steps_count"):
            section += f"  \n**Reasoning Steps:** {processing_metadata['reasoning_steps_count']}"

        if processing_metadata.get("unique_providers_used"):
            section += f"  \n**Unique Providers Used:** {processing_metadata['unique_providers_used']}"

        return section

    def _format_audit_trail(self, audit_trail: List[Dict[str, Any]]) -> str:
        """Format complete audit trail"""
        section = f"## 🔍 Audit Trail\n\n**Total Events:** {len(audit_trail)}"

        # Show key milestones
        key_events = [
            event
            for event in audit_trail
            if event.get("event_type")
            in [
                "ENGAGEMENT_STARTED",
                "CONSULTANT_SELECTION",
                "ANALYSIS_COMPLETE",
                "SENIOR_ADVISOR_ARBITRATION_START",
                "ENGAGEMENT_COMPLETED",
            ]
        ]

        if key_events:
            section += "\n\n### Key Milestones"
            for event in key_events:
                timestamp = event.get("timestamp", "Unknown time")
                event_type = event.get("event_type", "Unknown event")
                description = event.get(
                    "description",
                    event.get("data", {}).get("description", "No description"),
                )

                section += f"\n- **{timestamp}** - {event_type}: {description}"

        return section

    def _format_footer(self, timestamp: datetime) -> str:
        """Format document footer"""
        return f"""---

## 📄 Document Information

**Document Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Format:** Markdown (GitHub Flavored)  
**Generator:** METIS V5 Markdown Formatter  
**Architecture:** Multi-Single-Agent Paradigm  
**Quality Standard:** "Intelligence as an Asset"

*This document represents a complete cognitive intelligence analysis maintaining full transparency and auditability. All consultant perspectives are preserved independently to support human decision-making.*

**Powered by METIS V5 Cognitive Intelligence Platform**"""

    def _group_events_by_phase(
        self, events: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group context events by analysis phase for story narrative"""
        phases = {
            "🚀 Initialization": [],
            "🎯 Query Enhancement": [],
            "👥 Consultant Selection": [],
            "🧠 Analysis Execution": [],
            "🛡️ Quality Control": [],
            "🎩 Meta-Analysis": [],
            "✅ Completion": [],
        }

        phase_mapping = {
            "ENGAGEMENT_STARTED": "🚀 Initialization",
            "SOCRATIC_QUESTIONS": "🎯 Query Enhancement",
            "QUERY_ENHANCEMENT": "🎯 Query Enhancement",
            "CONSULTANT_SELECTION": "👥 Consultant Selection",
            "MODEL_APPLIED": "👥 Consultant Selection",
            "ANALYSIS_EXECUTION": "🧠 Analysis Execution",
            "LLM_CALL": "🧠 Analysis Execution",
            "DEVILS_ADVOCATE": "🛡️ Quality Control",
            "SENIOR_ADVISOR": "🎩 Meta-Analysis",
            "ENGAGEMENT_COMPLETED": "✅ Completion",
        }

        for event in events:
            event_type = event.get("event_type", "")

            # Map event to phase
            phase = None
            for key, phase_name in phase_mapping.items():
                if key in event_type:
                    phase = phase_name
                    break

            if phase:
                phases[phase].append(event)
            else:
                phases["🧠 Analysis Execution"].append(event)  # Default phase

        return phases

    def _format_phase_narrative(
        self, phase_name: str, phase_events: List[Dict[str, Any]]
    ) -> str:
        """Format a single phase as narrative"""
        if not phase_events:
            return ""

        section = f"## {phase_name}\n"

        for event in phase_events:
            timestamp = event.get("timestamp", "Unknown time")
            if self.options.include_timestamps:
                section += f"\n**{timestamp}**  "

            # Create narrative description
            narrative = self._create_event_narrative(event)
            section += f"{narrative}\n"

        return section

    def _create_event_narrative(self, event: Dict[str, Any]) -> str:
        """Create human-readable narrative for a context event"""
        event_type = event.get("event_type", "Unknown Event")
        data = event.get("data", {})

        # Map event types to narratives
        narrative_templates = {
            "ENGAGEMENT_STARTED": "🎬 Started new cognitive intelligence engagement",
            "CONSULTANT_SELECTION": "🎯 Selected optimal consultant team",
            "MODEL_APPLIED": "🧩 Applied cognitive model or framework",
            "LLM_CALL_START": "💭 Initiated AI reasoning process",
            "LLM_CALL_COMPLETE": "✅ Completed AI reasoning process",
            "DEVILS_ADVOCATE": "🛡️ Conducted quality control critique",
            "SENIOR_ADVISOR_ARBITRATION_START": "🎩 Began meta-analysis of consultant perspectives",
            "ENGAGEMENT_COMPLETED": "🎉 Completed cognitive intelligence analysis",
        }

        base_narrative = narrative_templates.get(event_type, f"📝 {event_type}")

        # Add specific details where available
        if data.get("consultant_id"):
            base_narrative += f" for {data['consultant_id']}"

        if data.get("model_name"):
            base_narrative += f" using {data['model_name']}"

        if data.get("processing_time"):
            base_narrative += f" (took {data['processing_time']:.2f}s)"

        return base_narrative


# Global instance management
_markdown_formatter = None


def get_markdown_formatter(
    options: Optional[MarkdownFormattingOptions] = None,
) -> MarkdownFormatter:
    """
    Get the global MarkdownFormatter instance

    Args:
        options: Formatting options (optional)

    Returns:
        Global MarkdownFormatter instance
    """
    global _markdown_formatter

    if _markdown_formatter is None or options:
        _markdown_formatter = MarkdownFormatter(options)

    return _markdown_formatter


def reset_markdown_formatter():
    """Reset the global instance (useful for testing)"""
    global _markdown_formatter
    _markdown_formatter = None
