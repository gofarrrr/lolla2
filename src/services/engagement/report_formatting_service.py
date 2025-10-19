"""
Report Formatting Service - Business Logic Layer
=================================================

Extracts report formatting logic from engagements.py route handlers.

This service handles:
- Flattening nested report structures for frontend compatibility
- Generating markdown reports
- Extracting and enriching report sections

Target Complexity: CC ≤ 10 per method
Reduces CC=51 hotspot to manageable methods
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportFormattingService:
    """
    Service for formatting and transforming report data.

    Responsibilities:
    - Flatten nested report structures for frontend compatibility
    - Extract and enrich report sections (consultants, recommendations, etc.)
    - Generate markdown reports from structured data
    """

    def flatten_report(self, report_data: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """
        Flatten nested report structure for frontend compatibility.

        Complexity: CC ≤ 4 (delegates to helper methods)

        Args:
            report_data: The nested report data structure
            trace_id: The engagement trace ID for logging

        Returns:
            Flattened report dict with top-level fields
        """
        if not report_data:
            return {}

        flattened = dict(report_data)

        # Extract each section using dedicated helper methods
        flattened = self._extract_consultant_analyses(flattened, report_data, trace_id)
        flattened = self._extract_strategic_recommendations(flattened, report_data)
        flattened = self._promote_da_transcript(flattened, report_data)
        flattened = self._extract_executive_summary(flattened, report_data)
        flattened = self._extract_quality_metrics(flattened, report_data)
        flattened = self._extract_evidence(flattened, report_data)
        flattened = self._extract_key_decisions(flattened, report_data)
        flattened = self._extract_consultant_methodology(flattened, report_data)

        self._log_flattening_summary(flattened, report_data, trace_id)

        return flattened

    def _extract_consultant_analyses(
        self,
        flattened: Dict[str, Any],
        report_data: Dict[str, Any],
        trace_id: str
    ) -> Dict[str, Any]:
        """
        Extract and enrich consultant analyses from parallel_analysis.

        Complexity: CC ≤ 6

        Args:
            flattened: The flattened dict being built
            report_data: The source report data
            trace_id: The engagement trace ID for logging

        Returns:
            Updated flattened dict
        """
        if 'parallel_analysis' not in report_data:
            return flattened

        parallel_analysis = report_data['parallel_analysis']
        if not isinstance(parallel_analysis, dict):
            return flattened

        consultant_analyses = parallel_analysis.get('consultant_analyses', [])

        if not consultant_analyses or not isinstance(consultant_analyses, list):
            return flattened

        # Enrich each consultant with grounding metadata
        for consultant in consultant_analyses:
            if isinstance(consultant, dict):
                consultant = self._add_grounding_metadata(consultant)

        flattened['consultant_analyses'] = consultant_analyses

        fully_grounded = sum(
            1 for c in consultant_analyses
            if isinstance(c, dict) and c.get('is_fully_grounded', False)
        )
        logger.info(
            f"🔧 FLATTEN: Extracted {len(consultant_analyses)} consultant_analyses "
            f"from parallel_analysis ({fully_grounded} fully grounded, "
            f"{len(consultant_analyses)-fully_grounded} partial/not grounded)"
        )

        return flattened

    def _add_grounding_metadata(self, consultant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add grounding metadata to a consultant analysis.

        Complexity: CC = 4

        Args:
            consultant: The consultant dict to enrich

        Returns:
            Enriched consultant dict
        """
        has_evidence = bool(consultant.get('evidence') or consultant.get('citations'))
        has_analysis = bool(consultant.get('analysis') or consultant.get('key_insights'))
        grounding_score = consultant.get('grounding_score', 0.0 if not has_evidence else 0.8)

        consultant['is_fully_grounded'] = has_evidence and has_analysis
        consultant['grounding_score'] = grounding_score

        if has_evidence and has_analysis:
            consultant['grounding_status'] = 'fully_grounded'
        elif has_evidence or has_analysis:
            consultant['grounding_status'] = 'partially_grounded'
        else:
            consultant['grounding_status'] = 'not_grounded'

        if not consultant['is_fully_grounded']:
            consultant['grounding_warning'] = (
                "This analysis may not be fully grounded in research data. "
                "Treat insights as directional rather than definitive."
            )

        return consultant

    def _extract_strategic_recommendations(
        self,
        flattened: Dict[str, Any],
        report_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract strategic recommendations from senior_advisor or recommendations field.

        Complexity: CC = 4

        Args:
            flattened: The flattened dict being built
            report_data: The source report data

        Returns:
            Updated flattened dict
        """
        if 'senior_advisor' in report_data and isinstance(report_data['senior_advisor'], dict):
            recommendations = report_data['senior_advisor'].get('recommendations', [])
            if recommendations:
                flattened['strategic_recommendations'] = recommendations
                logger.info(f"🔧 FLATTEN: Extracted {len(recommendations)} strategic_recommendations from senior_advisor")
        elif 'recommendations' in report_data and isinstance(report_data['recommendations'], list):
            flattened['strategic_recommendations'] = report_data['recommendations']
            logger.info(f"🔧 FLATTEN: Using existing recommendations as strategic_recommendations")

        return flattened

    def _promote_da_transcript(
        self,
        flattened: Dict[str, Any],
        report_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Promote da_transcript to devils_advocate_transcript at top level.

        Complexity: CC = 2

        Args:
            flattened: The flattened dict being built
            report_data: The source report data

        Returns:
            Updated flattened dict
        """
        if 'da_transcript' in report_data and 'devils_advocate_transcript' not in flattened:
            flattened['devils_advocate_transcript'] = report_data['da_transcript']
            logger.info(f"🔧 FLATTEN: Promoted da_transcript to devils_advocate_transcript")

        return flattened

    def _extract_executive_summary(
        self,
        flattened: Dict[str, Any],
        report_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract executive summary from nested structures.

        Complexity: CC = 4

        Args:
            flattened: The flattened dict being built
            report_data: The source report data

        Returns:
            Updated flattened dict
        """
        if 'executive_summary' in flattened and flattened['executive_summary']:
            return flattened

        if 'senior_advisor' in report_data and isinstance(report_data['senior_advisor'], dict):
            exec_summary = report_data['senior_advisor'].get('executive_summary')
            if exec_summary:
                flattened['executive_summary'] = exec_summary
                logger.info(f"🔧 FLATTEN: Extracted executive_summary from senior_advisor")

        return flattened

    def _extract_quality_metrics(
        self,
        flattened: Dict[str, Any],
        report_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract quality metrics from analysis_confidence.

        Complexity: CC = 2

        Args:
            flattened: The flattened dict being built
            report_data: The source report data

        Returns:
            Updated flattened dict
        """
        if 'quality_metrics' not in flattened and 'analysis_confidence' in report_data:
            flattened['quality_metrics'] = report_data['analysis_confidence']
            logger.info(f"🔧 FLATTEN: Using analysis_confidence as quality_metrics")

        return flattened

    def _extract_evidence(
        self,
        flattened: Dict[str, Any],
        report_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract evidence from research_grounding.

        Complexity: CC = 4

        Args:
            flattened: The flattened dict being built
            report_data: The source report data

        Returns:
            Updated flattened dict
        """
        if 'evidence' not in flattened and 'research_grounding' in report_data:
            if isinstance(report_data['research_grounding'], dict):
                evidence = report_data['research_grounding'].get('evidence', [])
                if evidence:
                    flattened['evidence'] = evidence
                    logger.info(f"🔧 FLATTEN: Extracted evidence from research_grounding")

        return flattened

    def _extract_key_decisions(
        self,
        flattened: Dict[str, Any],
        report_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract key decisions from senior_advisor.

        Complexity: CC = 4

        Args:
            flattened: The flattened dict being built
            report_data: The source report data

        Returns:
            Updated flattened dict
        """
        if 'key_decisions' not in flattened:
            if 'senior_advisor' in report_data and isinstance(report_data['senior_advisor'], dict):
                decisions = report_data['senior_advisor'].get('key_decisions', [])
                if decisions:
                    flattened['key_decisions'] = decisions
                    logger.info(f"🔧 FLATTEN: Extracted {len(decisions)} key_decisions from senior_advisor")

        return flattened

    def _extract_consultant_methodology(
        self,
        flattened: Dict[str, Any],
        report_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract consultant selection methodology.

        Complexity: CC = 4

        Args:
            flattened: The flattened dict being built
            report_data: The source report data

        Returns:
            Updated flattened dict
        """
        if 'consultant_selection_methodology' in flattened and flattened['consultant_selection_methodology']:
            return flattened

        if 'consultant_selection' in report_data and isinstance(report_data['consultant_selection'], dict):
            methodology = report_data['consultant_selection'].get('methodology')
            if methodology:
                flattened['consultant_selection_methodology'] = methodology
                logger.info(f"🔧 FLATTEN: Extracted consultant_selection_methodology")

        return flattened

    def _log_flattening_summary(
        self,
        flattened: Dict[str, Any],
        report_data: Dict[str, Any],
        trace_id: str
    ) -> None:
        """
        Log summary of flattening operation.

        Complexity: CC = 2

        Args:
            flattened: The flattened dict
            report_data: The original report data
            trace_id: The engagement trace ID
        """
        added_fields = [
            k for k in ['consultant_analyses', 'strategic_recommendations',
                       'devils_advocate_transcript', 'executive_summary',
                       'quality_metrics', 'key_decisions']
            if k in flattened and k not in report_data
        ]

        if added_fields:
            logger.info(
                f"🔧 FLATTEN: Successfully flattened {len(added_fields)} fields "
                f"for trace {trace_id}: {added_fields}"
            )
        else:
            logger.info(f"🔧 FLATTEN: No additional flattening needed for trace {trace_id}")

    def generate_markdown(self, final_output: Dict[str, Any]) -> str:
        """
        Generate markdown content from final output.

        Complexity: CC ≤ 3 (delegates to helper methods)

        Args:
            final_output: The final report data

        Returns:
            Markdown string
        """
        markdown = ["# Strategic Analysis Report\n"]

        markdown.extend(self._generate_executive_summary_section(final_output))
        markdown.extend(self._generate_recommendations_section(final_output))
        markdown.extend(self._generate_insights_section(final_output))
        markdown.extend(self._generate_decisions_section(final_output))
        markdown.extend(self._generate_research_section(final_output))
        markdown.extend(self._generate_metadata_section(final_output))

        markdown.append("---\n*Generated by Lolla Strategic Intelligence Platform*\n")

        return "".join(markdown)

    def _generate_executive_summary_section(self, final_output: Dict[str, Any]) -> List[str]:
        """Generate executive summary section. CC = 1"""
        return [
            "## Executive Summary\n",
            f"{final_output.get('executive_summary', 'Analysis completed.')}\n\n"
        ]

    def _generate_recommendations_section(self, final_output: Dict[str, Any]) -> List[str]:
        """Generate recommendations section. CC = 6"""
        lines = ["## Key Recommendations\n"]

        recommendations = final_output.get("strategic_recommendations")
        if isinstance(recommendations, list) and recommendations:
            for idx, rec in enumerate(recommendations, 1):
                if isinstance(rec, dict):
                    text = rec.get("recommendation") or "Recommendation"
                    priority = rec.get("priority")
                    lines.append(f"{idx}. {text}")
                    if priority:
                        lines.append(f" _(Priority: {priority})_")
                    lines.append("\n")
                    rationale = rec.get("rationale")
                    if rationale:
                        lines.append(f"   - Rationale: {rationale}\n")
                else:
                    lines.append(f"{idx}. {rec}\n")
            lines.append("\n")
        else:
            for i, rec in enumerate(final_output.get('key_recommendations', []), 1):
                lines.append(f"{i}. {rec}\n")
            lines.append("\n")

        return lines

    def _generate_insights_section(self, final_output: Dict[str, Any]) -> List[str]:
        """Generate critical insights section. CC = 2"""
        lines = []
        critical_insights = final_output.get("critical_insights") or []
        if critical_insights:
            lines.append("## Critical Insights\n")
            for insight in critical_insights:
                lines.append(f"- {insight}\n")
            lines.append("\n")
        return lines

    def _generate_decisions_section(self, final_output: Dict[str, Any]) -> List[str]:
        """Generate decisions required section. CC = 2"""
        lines = []
        decisions_required = final_output.get("key_decisions_required") or []
        if decisions_required:
            lines.append("## Decisions Required\n")
            for decision in decisions_required:
                lines.append(f"- {decision}\n")
            lines.append("\n")
        return lines

    def _generate_research_section(self, final_output: Dict[str, Any]) -> List[str]:
        """Generate enhancement research section. CC = 5"""
        lines = []
        enhancement_research = final_output.get("enhancement_research_answers") or []
        if not enhancement_research:
            return lines

        lines.append("## Enhancement Research\n")
        lines.append("*Questions you flagged for research during enhancement*\n\n")

        for idx, research in enumerate(enhancement_research, 1):
            question = research.get("question_text", "Unknown question")
            answer = research.get("answer", "No answer available")
            citations = research.get("citations", [])

            lines.append(f"### {idx}. {question}\n\n")
            lines.append(f"{answer}\n\n")

            if citations:
                lines.append("**Sources:**\n")
                for citation in citations[:3]:
                    title = citation.get("title") or "Untitled source"
                    url = citation.get("url", "")
                    lines.append(f"- [{title}]({url})\n")
                lines.append("\n")

        return lines

    def _generate_metadata_section(self, final_output: Dict[str, Any]) -> List[str]:
        """Generate analysis metadata section. CC = 4"""
        lines = ["## Analysis Metadata\n"]

        lines.extend(self._add_confidence_metrics(final_output))
        lines.extend(self._add_confidence_assessment(final_output))

        lines.append(
            f"- **Pipeline Efficiency**: {final_output.get('pipeline_efficiency_score', 0.0):.2f}\n"
        )
        lines.append(
            f"- **Total Processing Time**: {final_output.get('total_processing_time_ms', 0)} ms\n"
        )
        lines.append(
            f"- **Generated**: {final_output.get('generated_at', datetime.now().isoformat())}\n\n"
        )

        return lines

    def _add_confidence_metrics(self, final_output: Dict[str, Any]) -> List[str]:
        """Add confidence score and level to metadata. CC = 4"""
        lines = []

        confidence = final_output.get("analysis_confidence") or {}
        if not isinstance(confidence, dict):
            return lines

        confidence_score = confidence.get("confidence_score") or confidence.get("score")
        confidence_label = confidence.get("confidence_level") or confidence.get("label")

        if confidence_score is not None:
            lines.append(f"- **Confidence Score**: {float(confidence_score):.2f}\n")
        if confidence_label:
            lines.append(f"- **Confidence Level**: {confidence_label}\n")

        return lines

    def _add_confidence_assessment(self, final_output: Dict[str, Any]) -> List[str]:
        """Add confidence assessment details to metadata. CC = 6"""
        lines = []

        conf_assess = final_output.get("confidence_assessment")
        if not isinstance(conf_assess, dict):
            return lines

        # Probability of success
        try:
            p = float(conf_assess.get("probability_of_success", 0.0))
            band = float(conf_assess.get("confidence_band", 0.0))
            lines.append(f"- **Probability of Success**: {p:.2f} ± {band:.2f}\n")
        except Exception:
            pass

        # Key uncertainties
        ku = conf_assess.get("key_uncertainties") or []
        if ku:
            lines.append("- **Key Uncertainties:**\n")
            for u in ku[:6]:
                lines.append(f"  - {u}\n")

        # Critical triggers
        ct = conf_assess.get("critical_triggers") or []
        if ct:
            lines.append("- **Critical Triggers:**\n")
            for t in ct[:6]:
                lines.append(f"  - {t}\n")

        return lines
