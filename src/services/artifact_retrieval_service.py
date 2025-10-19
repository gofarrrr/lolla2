"""
ArtifactRetrievalService - Operation "Evidence Locker" Backend
============================================================

The authoritative "Librarian" service for extracting analysis artifacts from context streams.
This service connects to the context_streams database and intelligently parses complex JSON logs
to find and extract specific artifacts (final reports, intermediate analyses, etc.).

Author: ARC Chief System Architect
Status: CRITICAL INFRASTRUCTURE
"""

import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ArtifactRetrievalService:
    """
    🔐 The Evidence Locker Backend

    Responsible for:
    1. Connecting to context_streams database
    2. Parsing complex JSONB logs to find specific events
    3. Extracting raw artifact content from event payloads
    4. Returning authenticated, unabridged content
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the Evidence Locker service."""
        self.db_path = db_path or "context_streams.db"
        self.artifact_extractors = {
            "final_report": self._extract_final_report,
            "analysis_output": self._extract_analysis_output,
            "intermediate_analysis": self._extract_intermediate_analysis,
            "consultant_recommendation": self._extract_consultant_recommendation,
            "devils_advocate_challenges": self._extract_devils_advocate_challenges,
            "synthesis_result": self._extract_synthesis_result,
            "raw_llm_response": self._extract_raw_llm_response,
            "executive_summary": self._extract_executive_summary,
        }

    async def get_artifact(
        self, trace_id: str, artifact_type: str, event_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        🔍 Primary Evidence Locker Method

        Retrieves a specific artifact from a completed analysis run.

        Args:
            trace_id: Unique identifier for the analysis run
            artifact_type: Type of artifact to retrieve
            event_id: Optional specific event ID for precise targeting

        Returns:
            Dict containing artifact content and metadata, or None if not found
        """

        logger.info(
            f"🔐 Evidence Locker: Retrieving {artifact_type} for trace {trace_id}"
        )

        try:
            # Step 1: Fetch the raw context stream from database
            context_stream_data = await self._fetch_context_stream_from_db(trace_id)

            if not context_stream_data:
                logger.warning(f"⚠️ Trace {trace_id} not found in database")
                return None

            # Step 2: Parse the JSON log structure
            context_stream = json.loads(context_stream_data["context_stream"])

            # Step 3: Use specialized extractor to find the artifact
            if artifact_type not in self.artifact_extractors:
                logger.error(f"❌ Unknown artifact type: {artifact_type}")
                return None

            extractor = self.artifact_extractors[artifact_type]
            artifact_content = await extractor(context_stream, event_id)

            if not artifact_content:
                logger.warning(
                    f"⚠️ Artifact {artifact_type} not found in trace {trace_id}"
                )
                return None

            # Step 4: Package the result with metadata
            result = {
                "artifact_content": artifact_content,
                "metadata": {
                    "artifact_type": artifact_type,
                    "content_length": len(str(artifact_content)),
                    "extraction_method": extractor.__name__,
                    "event_id": event_id,
                },
                "trace_context": {
                    "trace_id": trace_id,
                    "total_events": context_stream_data.get("total_events", 0),
                    "final_status": context_stream_data.get("final_status", "unknown"),
                    "completed_at": context_stream_data.get("completed_at"),
                },
                "retrieved_at": datetime.now().isoformat(),
            }

            logger.info(
                f"✅ Evidence Locker success: {len(str(artifact_content))} chars retrieved"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed for trace {trace_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Evidence Locker retrieval failed: {e}")
            return None

    async def list_artifacts(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        📋 Discover Available Artifacts

        Scans a trace's context stream to identify all available artifacts.
        """

        try:
            context_stream_data = await self._fetch_context_stream_from_db(trace_id)
            if not context_stream_data:
                return []

            context_stream = json.loads(context_stream_data["context_stream"])
            available_artifacts = []

            # Test each extractor to see what's available
            for artifact_type, extractor in self.artifact_extractors.items():
                try:
                    content = await extractor(context_stream, None)
                    if content:
                        available_artifacts.append(
                            {
                                "artifact_type": artifact_type,
                                "available": True,
                                "content_preview": (
                                    str(content)[:100] + "..."
                                    if len(str(content)) > 100
                                    else str(content)
                                ),
                                "extractor": extractor.__name__,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Artifact {artifact_type} not available: {e}")

            return available_artifacts

        except Exception as e:
            logger.error(f"❌ Failed to list artifacts for {trace_id}: {e}")
            return []

    async def check_trace_health(self, trace_id: str) -> Dict[str, Any]:
        """
        🏥 Trace Health Check

        Verifies trace exists and provides health information.
        """

        try:
            context_stream_data = await self._fetch_context_stream_from_db(trace_id)

            if not context_stream_data:
                return {"exists": False, "status": "not_found"}

            return {
                "exists": True,
                "status": "healthy",
                "total_events": context_stream_data.get("total_events", 0),
                "completion_status": context_stream_data.get("final_status", "unknown"),
                "last_activity": context_stream_data.get("completed_at"),
                "engagement_type": context_stream_data.get("engagement_type"),
                "case_id": context_stream_data.get("case_id"),
            }

        except Exception as e:
            return {"exists": False, "status": "error", "error": str(e)}

    async def _fetch_context_stream_from_db(
        self, trace_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        💾 Database Connection and Retrieval - ABSOLUTE INTEGRITY

        Fetches the complete context stream record from SQLite database.
        ZERO TOLERANCE: Returns exact match ONLY or raises CriticalArtifactNotFoundError.
        NO FALLBACKS. NO GUESSING. NO DECEPTION.
        """

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # ONLY EXACT MATCH - No fallback logic
            # Convert UUID to string if necessary
            trace_id_str = str(trace_id)
            result = cursor.execute(
                """
                SELECT 
                    trace_id, 
                    context_stream, 
                    total_events, 
                    total_tokens, 
                    final_status,
                    completed_at,
                    engagement_type,
                    case_id
                FROM context_streams 
                WHERE trace_id = ?
            """,
                (trace_id_str,),
            ).fetchall()

            conn.close()

            # HONEST FAILURE: Exact count validation
            if len(result) == 0:
                logger.error(f"🚨 ZERO records found for trace_id {trace_id_str}")
                raise RuntimeError(
                    f"CriticalArtifactNotFoundError: trace_id {trace_id_str} does not exist in context_streams database"
                )
            elif len(result) > 1:
                logger.error(
                    f"🚨 MULTIPLE records found for trace_id {trace_id_str}: {len(result)} matches"
                )
                raise RuntimeError(
                    f"CriticalArtifactIntegrityError: trace_id {trace_id_str} has {len(result)} duplicate records - database corruption detected"
                )

            # EXACTLY ONE MATCH - Proceed with integrity
            record = result[0]
            logger.info(
                f"✅ EXACT MATCH: trace_id {trace_id_str} found with {record[2]} events"
            )

            return {
                "trace_id": record[0],
                "context_stream": record[1],
                "total_events": record[2],
                "total_tokens": record[3],
                "final_status": record[4],
                "completed_at": record[5],
                "engagement_type": record[6],
                "case_id": record[7],
            }

        except RuntimeError:
            # Re-raise integrity errors as-is
            raise
        except Exception as e:
            trace_id_str = (
                str(trace_id) if "trace_id_str" not in locals() else trace_id_str
            )
            logger.error(f"❌ Database connection failed for {trace_id_str}: {e}")
            raise RuntimeError(
                f"CriticalDatabaseError: Failed to query trace_id {trace_id_str}: {e}"
            )

    # ==========================================================================
    # ARTIFACT EXTRACTORS - Specialized parsers for different artifact types
    # ==========================================================================

    async def _extract_final_report(
        self, context_stream: Dict[str, Any], event_id: Optional[str] = None
    ) -> Optional[str]:
        """OPERATION EXTRACT THE TRUTH - Find REAL LLM-generated final report."""

        events = context_stream.get("events", [])

        # SURGICAL MANDATE: Look for final_report_generated event
        for event in reversed(events):
            event_type = event.get("event_type", "")
            data = event.get("data", {})

            # PRIMARY TARGET: final_report_generated event
            if event_type == "final_report_generated":
                # Extract the markdown report from the data payload
                # New schema (preferred): { data: { content: <markdown>, ... } }
                if "content" in data and isinstance(data["content"], str):
                    report_content = data["content"]
                    if len(str(report_content)) > 500:  # Must be substantial
                        logger.info(
                            f"✅ Found real final_report_generated.content with {len(str(report_content))} chars"
                        )
                        return str(report_content)

                # Legacy field: explicit markdown_report
                if "markdown_report" in data:
                    report_content = data["markdown_report"]
                    if len(str(report_content)) > 500:  # Must be substantial
                        logger.info(
                            f"✅ Found real final_report_generated.markdown_report with {len(str(report_content))} chars"
                        )
                        return str(report_content)

                # Other tolerated legacy fields
                for field in ["final_report", "report", "analysis", "output"]:
                    if field in data and len(str(data[field])) > 500:
                        logger.info(
                            f"✅ Found real content in final_report_generated.{field}: {len(str(data[field]))} chars"
                        )
                        return str(data[field])

            # SECONDARY TARGETS: Other events that might contain real LLM output
            output_event_types = [
                "analysis_complete",
                "synthesis_generated",
                "report_generated",
                "llm_response_complete",
                "final_analysis_complete",
                "strategic_analysis_complete",
            ]

            if event_type in output_event_types:
                # Look for substantial content in these events
                output_fields = [
                    "output",
                    "result",
                    "analysis",
                    "report",
                    "response",
                    "content",
                    "synthesis",
                ]
                for field in output_fields:
                    if field in data and len(str(data[field])) > 500:
                        logger.info(
                            f"✅ Found real content in {event_type}.{field}: {len(str(data[field]))} chars"
                        )
                        return str(data[field])

        # CRITICAL ERROR: No real LLM-generated content found
        event_types = [e.get("event_type") for e in events]

        error_msg = f"""CriticalArtifactNotFoundError: No final_report_generated event found in context stream.
        
FORENSIC ANALYSIS:
        - Total events examined: {len(events)}
        - Event types found: {list(set(event_types))}
        - No substantial LLM-generated content detected
        
ROOT CAUSE: The pipeline is not saving actual LLM outputs to the context stream.
The context stream contains only infrastructure events, not the actual analysis results.
        
This exposes a fundamental flaw: The pipeline completes successfully but fails to 
capture its own outputs. We cannot validate what doesn't exist.
        
RECOMMENDATION: Fix the pipeline to emit final_report_generated events with the 
actual LLM-generated markdown content in the data payload.
        """

        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def _extract_analysis_output(
        self, context_stream: Dict[str, Any], event_id: Optional[str] = None
    ) -> Optional[str]:
        """Extract general analysis output from context stream."""

        events = context_stream.get("events", [])

        for event in reversed(events):
            data = event.get("data", {})

            # Look for analysis outputs
            if "analysis_output" in data:
                return data["analysis_output"]
            if "result" in data and isinstance(data["result"], str):
                return data["result"]
            if "response" in data and isinstance(data["response"], str):
                return data["response"]

        return None

    async def _extract_intermediate_analysis(
        self, context_stream: Dict[str, Any], event_id: Optional[str] = None
    ) -> Optional[str]:
        """Extract intermediate analysis results."""

        events = context_stream.get("events", [])
        intermediate_results = []

        for event in events:
            event_type = event.get("event_type", "")
            data = event.get("data", {})

            if "intermediate" in event_type.lower() or "partial" in event_type.lower():
                if "analysis" in data:
                    intermediate_results.append(data["analysis"])
                if "result" in data:
                    intermediate_results.append(data["result"])

        return (
            "\n\n---\n\n".join(intermediate_results) if intermediate_results else None
        )

    async def _extract_consultant_recommendation(
        self, context_stream: Dict[str, Any], event_id: Optional[str] = None
    ) -> Optional[str]:
        """Extract consultant recommendations."""

        events = context_stream.get("events", [])

        for event in reversed(events):
            event_type = event.get("event_type", "")
            data = event.get("data", {})

            if "consultant" in event_type.lower():
                if "recommendation" in data:
                    return data["recommendation"]
                if "analysis" in data:
                    return data["analysis"]

        return None

    async def _extract_devils_advocate_challenges(
        self, context_stream: Dict[str, Any], event_id: Optional[str] = None
    ) -> Optional[str]:
        """Extract Devil's Advocate challenges."""

        events = context_stream.get("events", [])

        for event in reversed(events):
            event_type = event.get("event_type", "")
            data = event.get("data", {})

            if (
                "devils_advocate" in event_type.lower()
                or "challenge" in event_type.lower()
            ):
                if "challenges" in data:
                    return data["challenges"]
                if "critique" in data:
                    return data["critique"]
                if "analysis" in data:
                    return data["analysis"]

        return None

    async def _extract_synthesis_result(
        self, context_stream: Dict[str, Any], event_id: Optional[str] = None
    ) -> Optional[str]:
        """Extract synthesis results."""

        events = context_stream.get("events", [])

        for event in reversed(events):
            event_type = event.get("event_type", "")
            data = event.get("data", {})

            if "synthesis" in event_type.lower():
                if "synthesis_result" in data:
                    return data["synthesis_result"]
                if "result" in data:
                    return data["result"]

        return None

    async def _extract_raw_llm_response(
        self, context_stream: Dict[str, Any], event_id: Optional[str] = None
    ) -> Optional[str]:
        """Extract raw LLM responses."""

        events = context_stream.get("events", [])

        for event in reversed(events):
            data = event.get("data", {})

            if "llm_response" in data:
                if isinstance(data["llm_response"], dict):
                    return data["llm_response"].get("raw_text") or data[
                        "llm_response"
                    ].get("content")
                return data["llm_response"]

            if "raw_response" in data:
                return data["raw_response"]

        return None

    async def _extract_executive_summary(
        self, context_stream: Dict[str, Any], event_id: Optional[str] = None
    ) -> Optional[str]:
        """Extract executive summary."""

        events = context_stream.get("events", [])

        for event in reversed(events):
            data = event.get("data", {})

            if "executive_summary" in data:
                return data["executive_summary"]
            if "summary" in data:
                return data["summary"]

        return None
