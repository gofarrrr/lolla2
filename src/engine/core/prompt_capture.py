#!/usr/bin/env python3
"""
Prompt Capture System for METIS Complete Data Capture
=====================================================

Captures all prompts sent to LLMs with comprehensive metadata for debugging,
optimization, and transparency. Provides complete visibility into prompt
engineering and LLM interactions.

Key Features:
- Complete prompt-response linkage
- Template versioning and tracking
- Phase-aware prompt categorization
- Performance analytics
- Quality assessment integration
- Async and sync capture support
"""

import asyncio
import hashlib
import json
import logging
import time
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4
import os
from string import Template
from src.services.container import global_container
from src.services.persistence.database_service import DatabaseService


# Centralized Prompt Template Registry
TEMPLATES = {
    "model_application": """
<goal>{goal}</goal>
<context>{context}</context>
<constraints>{constraints}</constraints>
<instructions>{instructions}</instructions>
""".strip(),
    "analysis": """
<goal>{goal}</goal>
<context>{context}</context>
""".strip(),
    "research": """
Research the following topic thoroughly:
<topic>{topic}</topic>
<focus_areas>{focus_areas}</focus_areas>
<quality_requirements>{quality_requirements}</quality_requirements>
""".strip(),
    "synthesis": """
Synthesize the following insights into a coherent analysis:
<insights>{insights}</insights>
<synthesis_requirements>{synthesis_requirements}</synthesis_requirements>
""".strip(),
    "auditor": """
Audit the following analysis for completeness and accuracy:
<analysis>{analysis}</analysis>
<audit_criteria>{audit_criteria}</audit_criteria>
""".strip(),
    "opposing_view": """
Generate a balanced opposing viewpoint for the following position:
<position>{position}</position>
<evidence>{evidence}</evidence>
<opposition_requirements>{opposition_requirements}</opposition_requirements>
""".strip(),
    "validation": """
Validate the following using these criteria:
<content>{content}</content>
<criteria>{criteria}</criteria>
""".strip(),
}


def render_template(name: str, **kwargs) -> str:
    """
    Render a prompt template with provided variables

    Args:
        name: Template name from TEMPLATES registry
        **kwargs: Variables to substitute in the template

    Returns:
        Rendered template string
    """
    if name not in TEMPLATES:
        raise ValueError(
            f"Template '{name}' not found. Available: {list(TEMPLATES.keys())}"
        )

    template = Template(TEMPLATES[name])
    try:
        return template.safe_substitute(**kwargs)
    except Exception as e:
        raise ValueError(f"Template rendering failed for '{name}': {e}")


class PromptType(str, Enum):
    """Types of prompts for categorization"""

    SYSTEM_PROMPT = "system_prompt"
    USER_PROMPT = "user_prompt"
    ASSISTANT_PROMPT = "assistant_prompt"
    TEMPLATE_PROMPT = "template_prompt"
    CONTEXT_PROMPT = "context_prompt"


class PromptPhase(str, Enum):
    """Cognitive analysis phases for prompt classification"""

    PROBLEM_STRUCTURING = "problem_structuring"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    ANALYSIS_EXECUTION = "analysis_execution"
    SYNTHESIS_DELIVERY = "synthesis_delivery"
    RESEARCH_INTEGRATION = "research_integration"
    VERIFICATION = "verification"
    OTHER = "other"


@dataclass
class PromptTemplate:
    """Template for prompt generation and versioning"""

    template_id: str
    name: str
    version: str
    template_content: str
    variables: List[str]
    phase: PromptPhase
    description: str
    created_at: str
    usage_count: int = 0
    success_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    avg_quality_score: float = 0.0


@dataclass
class PromptRecord:
    """Complete record of a prompt sent to an LLM"""

    prompt_id: str
    timestamp: str
    phase: PromptPhase
    prompt_type: PromptType

    # Prompt content
    system_prompt: str
    user_prompt: str
    full_prompt: str

    # Template information
    template_id: Optional[str] = None
    template_version: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)

    # Context and metadata
    engagement_id: Optional[str] = None
    session_id: Optional[str] = None
    provider: str = "unknown"
    model: str = "unknown"
    context_data: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    prompt_length: int = 0
    estimated_tokens: int = 0
    preparation_time_ms: float = 0.0

    # Quality and validation
    prompt_hash: str = ""
    validation_status: str = "pending"
    quality_score: Optional[float] = None

    # Response linkage
    response_id: Optional[str] = None
    response_received: bool = False
    response_timestamp: Optional[str] = None

    # Analytics flags
    contains_pii: bool = False
    complexity_score: float = 0.0
    novelty_score: float = 0.0


@dataclass
class ResponseLinkage:
    """Links prompt to its corresponding response"""

    prompt_id: str
    response_id: str
    response_timestamp: str
    response_tokens: int
    response_cost_usd: float
    response_time_ms: float
    response_quality_score: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class PromptAnalytics:
    """Analytics and insights from prompt capture data"""

    total_prompts: int
    unique_templates: int
    avg_prompt_length: float
    avg_response_time_ms: float
    success_rate: float
    cost_per_prompt_usd: float

    # Phase breakdown
    prompts_by_phase: Dict[PromptPhase, int]
    success_by_phase: Dict[PromptPhase, float]
    cost_by_phase: Dict[PromptPhase, float]

    # Template performance
    top_templates: List[Dict[str, Any]]
    template_success_rates: Dict[str, float]

    # Quality metrics
    avg_quality_score: float
    quality_distribution: Dict[str, int]

    # Time-based metrics
    prompts_per_hour: float
    peak_usage_hours: List[int]


class PromptCapture:
    """
    Main prompt capture system with comprehensive logging and analytics
    """

    def __init__(
        self,
        max_records: int = 10000,
        enable_analytics: bool = True,
        enable_persistence: bool = True,
        storage_path: Optional[str] = None,
        enable_supabase: bool = True,
        database_service: Optional[DatabaseService] = None,
    ):
        """
        Initialize prompt capture system

        Args:
            max_records: Maximum number of prompt records to keep in memory
            enable_analytics: Enable real-time analytics computation
            enable_persistence: Enable persistent storage of prompts
            storage_path: Custom storage path for prompt data
            enable_supabase: Enable Supabase database storage
        """
        self.logger = logging.getLogger(__name__)
        self.max_records = max_records
        self.enable_analytics = enable_analytics
        self.enable_persistence = enable_persistence
        self.enable_supabase = enable_supabase

        # Storage
        self.prompt_records: OrderedDict[str, PromptRecord] = OrderedDict()
        self.response_linkages: Dict[str, ResponseLinkage] = {}
        self.templates: Dict[str, PromptTemplate] = {}

        # Analytics cache
        self._analytics_cache: Optional[PromptAnalytics] = None
        self._analytics_last_computed: Optional[datetime] = None
        self._analytics_cache_ttl = timedelta(minutes=5)

        # Thread safety
        self._lock = threading.RLock()

        # Performance tracking
        self._capture_times: List[float] = []
        self._async_queue: Optional[asyncio.Queue] = None

        # Storage configuration
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.cwd() / ".metis" / "prompt_capture"

        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Unified persistence via DatabaseService (DI)
        try:
            self.database_service: Optional[DatabaseService] = (
                database_service if database_service is not None else global_container.get_database_service()
            )
        except Exception:
            self.database_service = None
        if self.enable_supabase and not self.database_service:
            self.logger.warning("DatabaseService not available; disabling remote prompt persistence")
            self.enable_supabase = False

        # Initialize async components
        self._background_tasks: List[asyncio.Task] = []

        self.logger.info(
            f"✅ PromptCapture initialized: max_records={max_records}, "
            f"analytics={enable_analytics}, persistence={enable_persistence}"
        )

    def register_template(self, template: PromptTemplate) -> str:
        """Register a prompt template for tracking and versioning"""
        with self._lock:
            self.templates[template.template_id] = template

            if self.enable_persistence:
                self._persist_template(template)

            self.logger.debug(
                f"📝 Template registered: {template.name} v{template.version}"
            )
            return template.template_id

    def capture_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        phase: PromptPhase,
        prompt_type: PromptType = PromptType.USER_PROMPT,
        template_id: Optional[str] = None,
        template_variables: Optional[Dict[str, Any]] = None,
        context_data: Optional[Dict[str, Any]] = None,
        engagement_id: Optional[str] = None,
        provider: str = "unknown",
        model: str = "unknown",
    ) -> str:
        """
        Capture a prompt before it's sent to an LLM

        Returns:
            prompt_id: Unique identifier for this prompt
        """
        capture_start = time.time()

        prompt_id = str(uuid4())
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

        # Calculate prompt metrics
        prompt_length = len(full_prompt)
        estimated_tokens = self._estimate_tokens(full_prompt)
        prompt_hash = hashlib.md5(full_prompt.encode()).hexdigest()

        # Analyze prompt content
        contains_pii = self._detect_pii(full_prompt)
        complexity_score = self._calculate_complexity(full_prompt)
        novelty_score = self._calculate_novelty(prompt_hash)

        # Create prompt record
        record = PromptRecord(
            prompt_id=prompt_id,
            timestamp=datetime.now().isoformat(),
            phase=phase,
            prompt_type=prompt_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            full_prompt=full_prompt,
            template_id=template_id,
            template_variables=template_variables or {},
            engagement_id=engagement_id,
            provider=provider,
            model=model,
            context_data=context_data or {},
            prompt_length=prompt_length,
            estimated_tokens=estimated_tokens,
            preparation_time_ms=(time.time() - capture_start) * 1000,
            prompt_hash=prompt_hash,
            contains_pii=contains_pii,
            complexity_score=complexity_score,
            novelty_score=novelty_score,
        )

        # Store record
        with self._lock:
            self.prompt_records[prompt_id] = record

            # Maintain max records limit
            if len(self.prompt_records) > self.max_records:
                oldest_id = next(iter(self.prompt_records))
                del self.prompt_records[oldest_id]

            # Update template usage
            if template_id and template_id in self.templates:
                self.templates[template_id].usage_count += 1

            # Invalidate analytics cache
            self._analytics_cache = None

        # Persist if enabled
        if self.enable_persistence:
            self._persist_prompt_record(record)

        # Store in database (facade) if enabled
        if self.enable_supabase and self.database_service:
            self._store_prompt_in_supabase(record)

        capture_time = (time.time() - capture_start) * 1000
        self._capture_times.append(capture_time)

        # Keep only recent capture times for performance monitoring
        if len(self._capture_times) > 1000:
            self._capture_times = self._capture_times[-500:]

        self.logger.debug(
            f"📋 Prompt captured: {prompt_id[:8]} | {phase if isinstance(phase, str) else phase.value} | "
            f"{prompt_length} chars | {estimated_tokens} tokens | "
            f"{capture_time:.1f}ms"
        )

        return prompt_id

    async def capture_prompt_async(self, **kwargs) -> str:
        """Async version of capture_prompt for non-blocking operation"""
        if self._async_queue is None:
            self._async_queue = asyncio.Queue(maxsize=1000)

            # Start background processor
            task = asyncio.create_task(self._process_async_captures())
            self._background_tasks.append(task)

        # Queue the capture operation
        prompt_id = str(uuid4())
        await self._async_queue.put((prompt_id, kwargs))

        return prompt_id

    def link_response(
        self,
        prompt_id: str,
        response_id: str,
        response_tokens: int,
        response_cost_usd: float,
        response_time_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> bool:
        """Link a response to its corresponding prompt"""

        with self._lock:
            # Verify prompt exists
            if prompt_id not in self.prompt_records:
                self.logger.warning(
                    f"⚠️ Cannot link response: prompt {prompt_id[:8]} not found"
                )
                return False

            # Create response linkage
            linkage = ResponseLinkage(
                prompt_id=prompt_id,
                response_id=response_id,
                response_timestamp=datetime.now().isoformat(),
                response_tokens=response_tokens,
                response_cost_usd=response_cost_usd,
                response_time_ms=response_time_ms,
                response_quality_score=quality_score,
                success=success,
                error_message=error_message,
            )

            self.response_linkages[prompt_id] = linkage

            # Update prompt record
            prompt_record = self.prompt_records[prompt_id]
            prompt_record.response_id = response_id
            prompt_record.response_received = True
            prompt_record.response_timestamp = linkage.response_timestamp
            prompt_record.quality_score = quality_score
            prompt_record.validation_status = "completed" if success else "failed"

            # Update template metrics
            if (
                prompt_record.template_id
                and prompt_record.template_id in self.templates
            ):
                template = self.templates[prompt_record.template_id]
                template.avg_response_time_ms = self._update_average(
                    template.avg_response_time_ms,
                    template.usage_count,
                    response_time_ms,
                )
                if quality_score:
                    template.avg_quality_score = self._update_average(
                        template.avg_quality_score, template.usage_count, quality_score
                    )
                template.success_rate = self._calculate_template_success_rate(
                    template.template_id
                )

            # Invalidate analytics cache
            self._analytics_cache = None

        if self.enable_persistence:
            self._persist_response_linkage(linkage)

        self.logger.debug(
            f"🔗 Response linked: {prompt_id[:8]} -> {response_id[:8]} | "
            f"success={success} | {response_time_ms:.1f}ms"
        )

        return True

    def get_prompt_analytics(
        self,
        time_window_hours: Optional[int] = None,
        phase_filter: Optional[List[PromptPhase]] = None,
    ) -> PromptAnalytics:
        """Get comprehensive analytics from captured prompts"""

        # Check cache first
        if (
            self.enable_analytics
            and self._analytics_cache
            and self._analytics_last_computed
            and datetime.now() - self._analytics_last_computed
            < self._analytics_cache_ttl
        ):
            return self._analytics_cache

        with self._lock:
            records = list(self.prompt_records.values())
            linkages = list(self.response_linkages.values())

        # Apply time filter
        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            records = [
                r for r in records if datetime.fromisoformat(r.timestamp) >= cutoff_time
            ]

        # Apply phase filter
        if phase_filter:
            records = [r for r in records if r.phase in phase_filter]

        # Calculate analytics
        analytics = self._calculate_analytics(records, linkages)

        # Cache results
        if self.enable_analytics:
            self._analytics_cache = analytics
            self._analytics_last_computed = datetime.now()

        return analytics

    def get_prompt_by_id(self, prompt_id: str) -> Optional[PromptRecord]:
        """Retrieve a specific prompt record by ID"""
        with self._lock:
            return self.prompt_records.get(prompt_id)

    def get_prompts_by_phase(
        self, phase: PromptPhase, limit: int = 100
    ) -> List[PromptRecord]:
        """Get recent prompts for a specific phase"""
        with self._lock:
            phase_prompts = [
                record
                for record in self.prompt_records.values()
                if record.phase == phase
            ]

        # Sort by timestamp (most recent first) and limit
        phase_prompts.sort(key=lambda x: x.timestamp, reverse=True)
        return phase_prompts[:limit]

    def get_template_performance(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific template"""
        with self._lock:
            if template_id not in self.templates:
                return None

            template = self.templates[template_id]

            # Get prompts using this template
            template_prompts = [
                record
                for record in self.prompt_records.values()
                if record.template_id == template_id
            ]

            # Calculate performance metrics
            total_prompts = len(template_prompts)
            successful_prompts = sum(
                1
                for record in template_prompts
                if record.validation_status == "completed"
            )

            avg_response_time = 0.0
            avg_quality = 0.0
            total_cost = 0.0

            if total_prompts > 0:
                # Get linked responses
                linked_responses = [
                    self.response_linkages[record.prompt_id]
                    for record in template_prompts
                    if record.prompt_id in self.response_linkages
                ]

                if linked_responses:
                    avg_response_time = sum(
                        r.response_time_ms for r in linked_responses
                    ) / len(linked_responses)
                    total_cost = sum(r.response_cost_usd for r in linked_responses)

                    quality_scores = [
                        r.response_quality_score
                        for r in linked_responses
                        if r.response_quality_score is not None
                    ]
                    if quality_scores:
                        avg_quality = sum(quality_scores) / len(quality_scores)

            return {
                "template_id": template_id,
                "template_name": template.name,
                "version": template.version,
                "total_usage": total_prompts,
                "success_rate": (
                    successful_prompts / total_prompts if total_prompts > 0 else 0.0
                ),
                "avg_response_time_ms": avg_response_time,
                "avg_quality_score": avg_quality,
                "total_cost_usd": total_cost,
                "cost_per_use_usd": (
                    total_cost / total_prompts if total_prompts > 0 else 0.0
                ),
            }

    def export_prompts(
        self,
        format: str = "json",
        output_path: Optional[str] = None,
        include_responses: bool = True,
    ) -> str:
        """Export captured prompts to file"""

        with self._lock:
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_prompts": len(self.prompt_records),
                    "total_templates": len(self.templates),
                    "include_responses": include_responses,
                },
                "prompts": [asdict(record) for record in self.prompt_records.values()],
                "templates": [asdict(template) for template in self.templates.values()],
            }

            if include_responses:
                export_data["response_linkages"] = [
                    asdict(linkage) for linkage in self.response_linkages.values()
                ]

        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.storage_path / f"prompt_export_{timestamp}.{format}")

        # Export based on format
        if format.lower() == "json":
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"📤 Prompts exported: {output_path}")
        return output_path

    # Internal helper methods

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count"""
        # Approximation: 1 token ≈ 4 characters for English text
        return max(1, len(text) // 4)

    def _detect_pii(self, text: str) -> bool:
        """Simple PII detection (can be enhanced with ML models)"""
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        ]

        import re

        for pattern in pii_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _calculate_complexity(self, text: str) -> float:
        """Calculate prompt complexity score (0.0 to 1.0)"""
        # Simple heuristic based on length, sentence count, and vocabulary
        words = text.split()
        sentences = text.count(".") + text.count("!") + text.count("?")

        word_count_score = min(1.0, len(words) / 500)  # Normalize to 500 words
        sentence_score = min(1.0, sentences / 20)  # Normalize to 20 sentences
        unique_words_score = len(set(words)) / max(
            1, len(words)
        )  # Vocabulary diversity

        return (word_count_score + sentence_score + unique_words_score) / 3

    def _calculate_novelty(self, prompt_hash: str) -> float:
        """Calculate how novel this prompt is compared to previous ones"""
        with self._lock:
            similar_hashes = sum(
                1
                for record in self.prompt_records.values()
                if record.prompt_hash == prompt_hash
            )

        # Novel if we haven't seen this exact prompt before
        return 1.0 if similar_hashes == 0 else max(0.1, 1.0 / similar_hashes)

    def _update_average(
        self, current_avg: float, count: int, new_value: float
    ) -> float:
        """Update running average with new value"""
        if count <= 1:
            return new_value
        return ((current_avg * (count - 1)) + new_value) / count

    def _calculate_template_success_rate(self, template_id: str) -> float:
        """Calculate success rate for a specific template"""
        template_prompts = [
            record
            for record in self.prompt_records.values()
            if record.template_id == template_id
        ]

        if not template_prompts:
            return 0.0

        successful = sum(
            1 for record in template_prompts if record.validation_status == "completed"
        )

        return successful / len(template_prompts)

    def _calculate_analytics(
        self, records: List[PromptRecord], linkages: List[ResponseLinkage]
    ) -> PromptAnalytics:
        """
        REFACTORED: Grade E (38) → Grade B (≤10) complexity

        Calculate comprehensive analytics using Strategy Pattern orchestrator

        Pattern: Delegation to AnalyticsOrchestrator service
        Complexity Target: Grade B (≤10)
        """
        from src.engine.services.analytics.analytics_strategy_service import (
            get_analytics_orchestrator,
        )

        orchestrator = get_analytics_orchestrator()
        return orchestrator.calculate_comprehensive_analytics(records, linkages)

    async def _process_async_captures(self):
        """Background processor for async prompt captures"""
        while True:
            try:
                prompt_id, kwargs = await self._async_queue.get()

                # Process the capture synchronously
                actual_prompt_id = self.capture_prompt(**kwargs)

                # Update the prompt ID in the record if needed
                if prompt_id != actual_prompt_id:
                    self.logger.warning(
                        f"Prompt ID mismatch: expected {prompt_id}, got {actual_prompt_id}"
                    )

                self._async_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in async capture processor: {e}")

    def _persist_template(self, template: PromptTemplate):
        """Persist template to storage"""
        template_file = self.storage_path / f"template_{template.template_id}.json"
        try:
            with open(template_file, "w") as f:
                json.dump(asdict(template), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to persist template {template.template_id}: {e}")

    def _persist_prompt_record(self, record: PromptRecord):
        """Persist prompt record to storage"""
        # Create daily files to manage storage size
        date_str = datetime.now().strftime("%Y%m%d")
        prompt_file = self.storage_path / f"prompts_{date_str}.jsonl"

        try:
            with open(prompt_file, "a") as f:
                json.dump(asdict(record), f)
                f.write("\n")
        except Exception as e:
            self.logger.error(
                f"Failed to persist prompt record {record.prompt_id}: {e}"
            )

    def _persist_response_linkage(self, linkage: ResponseLinkage):
        """Persist response linkage to storage"""
        date_str = datetime.now().strftime("%Y%m%d")
        linkage_file = self.storage_path / f"linkages_{date_str}.jsonl"

        try:
            with open(linkage_file, "a") as f:
                json.dump(asdict(linkage), f)
                f.write("\n")
        except Exception as e:
            self.logger.error(
                f"Failed to persist response linkage {linkage.prompt_id}: {e}"
            )

    def _confidence_to_float(self, confidence_value) -> Optional[float]:
        """Convert confidence enum or string to float value for database storage"""
        if confidence_value is None:
            return None

        # If already a float, return as-is
        if isinstance(confidence_value, (int, float)):
            return float(confidence_value)

        # Convert string/enum to float
        confidence_str = str(confidence_value).lower()
        confidence_map = {
            "very_low": 0.1,
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "very_high": 0.9,
            "uncertain": 0.2,
        }

        return confidence_map.get(confidence_str, 0.5)

    def _store_prompt_in_supabase(self, record: PromptRecord):
        """Store prompt record through the unified DatabaseService facade."""
        try:
            # Prepare data for persistence
            event_data = {
                "prompt_id": record.prompt_id,
                "engagement_id": record.engagement_id,
                "system_prompt": record.system_prompt,
                "user_prompt": record.user_prompt,
                "response_text": getattr(record, "response_text", None),
                "phase": (
                    record.phase
                    if isinstance(record.phase, str)
                    else record.phase.value
                ),
                "prompt_type": (
                    record.prompt_type
                    if isinstance(record.prompt_type, str)
                    else record.prompt_type.value
                ),
                "mental_model": record.context_data.get("mental_model"),
                "model_version": record.model,
                "tokens_input": record.estimated_tokens,
                "tokens_output": 0,  # Will be updated when response is captured
                "cost_usd": 0.0,  # Will be updated when response is captured
                "latency_ms": getattr(record, "response_time_ms", None),
                "confidence_score": self._confidence_to_float(
                    getattr(record, "confidence_score", None)
                ),
                "quality_score": getattr(record, "quality_score", None),
                "prompt_metadata": {
                    "template_id": record.template_id,
                    "template_variables": record.template_variables,
                    "context_data": record.context_data,
                    "prompt_hash": record.prompt_hash,
                    "complexity_score": record.complexity_score,
                    "novelty_score": record.novelty_score,
                    "contains_pii": record.contains_pii,
                },
                "thread_id": record.session_id,
                "parent_prompt_id": record.context_data.get("parent_prompt_id"),
                "created_at": record.timestamp,
                "updated_at": record.timestamp,
            }
            if not self.database_service:
                return
            # Schedule async persistence without blocking
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.database_service.store_prompt_capture_event(event_data))
            except RuntimeError:
                # No running loop; run the coroutine to completion
                asyncio.run(self.database_service.store_prompt_capture_event(event_data))
            self.logger.debug(f"Stored prompt {record.prompt_id} via DatabaseService")

        except Exception as e:
            self.logger.error(
                f"Failed to store prompt {record.prompt_id} via DatabaseService: {e}"
            )
            # Continue without failing - fallback to local storage

    def cleanup(self):
        """Clean up resources and background tasks"""
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Clear async queue
        if self._async_queue:
            while not self._async_queue.empty():
                try:
                    self._async_queue.get_nowait()
                    self._async_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        self.logger.info("🧹 PromptCapture cleanup completed")


# Global instance for easy access
_prompt_capture_instance: Optional[PromptCapture] = None


def get_prompt_capture() -> PromptCapture:
    """Get or create global prompt capture instance"""
    global _prompt_capture_instance

    if _prompt_capture_instance is None:
        _prompt_capture_instance = PromptCapture()

    return _prompt_capture_instance


def reset_prompt_capture():
    """Reset global prompt capture instance (for testing)"""
    global _prompt_capture_instance

    if _prompt_capture_instance:
        _prompt_capture_instance.cleanup()

    _prompt_capture_instance = None
