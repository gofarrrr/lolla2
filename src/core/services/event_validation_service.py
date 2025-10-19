"""
Event Validation Service

Extracted from UnifiedContextStream (Task 3.0)
Handles event schema validation, transition validation, and PII scrubbing.

Created: 2025-10-18
Campaign: Operation Lean
Original Lines: 336-458 from unified_context_stream.py
"""

import re
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class EventValidationService:
    """
    Event validation and PII scrubbing service.

    Responsibilities:
    - Load and validate against event allowlist
    - Validate event schemas
    - Validate event state transitions
    - Scrub PII from event data
    """

    def __init__(
        self,
        allowlist_path: Optional[Path] = None,
        pii_engine: Optional[Any] = None,
        strict_validation_enabled: bool = True
    ):
        """
        Initialize event validation service.

        Args:
            allowlist_path: Path to event_allowlist.yaml (None = auto-detect)
            pii_engine: Optional PII redaction engine
            strict_validation_enabled: Enable strict validation
        """
        self.strict_validation_enabled = strict_validation_enabled
        self.pii_engine = pii_engine
        self.pii_redaction_enabled = pii_engine is not None
        self.event_allowlist: Dict[str, Any] = {}

        # Auto-detect allowlist path if not provided
        if allowlist_path is None:
            allowlist_path = Path(__file__).parent.parent / "event_allowlist.yaml"

        self._load_event_allowlist(allowlist_path)

    def _load_event_allowlist(self, allowlist_path: Path) -> None:
        """
        Load event transition allowlist from YAML file.

        Args:
            allowlist_path: Path to YAML allowlist file
        """
        try:
            import yaml

            with open(allowlist_path, "r") as f:
                self.event_allowlist = yaml.safe_load(f) or {}
            logger.info(f"✅ Loaded event allowlist with {len(self.event_allowlist)} event types")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load event allowlist: {e}. Strict validation disabled.")
            self.strict_validation_enabled = False

    def validate_event_schema(self, event_type: Any, data: Dict[str, Any]) -> bool:
        """
        Validate event payload against schema.

        Args:
            event_type: Type of event to validate (ContextEventType or str)
            data: Event data dictionary

        Returns:
            True if schema is valid, False otherwise
        """
        if not self.strict_validation_enabled:
            return True

        try:
            from src.core.event_schemas import validate_event_payload

            # Extract string value from enum if needed
            event_type_value = event_type.value if hasattr(event_type, 'value') else str(event_type)

            is_valid, errors = validate_event_payload(event_type_value, data)
            if not is_valid:
                logger.warning(
                    f"❌ Event schema validation failed for {event_type_value}: {errors}"
                )
            return is_valid
        except ImportError:
            logger.warning("⚠️  event_schemas module not available, skipping schema validation")
            return True

    def validate_event_transition(
        self,
        event_type: Any,
        previous_event_type: Optional[Any] = None
    ) -> bool:
        """
        Validate event transition based on allowlist.

        Args:
            event_type: Current event type
            previous_event_type: Previous event type (None if first event)

        Returns:
            True if transition is valid, False otherwise
        """
        if not self.strict_validation_enabled or not self.event_allowlist:
            return True

        # First event - allow any
        if previous_event_type is None:
            return True

        # Extract string values from enums if needed
        last_event_type_str = (
            previous_event_type.value
            if hasattr(previous_event_type, 'value')
            else str(previous_event_type)
        )
        current_event_type_str = (
            event_type.value if hasattr(event_type, 'value') else str(event_type)
        )

        # Check if transition is allowed
        allowed_transitions = self.event_allowlist.get(last_event_type_str.upper(), {})
        allowed_next = allowed_transitions.get("allowed_next", [])

        # Special cases
        if "*" in allowed_next:
            # Any transition allowed
            return True

        if current_event_type_str.upper() in [t.upper() for t in allowed_next]:
            return True

        logger.warning(
            f"❌ Invalid event transition: {last_event_type_str} → {current_event_type_str}. "
            f"Allowed: {allowed_next}"
        )
        return False

    def is_event_allowed(self, event_type: Any) -> bool:
        """
        Check if event type is in the allowlist.

        Args:
            event_type: Event type to check

        Returns:
            True if event is allowed, False if not or allowlist not loaded
        """
        if not self.event_allowlist:
            # No allowlist loaded - allow all
            return True

        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        return event_type_str.upper() in self.event_allowlist

    def scrub_pii(self, data_string: str) -> str:
        """
        ENTERPRISE SECURITY (Phase 6): Enhanced PII scrubber.
        Redacts emails, phones, SSNs, credit cards, API keys, and more.

        Falls back to legacy regex patterns if engine unavailable.

        Args:
            data_string: String potentially containing PII

        Returns:
            String with PII replaced by [REDACTED]
        """
        if not isinstance(data_string, str):
            return data_string

        # Use new PII engine if available
        if self.pii_redaction_enabled and self.pii_engine:
            try:
                result = self.pii_engine.redact(data_string)
                return result.redacted_text
            except Exception as e:
                logger.warning(f"⚠️ PII engine failed, using legacy patterns: {e}")

        # Legacy fallback patterns
        email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        phone_regex = r"\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b"
        ssn_regex = r"\b\d{3}-\d{2}-\d{4}\b"

        scrubbed = re.sub(email_regex, "[REDACTED_EMAIL]", data_string)
        scrubbed = re.sub(phone_regex, "[REDACTED_PHONE]", scrubbed)
        scrubbed = re.sub(ssn_regex, "[REDACTED_SSN]", scrubbed)

        return scrubbed

    def scrub_structure(self, value: Any) -> Any:
        """
        PLATFORM HARDENING: Recursively scrub PII from nested structures.
        Handles dictionaries, lists, and strings.

        Args:
            value: Any value (dict, list, str, or primitive)

        Returns:
            Value with all PII scrubbed
        """
        if isinstance(value, dict):
            return {k: self.scrub_structure(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.scrub_structure(v) for v in value]
        elif isinstance(value, str):
            return self.scrub_pii(value)
        else:
            return value
