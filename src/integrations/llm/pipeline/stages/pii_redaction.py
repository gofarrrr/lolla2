"""
PII Redaction Pipeline Stage

Redacts personally identifiable information (PII) from messages before LLM call.

Extracted from: unified_client.py::_redact_pii_from_messages() (lines 1143-1169)
Design: Single Responsibility - Only handles PII redaction
Complexity Target: CC < 5
"""

from typing import List, Dict, Optional, Any
import logging

from ..context import LLMCallContext
from ..stage import PipelineStage


class PIIRedactionStage(PipelineStage):
    """
    Pipeline stage that redacts PII from user messages.

    This stage:
    1. Scans messages for PII patterns (email, phone, SSN, credit cards, etc.)
    2. Replaces PII with [REDACTED] tokens
    3. Returns context with redacted messages

    Behavior (extracted verbatim from unified_client.py):
    - Uses PII redaction engine to detect and redact
    - Only processes string content (skips dict/list content)
    - Fail-open: On errors, returns original messages unchanged
    - Logs redaction count for transparency

    Dependencies:
        - src.engine.privacy.pii_redaction_engine (PIIRedactionEngine instance)

    Attributes:
        pii_engine: PIIRedactionEngine instance (optional, can be None if disabled)
        enabled: Whether stage is enabled (inherited from PipelineStage)

    Example:
        ```python
        from src.engine.privacy.pii_redaction_engine import get_pii_redaction_engine

        engine = get_pii_redaction_engine()
        stage = PIIRedactionStage(pii_engine=engine)

        context = LLMCallContext(
            messages=[{"role": "user", "content": "My email is john@example.com"}],
            model="deepseek-chat",
            provider="deepseek",
            kwargs={}
        )

        new_context = await stage.execute(context)
        # new_context.modified_messages[0]["content"] == "My email is [REDACTED]"
        ```

    References:
        - Original: unified_client.py lines 491-492, 1143-1169
        - ADR: docs/adr/ADR-001-unified-client-pipeline-refactoring.md
    """

    def __init__(
        self,
        pii_engine: Optional[Any] = None,
        enabled: bool = True
    ):
        """
        Initialize PII redaction stage.

        Args:
            pii_engine: PIIRedactionEngine instance (or None to disable)
            enabled: Whether stage is enabled

        Note:
            If pii_engine is None, stage becomes a no-op (returns context unchanged).
        """
        super().__init__(enabled=enabled)
        self.pii_engine = pii_engine

    @property
    def name(self) -> str:
        """Stage name for logging and metadata."""
        return "PIIRedaction"

    async def execute(self, context: LLMCallContext) -> LLMCallContext:
        """
        Redact PII from messages.

        This implementation is extracted verbatim from unified_client.py
        with minimal changes to adapt to pipeline context pattern.

        Args:
            context: Input context with messages to redact

        Returns:
            New context with redacted messages (if any PII found)

        Workflow:
            1. Get effective messages from context
            2. For each message:
               - Skip non-string content (dict/list content)
               - Redact PII from string content
            3. Return new context with redacted messages (if changes made)
        """
        # No-op if engine not provided
        if not self.pii_engine:
            self.logger.debug("PII engine not configured, skipping")
            return context

        try:
            messages = context.get_effective_messages()
            redacted_messages: List[Dict[str, str]] = []
            total_redactions = 0
            messages_modified = 0

            # Redact each message
            for msg in messages:
                content = msg.get("content")

                # Only redact string content (not dict/list for multimodal)
                if isinstance(content, str):
                    result = self.pii_engine.redact(content)
                    redacted_content = result.redacted_text if hasattr(result, 'redacted_text') else result

                    # Count redactions from result
                    redaction_count = result.redaction_count if hasattr(result, 'redaction_count') else 0
                    if redaction_count > 0:
                        total_redactions += redaction_count
                        messages_modified += 1

                    # Create redacted message
                    redacted_msg = msg.copy()
                    redacted_msg["content"] = redacted_content
                    redacted_messages.append(redacted_msg)

                else:
                    # Non-string content (multimodal), pass through
                    redacted_messages.append(msg)

            # Add stage metadata
            metadata = {
                "action": "redacted" if total_redactions > 0 else "pass",
                "redactions_count": total_redactions,
                "messages_modified": messages_modified,
                "messages_total": len(messages),
            }

            new_context = context.with_stage_metadata(self.name, metadata)

            # Update messages if any redactions occurred
            if total_redactions > 0:
                new_context = new_context.with_modified_messages(redacted_messages)
                self.logger.info(
                    f"🔒 PII redaction: {total_redactions} instance(s) redacted "
                    f"across {messages_modified} message(s)"
                )
            else:
                self.logger.debug("✅ PII redaction: No PII detected")

            return new_context

        except Exception as e:
            # Fail open: Log error but don't halt pipeline
            self.logger.warning(f"⚠️ PII redaction error: {e}", exc_info=True)

            # Add error to context (non-fatal)
            error_context = context.with_error(f"PII redaction error: {str(e)}")

            # Add failure metadata
            error_context = error_context.with_stage_metadata(self.name, {
                "action": "error_fail_open",
                "error": str(e),
            })

            return error_context
