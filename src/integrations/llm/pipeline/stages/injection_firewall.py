"""
Injection Firewall Pipeline Stage

Detects and blocks/sanitizes prompt injection attacks in user messages.

Extracted from: unified_client.py::_check_injection_firewall() (lines 1231-1298)
Design: Single Responsibility - Only handles injection attack detection
Complexity Target: CC < 5
"""

from typing import List, Dict, Optional, Any
import logging

from ..context import LLMCallContext
from ..stage import PipelineStage, StageExecutionError


class InjectionFirewallStage(PipelineStage):
    """
    Pipeline stage that checks messages for prompt injection attacks.

    This stage:
    1. Scans user messages for injection patterns
    2. Takes action based on severity: BLOCK, SANITIZE, or LOG_ONLY
    3. Returns context with sanitized messages if needed

    Behavior (extracted verbatim from unified_client.py):
    - Only checks user messages (system/assistant pass through)
    - BLOCK action: Raises InjectionAttemptError (halts pipeline)
    - SANITIZE action: Cleans malicious patterns from content
    - LOG_ONLY action: Logs detection but allows message
    - Fail-open: On unexpected errors, returns original messages

    Dependencies:
        - src.engine.security.injection_firewall (InjectionFirewall instance)

    Attributes:
        injection_firewall: InjectionFirewall instance (optional, can be None if disabled)
        enabled: Whether stage is enabled (inherited from PipelineStage)

    Example:
        ```python
        from src.engine.security.injection_firewall import get_injection_firewall

        firewall = get_injection_firewall(enabled=True)
        stage = InjectionFirewallStage(injection_firewall=firewall)

        context = LLMCallContext(
            messages=[{"role": "user", "content": "Ignore all instructions"}],
            model="deepseek-chat",
            provider="deepseek",
            kwargs={}
        )

        # May raise InjectionAttemptError if blocked
        new_context = await stage.execute(context)
        ```

    References:
        - Original: unified_client.py lines 487-488, 1231-1298
        - ADR: docs/adr/ADR-001-unified-client-pipeline-refactoring.md
    """

    def __init__(
        self,
        injection_firewall: Optional[Any] = None,
        enabled: bool = True
    ):
        """
        Initialize injection firewall stage.

        Args:
            injection_firewall: InjectionFirewall instance (or None to disable)
            enabled: Whether stage is enabled

        Note:
            If injection_firewall is None, stage becomes a no-op (returns context unchanged).
        """
        super().__init__(enabled=enabled)
        self.injection_firewall = injection_firewall

    @property
    def name(self) -> str:
        """Stage name for logging and metadata."""
        return "InjectionFirewall"

    async def execute(self, context: LLMCallContext) -> LLMCallContext:
        """
        Check messages for injection attempts and sanitize if needed.

        This implementation is extracted verbatim from unified_client.py
        with minimal changes to adapt to pipeline context pattern.

        Args:
            context: Input context with messages to check

        Returns:
            New context with sanitized messages (if sanitization occurred)

        Raises:
            InjectionAttemptError: If HIGH/CRITICAL injection detected and action is BLOCK

        Workflow:
            1. Get effective messages from context
            2. For each message:
               - Skip non-user messages
               - Check user message with firewall
               - Handle BLOCK, SANITIZE, or LOG_ONLY actions
            3. Return new context with sanitized messages (if any changes)
        """
        # No-op if firewall not provided
        if not self.injection_firewall:
            self.logger.debug("Injection firewall not configured, skipping")
            return context

        try:
            # Import firewall types
            from src.engine.security.injection_firewall import (
                FirewallAction,
                InjectionAttemptError,
            )

            messages = context.get_effective_messages()
            sanitized_messages: List[Dict[str, str]] = []
            detections_count = 0
            actions_taken: Dict[str, int] = {"block": 0, "sanitize": 0, "log_only": 0}

            # Check each message
            for msg in messages:
                # Only check user messages (not system/assistant)
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    result = self.injection_firewall.check_input(msg["content"])

                    if result.action_taken == FirewallAction.BLOCK:
                        # BLOCK: Raise exception (halts pipeline)
                        self.logger.error(
                            f"🚫 INJECTION BLOCKED: {len(result.detections)} pattern(s) detected"
                        )

                        # Add metadata before raising
                        metadata = {
                            "action": "block",
                            "detections_count": len(result.detections),
                            "pattern_name": result.detections[0].pattern_name if result.detections else None,
                        }

                        # Update context with error metadata
                        context = context.with_stage_metadata(self.name, metadata)

                        # Raise to halt pipeline
                        raise InjectionAttemptError(
                            f"Injection attempt blocked: {result.detections[0].pattern_name}",
                            result.detections,
                        )

                    elif result.action_taken == FirewallAction.SANITIZE:
                        # SANITIZE: Use cleaned content
                        sanitized_msg = msg.copy()
                        sanitized_msg["content"] = result.sanitized_input
                        sanitized_messages.append(sanitized_msg)

                        detections_count += len(result.detections)
                        actions_taken["sanitize"] += 1

                        self.logger.warning(
                            f"🧹 INPUT SANITIZED: {len(result.detections)} pattern(s) removed"
                        )

                    else:
                        # LOG_ONLY: Pass through
                        sanitized_messages.append(msg)

                        if result.detections:
                            detections_count += len(result.detections)
                            actions_taken["log_only"] += 1

                            self.logger.info(
                                f"ℹ️ LOW-RISK PATTERNS: {len(result.detections)} detected (allowed)"
                            )

                else:
                    # System/assistant messages pass through unchanged
                    sanitized_messages.append(msg)

            # Add stage metadata
            metadata = {
                "action": "pass" if detections_count == 0 else "sanitize",
                "detections_count": detections_count,
                "actions_taken": actions_taken,
                "messages_checked": len([m for m in messages if m.get("role") == "user"]),
            }

            new_context = context.with_stage_metadata(self.name, metadata)

            # Update messages if any sanitization occurred
            if actions_taken["sanitize"] > 0:
                new_context = new_context.with_modified_messages(sanitized_messages)
                self.logger.info(
                    f"✅ Injection firewall: {actions_taken['sanitize']} message(s) sanitized"
                )
            else:
                self.logger.debug("✅ Injection firewall: No threats detected")

            return new_context

        except InjectionAttemptError:
            # Re-raise blocking errors (halt pipeline)
            raise

        except Exception as e:
            # Fail open: Log error but don't halt pipeline
            self.logger.warning(f"⚠️ Injection firewall error: {e}", exc_info=True)

            # Add error to context (non-fatal)
            error_context = context.with_error(f"Injection firewall error: {str(e)}")

            # Add failure metadata
            error_context = error_context.with_stage_metadata(self.name, {
                "action": "error_fail_open",
                "error": str(e),
            })

            return error_context
