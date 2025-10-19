"""
Output Contract Pipeline Stage

Injects output contract prompt into system message to enforce structured output.

Extracted from: unified_client.py::_append_contract_prompt() (lines 1300-1353)
Design: Single Responsibility - Only handles contract prompt injection
Complexity Target: CC < 5
"""

from typing import List, Dict, Optional, Any, Callable
import logging

from ..context import LLMCallContext
from ..stage import PipelineStage


class OutputContractStage(PipelineStage):
    """
    Pipeline stage that injects output contract prompts into system message.

    This stage:
    1. Gets contract prompt for specified contract type (e.g., "analysis", "structured_query")
    2. Finds existing system message or creates new one
    3. Appends contract prompt to system message
    4. Returns context with modified messages

    Behavior (extracted verbatim from unified_client.py):
    - Looks up contract prompt by name
    - If no contract specified, passes through unchanged
    - If contract not found, logs warning and passes through
    - Appends to existing system message or inserts new one at beginning
    - Fail-open: On errors, returns original messages

    Dependencies:
        - Contract prompt getter function (typically from output_contracts module)

    Attributes:
        get_contract_prompt: Function to retrieve contract prompt by name
        enabled: Whether stage is enabled (inherited from PipelineStage)

    Example:
        ```python
        from src.engine.contracts.output_contracts import get_contract_prompt

        stage = OutputContractStage(get_contract_prompt_func=get_contract_prompt)

        context = LLMCallContext(
            messages=[{"role": "user", "content": "Analyze this data"}],
            model="deepseek-chat",
            provider="deepseek",
            kwargs={"output_contract": "analysis"}
        )

        new_context = await stage.execute(context)
        # new_context has system message with contract prompt
        ```

    References:
        - Original: unified_client.py lines 495-496, 1300-1353
        - ADR: docs/adr/ADR-001-unified-client-pipeline-refactoring.md
    """

    def __init__(
        self,
        get_contract_prompt_func: Optional[Callable[[str], Optional[str]]] = None,
        enabled: bool = True
    ):
        """
        Initialize output contract stage.

        Args:
            get_contract_prompt_func: Function that takes contract_name and returns prompt string
                                      (or None if disabled/not found)
            enabled: Whether stage is enabled

        Note:
            If get_contract_prompt_func is None, stage becomes a no-op.
        """
        super().__init__(enabled=enabled)
        self.get_contract_prompt_func = get_contract_prompt_func

    @property
    def name(self) -> str:
        """Stage name for logging and metadata."""
        return "OutputContract"

    async def execute(self, context: LLMCallContext) -> LLMCallContext:
        """
        Inject contract prompt into system message.

        This implementation is extracted verbatim from unified_client.py
        with minimal changes to adapt to pipeline context pattern.

        Args:
            context: Input context with messages and output_contract in kwargs

        Returns:
            New context with modified messages (if contract specified)

        Workflow:
            1. Check if output_contract specified in kwargs
            2. Get contract prompt for contract name
            3. Find system message or create new one
            4. Append contract prompt to system message
            5. Return new context with modified messages
        """
        # No-op if function not provided
        if not self.get_contract_prompt_func:
            self.logger.debug("Contract prompt function not configured, skipping")
            return context

        # Check if output_contract specified
        output_contract = context.kwargs.get("output_contract")
        if not output_contract:
            self.logger.debug("No output contract specified, skipping")
            return context

        try:
            # Get contract prompt
            contract_prompt = self.get_contract_prompt_func(output_contract)
            if not contract_prompt:
                self.logger.warning(
                    f"⚠️ Contract '{output_contract}' not found, skipping"
                )

                # Add metadata (contract not found)
                return context.with_stage_metadata(self.name, {
                    "action": "skip",
                    "reason": "contract_not_found",
                    "contract_name": output_contract,
                })

            # Get effective messages
            messages = context.get_effective_messages()
            modified_messages = [msg.copy() for msg in messages]  # Deep copy

            # Find system message or create one
            system_msg_idx = None
            for idx, msg in enumerate(modified_messages):
                if msg.get("role") == "system":
                    system_msg_idx = idx
                    break

            if system_msg_idx is not None:
                # Append to existing system message
                modified_messages[system_msg_idx]["content"] += (
                    "\n\n" + contract_prompt
                )
                self.logger.info(
                    f"✅ Contract prompt appended to existing system message: {output_contract}"
                )
            else:
                # Insert new system message at beginning
                modified_messages.insert(
                    0, {"role": "system", "content": contract_prompt}
                )
                self.logger.info(
                    f"✅ Contract prompt inserted as new system message: {output_contract}"
                )

            # Add stage metadata
            metadata = {
                "action": "injected",
                "contract_name": output_contract,
                "system_message_existed": system_msg_idx is not None,
                "prompt_length": len(contract_prompt),
            }

            new_context = context.with_stage_metadata(self.name, metadata)

            # Update messages
            new_context = new_context.with_modified_messages(modified_messages)

            return new_context

        except Exception as e:
            # Fail open: Log error but don't halt pipeline
            self.logger.warning(f"⚠️ Failed to append contract prompt: {e}", exc_info=True)

            # Add error to context (non-fatal)
            error_context = context.with_error(f"Output contract error: {str(e)}")

            # Add failure metadata
            error_context = error_context.with_stage_metadata(self.name, {
                "action": "error_fail_open",
                "error": str(e),
                "contract_name": output_contract,
            })

            return error_context
