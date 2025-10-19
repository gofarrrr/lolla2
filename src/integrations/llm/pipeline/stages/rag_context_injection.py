"""
RAG Context Injection Pipeline Stage

Retrieves and injects relevant context from Memory V2 into system message.

Extracted from: unified_client.py::call_llm() RAG injection (lines 498-516)
Design: Single Responsibility - Only handles RAG context injection
Complexity Target: CC < 5
"""

from typing import List, Dict, Optional, Any, Callable
import logging

from ..context import LLMCallContext
from ..stage import PipelineStage


class RAGContextInjectionStage(PipelineStage):
    """
    Pipeline stage that injects RAG context from Memory V2 into system message.

    This stage:
    1. Extracts first user message as query
    2. Calls context builder to retrieve relevant context (typically from vector DB)
    3. Injects context into system message (prepends or merges)
    4. Returns context with modified messages

    Behavior (extracted verbatim from unified_client.py):
    - Feature flag controlled (FF_RAG_DECAY_RETRIEVAL)
    - Uses first user message as retrieval query
    - Calls build_context_system_message(query, k=RAG_K)
    - Merges with existing system message or inserts new one
    - Fail-silent: On errors, passes through unchanged

    Dependencies:
        - Context builder function (typically build_context_system_message from retrieval)
        - RAG_K parameter (number of chunks to retrieve, default: 3)

    Attributes:
        build_context_func: Function to build RAG context from query
        rag_k: Number of chunks to retrieve (default: 3)
        enabled: Whether stage is enabled (inherited from PipelineStage)

    Example:
        ```python
        from src.engine.retrieval.context_injector import build_context_system_message

        stage = RAGContextInjectionStage(
            build_context_func=build_context_system_message,
            rag_k=5
        )

        context = LLMCallContext(
            messages=[{"role": "user", "content": "What is our return policy?"}],
            model="deepseek-chat",
            provider="deepseek",
            kwargs={}
        )

        new_context = await stage.execute(context)
        # new_context has system message with relevant retrieved context
        ```

    References:
        - Original: unified_client.py lines 498-516
        - ADR: docs/adr/ADR-001-unified-client-pipeline-refactoring.md
    """

    def __init__(
        self,
        build_context_func: Optional[Callable[[str, int], Optional[str]]] = None,
        rag_k: int = 3,
        enabled: bool = True
    ):
        """
        Initialize RAG context injection stage.

        Args:
            build_context_func: Function that takes (query, k) and returns context string
                               (or None if no context found)
            rag_k: Number of chunks to retrieve (default: 3)
            enabled: Whether stage is enabled

        Note:
            If build_context_func is None, stage becomes a no-op.
        """
        super().__init__(enabled=enabled)
        self.build_context_func = build_context_func
        self.rag_k = rag_k

    @property
    def name(self) -> str:
        """Stage name for logging and metadata."""
        return "RAGContextInjection"

    async def execute(self, context: LLMCallContext) -> LLMCallContext:
        """
        Inject RAG context into system message.

        This implementation is extracted verbatim from unified_client.py
        with minimal changes to adapt to pipeline context pattern.

        Args:
            context: Input context with messages

        Returns:
            New context with modified messages (if RAG context retrieved)

        Workflow:
            1. Get first user message as query
            2. Call context builder to retrieve relevant context
            3. If context found, inject into system message
            4. Return new context with modified messages
        """
        # No-op if function not provided
        if not self.build_context_func:
            self.logger.debug("RAG context builder not configured, skipping")
            return context

        try:
            messages = context.get_effective_messages()

            # Extract first user message as query
            user_msg = next(
                (m for m in messages if m.get("role") == "user" and m.get("content")),
                None
            )

            if not user_msg:
                self.logger.debug("No user message found for RAG query, skipping")
                return context.with_stage_metadata(self.name, {
                    "action": "skip",
                    "reason": "no_user_message",
                })

            query = user_msg.get("content", "")
            if not query:
                self.logger.debug("Empty user message, skipping RAG retrieval")
                return context.with_stage_metadata(self.name, {
                    "action": "skip",
                    "reason": "empty_query",
                })

            # Build RAG context
            self.logger.debug(f"Retrieving RAG context for query (k={self.rag_k})")
            rag_context = self.build_context_func(query, k=self.rag_k)

            if not rag_context:
                self.logger.debug("No RAG context retrieved, skipping injection")
                return context.with_stage_metadata(self.name, {
                    "action": "skip",
                    "reason": "no_context_found",
                    "query_length": len(query),
                })

            # Inject context into system message
            modified_messages = [msg.copy() for msg in messages]  # Deep copy

            # Find system message or create one
            sys_idx = next(
                (i for i, m in enumerate(modified_messages) if m.get("role") == "system"),
                None
            )

            if sys_idx is not None:
                # Merge with existing system message
                existing_content = modified_messages[sys_idx].get("content", "")
                modified_messages[sys_idx]["content"] = (
                    existing_content + "\n\n" + rag_context
                ).strip()
                self.logger.info(
                    f"✅ RAG context merged with existing system message "
                    f"({len(rag_context)} chars)"
                )
            else:
                # Insert new system message at beginning
                modified_messages = [
                    {"role": "system", "content": rag_context}
                ] + modified_messages
                self.logger.info(
                    f"✅ RAG context inserted as new system message "
                    f"({len(rag_context)} chars)"
                )

            # Add stage metadata
            metadata = {
                "action": "injected",
                "query_length": len(query),
                "context_length": len(rag_context),
                "rag_k": self.rag_k,
                "system_message_existed": sys_idx is not None,
            }

            new_context = context.with_stage_metadata(self.name, metadata)

            # Update messages
            new_context = new_context.with_modified_messages(modified_messages)

            return new_context

        except Exception as e:
            # Fail silent: Log error but don't halt pipeline (matches original behavior)
            self.logger.warning(f"⚠️ RAG context injection error: {e}", exc_info=True)

            # Add error to context (non-fatal)
            error_context = context.with_error(f"RAG context injection error: {str(e)}")

            # Add failure metadata
            error_context = error_context.with_stage_metadata(self.name, {
                "action": "error_fail_silent",
                "error": str(e),
            })

            return error_context
