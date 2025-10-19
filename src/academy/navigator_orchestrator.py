"""Navigator Orchestrator — stateful conversational controller for the Academy experience."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.integrations.llm.unified_client import UnifiedLLMClient
from src.services.knowledge_retrieval_service import KnowledgeRetrievalService

from src.academy.navigator import (
    HandlerResult,
    NavigatorRuntime,
    NavigatorSession,
    NavigatorState,
    build_state_handlers,
)

logger = logging.getLogger(__name__)


class NavigatorOrchestrator:
    """State machine driving the Mental Model Navigator conversation."""

    def __init__(
        self,
        session_id: str,
        knowledge_service: KnowledgeRetrievalService,
        llm_client: Optional[UnifiedLLMClient] = None,
    ) -> None:
        self.session = NavigatorSession(session_id=session_id)
        self.knowledge_service = knowledge_service
        self.llm_client = llm_client or UnifiedLLMClient()

        self.state_transitions: Dict[NavigatorState, NavigatorState] = {
            NavigatorState.INITIAL: NavigatorState.CLARIFYING,
            NavigatorState.CLARIFYING: NavigatorState.CONTEXT_GATHERING,
            NavigatorState.CONTEXT_GATHERING: NavigatorState.MODEL_DISCOVERY,
            NavigatorState.MODEL_DISCOVERY: NavigatorState.MODEL_SELECTION,
            NavigatorState.MODEL_SELECTION: NavigatorState.MODEL_EXPLANATION,
            NavigatorState.MODEL_EXPLANATION: NavigatorState.APPLICATION_DESIGN,
            NavigatorState.APPLICATION_DESIGN: NavigatorState.IMPLEMENTATION_GUIDANCE,
            NavigatorState.IMPLEMENTATION_GUIDANCE: NavigatorState.VALIDATION_FRAMEWORK,
            NavigatorState.VALIDATION_FRAMEWORK: NavigatorState.NEXT_STEPS,
            NavigatorState.NEXT_STEPS: NavigatorState.COMPLETED,
            NavigatorState.COMPLETED: NavigatorState.COMPLETED,
        }

        self.runtime = NavigatorRuntime(
            session=self.session,
            knowledge_service=self.knowledge_service,
            llm_client=self.llm_client,
            state_transitions=self.state_transitions,
        )
        self.handlers = build_state_handlers(self.runtime)

        self._dev_validation_issues = set() if os.getenv("ENV", "production") != "production" else None

        logger.info("🎓 NavigatorOrchestrator initialized - Session: %s", session_id)

    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a user message through the state machine."""

        context = context or {}
        try:
            self.session.last_activity = datetime.now()
            self._record_user_message(message)

            handler = self.handlers.get(self.session.state)
            if handler is None:
                raise ValueError(f"No handler registered for state: {self.session.state}")

            result = await handler(message, context)
            self._record_assistant_message(result)
            if os.getenv("ENV", "production") != "production":
                self._dev_validate_handler_result(result)
            return result.to_dict()

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("❌ Error processing navigator message: %s", exc)
            return {
                "response": "I encountered an error processing your message. Please try again.",
                "state": self.session.state.value,
                "error": str(exc),
            }

    def _record_user_message(self, message: str) -> None:
        self.session.conversation_history.append(
            {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat(),
                "state": self.session.state.value,
            }
        )

    def _record_assistant_message(self, result: HandlerResult) -> None:
        self.session.conversation_history.append(
            {
                "role": "assistant",
                "content": result.response,
                "timestamp": datetime.now().isoformat(),
                "state": self.session.state.value,
                "metadata": result.metadata,
            }
        )

    def get_session_status(self) -> Dict[str, Any]:
        """Expose the runtime session status."""

        return self.runtime.get_session_status()

    def get_session_summary(self) -> Dict[str, Any]:
        """Alias for session status retained for backward compatibility."""

        return self.runtime.get_session_status()

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Retrieve the full conversation history."""

        return list(self.session.conversation_history)

    async def reset_session(self) -> None:
        """Reset the navigator session to the initial state."""

        await self.runtime.reset_session()
        self.handlers = build_state_handlers(self.runtime)
        logger.info("🔄 Navigator session reset - Session: %s", self.session.session_id)

    def _dev_validate_handler_result(self, result: HandlerResult) -> None:
        """Emit validation warnings in non-production environments."""

        issues = []
        if result.state != self.session.state.value:
            issues.append(
                f"state-mismatch(handler={result.state}, session={self.session.state.value})"
            )
        if not isinstance(result.metadata, dict):
            issues.append("metadata-not-dict")
        if not isinstance(result.suggested_actions, list):
            issues.append("suggested_actions-not-list")
        elif any(not isinstance(action, str) for action in result.suggested_actions):
            issues.append("suggested_actions-non-string")

        if self.session.conversation_history:
            last_entry = self.session.conversation_history[-1]
            if last_entry.get("role") == "assistant" and last_entry.get("content") != result.response:
                issues.append("history-response-mismatch")

        if not issues or self._dev_validation_issues is None:
            return

        new_issues = [issue for issue in issues if issue not in self._dev_validation_issues]
        if new_issues:
            self._dev_validation_issues.update(new_issues)
            logger.warning("Navigator dev validation issues: %s", ", ".join(new_issues))
