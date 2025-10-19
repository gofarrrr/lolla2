"""
Incremental Context Manager Stub Implementation
Lightweight stub for ToolDecisionFramework compatibility
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from src.core.unified_context_stream import (
    UnifiedContextStream,
    ContextEventType,
)

logger = logging.getLogger(__name__)


@dataclass
class ContextUpdate:
    """Simple context update representation"""

    update_id: str
    update_type: str
    content: Dict[str, Any]
    timestamp: datetime
    tokens_estimated: int = 0


class IncrementalContextManager:
    """
    Stub implementation of IncrementalContextManager for ToolDecisionFramework compatibility

    This is a simplified version that provides the interface needed by ToolDecisionFramework
    without the full complexity of the V4 implementation.
    """

    def __init__(self, context_stream: UnifiedContextStream):
        self.context_stream = context_stream
        self.updates_log: List[ContextUpdate] = []
        self.cache: Dict[str, Any] = {}

        logger.info("🔧 IncrementalContextManager stub initialized")

    def add_update(self, update_type: str, content: Dict[str, Any]) -> ContextUpdate:
        """Add a context update"""
        update = ContextUpdate(
            update_id=f"update_{len(self.updates_log)}",
            update_type=update_type,
            content=content,
            timestamp=datetime.utcnow(),
        )
        self.updates_log.append(update)

        # Add to context stream as well
        self.context_stream.add_event(
            ContextEventType.REASONING_STEP,
            {
                "update_id": update.update_id,
                "update_type": update_type,
                "content": content,
            },
        )

        return update

    def get_current_context(self) -> Dict[str, Any]:
        """Get current context state"""
        return {
            "updates_count": len(self.updates_log),
            "recent_updates": [u.update_type for u in self.updates_log[-5:]],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_incremental_updates(self, since: datetime = None) -> List[ContextUpdate]:
        """Get updates since a specific timestamp"""
        if since is None:
            return self.updates_log
        return [u for u in self.updates_log if u.timestamp > since]

    def get_context_summary(self, max_tokens: int = 1000) -> str:
        """Get a summary of the current context"""
        return f"Context with {len(self.updates_log)} updates, latest: {datetime.utcnow().isoformat()}"

    def compress_old_context(self, older_than: timedelta = None) -> int:
        """Compress old context entries (stub implementation)"""
        if older_than is None:
            older_than = timedelta(hours=1)

        cutoff = datetime.utcnow() - older_than
        compressed_count = len([u for u in self.updates_log if u.timestamp < cutoff])

        logger.debug(f"🗜️ Compressed {compressed_count} old context entries")
        return compressed_count
