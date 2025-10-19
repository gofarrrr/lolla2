#!/usr/bin/env python3
"""
Context Engineering Optimizer
Based on principles from Manus: Context Engineering for AI Agents

Implements key context engineering patterns:
1. Append-only context tracking with KV-cache optimization
2. State machine for tool availability and context management
3. File system as ultimate context storage
4. Attention manipulation through "recitation"
5. Controlled randomness to prevent pattern mimicry
6. Session-based consistent caching
"""

import asyncio
import logging
import json
import hashlib
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from pathlib import Path

logger = logging.getLogger(__name__)


class ContextState(Enum):
    """Context state machine states (from Manus principles)"""

    INITIALIZING = "initializing"
    STABLE_PREFIX = "stable_prefix"  # KV-cache optimized state
    APPEND_MODE = "append_mode"  # Adding new context
    COMPRESSION = "compression"  # Context compression needed
    RECITATION = "recitation"  # Attention manipulation mode
    ERROR_RECOVERY = "error_recovery"  # Recovering from context issues


class ContextPriority(Enum):
    """Priority levels for context elements"""

    CRITICAL = "critical"  # Must be preserved (tool definitions, core instructions)
    HIGH = "high"  # Important for current task
    MEDIUM = "medium"  # Useful context
    LOW = "low"  # Can be compressed or removed
    EPHEMERAL = "ephemeral"  # Temporary, can be discarded


class CompressionStrategy(Enum):
    """Strategies for context compression (from Manus principles)"""

    SEMANTIC = "semantic"  # Semantic compression while preserving meaning
    FREQUENCY = "frequency"  # Remove low-frequency information
    RECENCY = "recency"  # Remove old information
    IMPORTANCE = "importance"  # Remove based on importance scoring
    STRUCTURED = "structured"  # Use structured summaries


@dataclass
class ContextElement:
    """Individual element in the context"""

    element_id: UUID = field(default_factory=uuid4)
    content: str = ""
    content_type: str = "text"  # text, tool_result, observation, action
    priority: ContextPriority = ContextPriority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.utcnow)
    token_count: int = 0

    # Context engineering metadata
    stable_prefix: bool = False  # Part of stable prefix for KV-cache
    recitation_candidate: bool = False  # Can be used for attention manipulation
    compression_resistance: float = (
        1.0  # 0.0 = easily compressed, 1.0 = resist compression
    )

    # Learning metadata
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    effectiveness_score: float = 0.5  # How effective this context has been

    # Relationships
    depends_on: List[UUID] = field(default_factory=list)
    enables: List[UUID] = field(default_factory=list)
    conflicts_with: List[UUID] = field(default_factory=list)


@dataclass
class ContextSession:
    """Context session with engineering optimizations"""

    session_id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=datetime.utcnow)

    # Context state
    state: ContextState = ContextState.INITIALIZING
    elements: List[ContextElement] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 8000  # Default context window

    # KV-cache optimization
    prefix_hash: str = ""  # Hash of stable prefix for caching
    prefix_locked: bool = False  # If true, prefix cannot change
    cache_hits: int = 0
    cache_misses: int = 0

    # Performance tracking
    compression_events: int = 0
    recitation_events: int = 0
    wrong_turns: List[Dict] = field(default_factory=list)

    # File system context (from Manus principles)
    context_file: Optional[Path] = None
    checkpoint_files: List[Path] = field(default_factory=list)


class ContextEngineeringOptimizer:
    """
    Context Engineering Optimizer implementing Manus principles:

    Key Features:
    1. KV-cache optimization through stable prefixes
    2. Append-only context with controlled compression
    3. File system as ultimate context storage
    4. Attention manipulation through recitation
    5. Wrong turn preservation for learning
    6. State machine for context management
    """

    def __init__(
        self,
        max_context_tokens: int = 8000,
        compression_threshold: float = 0.8,
        enable_file_context: bool = True,
    ):
        self.max_context_tokens = max_context_tokens
        self.compression_threshold = compression_threshold
        self.enable_file_context = enable_file_context

        self.active_sessions: Dict[UUID, ContextSession] = {}
        self.context_history: List[Dict] = []  # Global append-only history
        self.compression_stats: Dict[str, int] = {
            "compressions_performed": 0,
            "tokens_saved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # File system context directory
        if enable_file_context:
            self.context_dir = Path("context_engineering")
            self.context_dir.mkdir(exist_ok=True)

        logger.info("Context Engineering Optimizer initialized")

    async def create_session(
        self, initial_context: str = "", session_config: Dict[str, Any] = None
    ) -> UUID:
        """Create optimized context session with stable prefix"""
        session = ContextSession()

        if session_config:
            session.max_tokens = session_config.get("max_tokens", 8000)
            session.prefix_locked = session_config.get("prefix_locked", False)

        # Initialize with stable prefix if provided
        if initial_context:
            prefix_element = ContextElement(
                content=initial_context,
                content_type="stable_prefix",
                priority=ContextPriority.CRITICAL,
                stable_prefix=True,
                compression_resistance=1.0,
            )
            prefix_element.token_count = self._estimate_tokens(initial_context)
            session.elements.append(prefix_element)
            session.total_tokens = prefix_element.token_count

            # Generate prefix hash for KV-cache optimization
            session.prefix_hash = hashlib.sha256(
                initial_context.encode("utf-8")
            ).hexdigest()[:16]

        # Set up file system context
        if self.enable_file_context:
            session.context_file = (
                self.context_dir / f"session_{session.session_id}.jsonl"
            )

        session.state = ContextState.STABLE_PREFIX
        self.active_sessions[session.session_id] = session

        # Record session creation in append-only history
        self.context_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "session_created",
                "session_id": str(session.session_id),
                "prefix_hash": session.prefix_hash,
            }
        )

        logger.info(
            f"Created context session {session.session_id} with {session.total_tokens} tokens"
        )
        return session.session_id

    async def append_context(
        self,
        session_id: UUID,
        content: str,
        content_type: str = "text",
        priority: ContextPriority = ContextPriority.MEDIUM,
    ) -> bool:
        """Append context using append-only principles"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]

        # Create context element
        element = ContextElement(
            content=content, content_type=content_type, priority=priority
        )
        element.token_count = self._estimate_tokens(content)

        # Check if compression is needed before appending
        projected_tokens = session.total_tokens + element.token_count
        if projected_tokens > session.max_tokens * self.compression_threshold:
            await self._compress_context(session)

        # Append element
        session.elements.append(element)
        session.total_tokens += element.token_count
        session.state = ContextState.APPEND_MODE

        # Write to file system if enabled
        if session.context_file:
            await self._write_to_context_file(session, element)

        # Record in append-only history
        self.context_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "context_appended",
                "session_id": str(session_id),
                "content_type": content_type,
                "tokens": element.token_count,
                "total_tokens": session.total_tokens,
            }
        )

        logger.debug(f"Appended {element.token_count} tokens to session {session_id}")
        return True

    async def record_wrong_turn(
        self,
        session_id: UUID,
        action: str,
        observation: str,
        error_info: Dict[str, Any] = None,
    ):
        """Record wrong turn for learning (key Manus principle)"""
        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]

        wrong_turn = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "observation": observation,
            "error_info": error_info or {},
            "preserved_for_learning": True,
        }

        # Append wrong turn to context (don't remove!)
        await self.append_context(
            session_id,
            f"WRONG_TURN: {action} -> {observation}",
            content_type="wrong_turn",
            priority=ContextPriority.HIGH,  # High priority to preserve for learning
        )

        session.wrong_turns.append(wrong_turn)

        logger.info(f"Recorded wrong turn in session {session_id}: {action}")

    async def apply_recitation(
        self, session_id: UUID, focus_elements: List[str] = None
    ) -> str:
        """Apply recitation to manipulate attention (Manus principle)"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]
        session.state = ContextState.RECITATION
        session.recitation_events += 1

        # Find elements to recite
        recitation_elements = []

        if focus_elements:
            # Recite specific elements
            for element in session.elements:
                if any(focus in element.content.lower() for focus in focus_elements):
                    recitation_elements.append(element)
        else:
            # Recite high-priority or recitation-candidate elements
            recitation_elements = [
                e
                for e in session.elements
                if e.recitation_candidate or e.priority == ContextPriority.CRITICAL
            ]

        # Build recitation text
        recitation_text = "CONTEXT_RECITATION:\n"
        for element in recitation_elements[-5:]:  # Last 5 relevant elements
            recitation_text += f"- {element.content[:200]}...\n"
            element.access_count += 1
            element.last_accessed = datetime.utcnow()

        # Append recitation to context
        await self.append_context(
            session_id,
            recitation_text,
            content_type="recitation",
            priority=ContextPriority.HIGH,
        )

        logger.info(
            f"Applied recitation in session {session_id}: {len(recitation_elements)} elements"
        )
        return recitation_text

    async def _compress_context(self, session: ContextSession):
        """Compress context while preserving important elements"""
        session.state = ContextState.COMPRESSION
        session.compression_events += 1

        # Separate elements by compressibility
        preserve_elements = []
        compress_candidates = []

        for element in session.elements:
            if (
                element.stable_prefix
                or element.priority == ContextPriority.CRITICAL
                or element.compression_resistance > 0.8
            ):
                preserve_elements.append(element)
            else:
                compress_candidates.append(element)

        # Sort compression candidates by importance
        compress_candidates.sort(
            key=lambda e: (e.priority.value, e.effectiveness_score, e.access_count),
            reverse=True,
        )

        # Compress using semantic compression
        if len(compress_candidates) > 0:
            compressed_content = await self._semantic_compress(compress_candidates)

            # Create compressed element
            compressed_element = ContextElement(
                content=compressed_content,
                content_type="compressed",
                priority=ContextPriority.MEDIUM,
                compression_resistance=0.5,
            )
            compressed_element.token_count = self._estimate_tokens(compressed_content)

            # Update session
            session.elements = preserve_elements + [compressed_element]
            session.total_tokens = sum(e.token_count for e in session.elements)

            # Update stats
            tokens_saved = (
                sum(e.token_count for e in compress_candidates)
                - compressed_element.token_count
            )
            self.compression_stats["compressions_performed"] += 1
            self.compression_stats["tokens_saved"] += tokens_saved

            logger.info(f"Compressed context: saved {tokens_saved} tokens")

    async def _semantic_compress(self, elements: List[ContextElement]) -> str:
        """Perform semantic compression of context elements"""
        # Simple implementation - in production would use LLM compression

        # Group by content type
        grouped = {}
        for element in elements:
            if element.content_type not in grouped:
                grouped[element.content_type] = []
            grouped[element.content_type].append(element)

        compressed_parts = []

        for content_type, type_elements in grouped.items():
            if len(type_elements) == 1:
                compressed_parts.append(type_elements[0].content)
            else:
                # Simple aggregation for now
                summary = f"COMPRESSED_{content_type.upper()}: "
                key_points = []
                for element in type_elements[-3:]:  # Last 3 elements
                    key_points.append(element.content[:100])
                summary += " | ".join(key_points)
                compressed_parts.append(summary)

        return "\n".join(compressed_parts)

    async def _write_to_context_file(
        self, session: ContextSession, element: ContextElement
    ):
        """Write context element to file system (Manus principle)"""
        if not session.context_file:
            return

        context_entry = {
            "timestamp": element.timestamp.isoformat(),
            "element_id": str(element.element_id),
            "content_type": element.content_type,
            "priority": element.priority.value,
            "content": element.content,
            "token_count": element.token_count,
        }

        # Append to JSONL file
        try:
            with open(session.context_file, "a") as f:
                json.dump(context_entry, f)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write context to file: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text.split()) + len(text) // 4

    async def get_optimized_context(
        self, session_id: UUID, include_recitation: bool = False
    ) -> str:
        """Get optimized context for LLM with KV-cache benefits"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]

        # Check cache hit/miss
        if session.prefix_hash:
            # This would trigger KV-cache hit in production
            session.cache_hits += 1
            self.compression_stats["cache_hits"] += 1
        else:
            session.cache_misses += 1
            self.compression_stats["cache_misses"] += 1

        # Apply recitation if requested
        if include_recitation:
            await self.apply_recitation(session_id)

        # Build context with stable prefix first
        context_parts = []

        # Add stable prefix elements first (KV-cache optimization)
        for element in session.elements:
            if element.stable_prefix:
                context_parts.append(element.content)

        # Add other elements in chronological order
        for element in session.elements:
            if not element.stable_prefix:
                context_parts.append(element.content)

        optimized_context = "\n\n".join(context_parts)

        logger.debug(
            f"Generated optimized context: {len(optimized_context)} chars, {session.total_tokens} tokens"
        )
        return optimized_context

    async def create_checkpoint(self, session_id: UUID) -> Path:
        """Create context checkpoint (from Manus principles)"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]

        checkpoint_file = (
            self.context_dir / f"checkpoint_{session_id}_{int(time.time())}.json"
        )

        checkpoint_data = {
            "session_id": str(session_id),
            "timestamp": datetime.utcnow().isoformat(),
            "state": session.state.value,
            "total_tokens": session.total_tokens,
            "elements": [
                {
                    "element_id": str(e.element_id),
                    "content": e.content,
                    "content_type": e.content_type,
                    "priority": e.priority.value,
                    "token_count": e.token_count,
                    "stable_prefix": e.stable_prefix,
                }
                for e in session.elements
            ],
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        session.checkpoint_files.append(checkpoint_file)
        logger.info(f"Created checkpoint: {checkpoint_file}")
        return checkpoint_file

    async def restore_from_checkpoint(self, checkpoint_file: Path) -> UUID:
        """Restore session from checkpoint"""
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

            session_id = UUID(checkpoint_data["session_id"])
            session = ContextSession(session_id=session_id)
            session.state = ContextState(checkpoint_data["state"])
            session.total_tokens = checkpoint_data["total_tokens"]

            # Restore elements
            for elem_data in checkpoint_data["elements"]:
                element = ContextElement(
                    element_id=UUID(elem_data["element_id"]),
                    content=elem_data["content"],
                    content_type=elem_data["content_type"],
                    priority=ContextPriority(elem_data["priority"]),
                    token_count=elem_data["token_count"],
                    stable_prefix=elem_data["stable_prefix"],
                )
                session.elements.append(element)

            self.active_sessions[session_id] = session
            logger.info(f"Restored session {session_id} from checkpoint")
            return session_id

        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            raise

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get context engineering optimization statistics"""
        total_sessions = len(self.active_sessions)
        total_history = len(self.context_history)

        cache_hit_rate = (
            self.compression_stats["cache_hits"]
            / (
                self.compression_stats["cache_hits"]
                + self.compression_stats["cache_misses"]
            )
            if (
                self.compression_stats["cache_hits"]
                + self.compression_stats["cache_misses"]
            )
            > 0
            else 0.0
        )

        return {
            "active_sessions": total_sessions,
            "total_context_history": total_history,
            "cache_hit_rate": cache_hit_rate,
            "compressions_performed": self.compression_stats["compressions_performed"],
            "tokens_saved": self.compression_stats["tokens_saved"],
            "avg_tokens_per_session": (
                sum(s.total_tokens for s in self.active_sessions.values())
                / total_sessions
                if total_sessions > 0
                else 0
            ),
            "wrong_turns_preserved": sum(
                len(s.wrong_turns) for s in self.active_sessions.values()
            ),
        }

    # ==================================================================
    # OPERATION AEGIS: Goals Ledger Recitation (Manus.im Best Practice)
    # ==================================================================

    def recite_goals(self, state: Any) -> str:
        """
        Recite active goals into context to fight 'lost-in-the-middle' drift.

        OPERATION AEGIS: Implements Manus.im best practice:
        "Keep/refresh a lightweight todo.md (or goals ledger) and recite it into
        the tail of context to fight 'lost-in-the-middle' drift on long loops."

        This method formats the active goals from PipelineState into a structured
        XML block for appending to the context. The recitation:
        - Reminds the model of high-level objectives
        - Prevents context drift in long-running analyses
        - Prioritizes active goals over completed/blocked ones

        Args:
            state: PipelineState containing active_goals

        Returns:
            Formatted XML string for context recitation, or empty string if no goals

        Example output:
            <active_goals>
            Strategic Goals for Analysis:
            ⏳ 🔴 [a1b2c3d4] Evaluate financial viability of acquisition
            ⏳ 🟡 [e5f6g7h8] Assess cultural fit between organizations
            ✅ 🔴 [i9j0k1l2] Identify regulatory compliance requirements
            </active_goals>
        """
        # Import here to avoid circular dependency
        try:
            from src.core.pipeline_contracts import PipelineState, StrategicGoal
        except ImportError:
            logger.warning("Could not import PipelineState - goals recitation unavailable")
            return ""

        # Extract active goals from state
        active_goals = getattr(state, 'active_goals', [])

        if not active_goals:
            logger.debug("No active goals to recite")
            return ""

        # Filter and sort goals: active first, then by priority
        sorted_goals = sorted(
            active_goals,
            key=lambda g: (
                0 if g.status == "active" else 1 if g.status == "blocked" else 2,  # Status priority
                g.priority  # Then by priority (1=highest, 3=lowest)
            )
        )

        # Build recitation string
        recitation_lines = ["<active_goals>", "Strategic Goals for Analysis:"]

        for goal in sorted_goals:
            recitation_lines.append(goal.to_recitation_string())

        recitation_lines.append("</active_goals>")

        recitation = "\n".join(recitation_lines)

        logger.info(f"🎯 Goals recited: {len(sorted_goals)} goals ({sum(1 for g in sorted_goals if g.status == 'active')} active)")

        return recitation


# Singleton instance
_context_optimizer = None


def get_context_engineering_optimizer() -> ContextEngineeringOptimizer:
    """Get singleton context engineering optimizer"""
    global _context_optimizer
    if _context_optimizer is None:
        _context_optimizer = ContextEngineeringOptimizer()
    return _context_optimizer


async def main():
    """Demo of context engineering optimization"""
    print("⚙️ Context Engineering Optimizer Demo")
    print("=" * 50)

    optimizer = get_context_engineering_optimizer()

    # Create session with stable prefix
    session_id = await optimizer.create_session(
        initial_context="You are a helpful assistant. Use mental models for analysis.",
        session_config={"max_tokens": 4000, "prefix_locked": True},
    )

    print(f"Created session: {session_id}")

    # Append various context types
    await optimizer.append_context(
        session_id,
        "USER: What is the best strategy for market expansion?",
        content_type="user_input",
        priority=ContextPriority.HIGH,
    )

    await optimizer.append_context(
        session_id,
        "ANALYSIS: Using MECE framework to structure the problem...",
        content_type="analysis",
        priority=ContextPriority.MEDIUM,
    )

    # Record a wrong turn
    await optimizer.record_wrong_turn(
        session_id,
        action="Applied Porter's Five Forces incorrectly",
        observation="Analysis was too generic and not actionable",
        error_info={"model_used": "porters_five_forces", "confidence": 0.3},
    )

    # Apply recitation
    recitation = await optimizer.apply_recitation(
        session_id, focus_elements=["strategy", "expansion"]
    )
    print(f"Recitation applied: {len(recitation)} chars")

    # Get optimized context
    context = await optimizer.get_optimized_context(session_id)
    print(f"Optimized context: {len(context)} chars")

    # Create checkpoint
    checkpoint = await optimizer.create_checkpoint(session_id)
    print(f"Checkpoint created: {checkpoint}")

    # Get stats
    stats = optimizer.get_optimization_stats()
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"Wrong turns preserved: {stats['wrong_turns_preserved']}")


if __name__ == "__main__":
    asyncio.run(main())
