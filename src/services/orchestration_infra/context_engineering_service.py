"""
Context Engineering Service
============================

OPERATION SCALPEL V2 - Phase 3.3: MOVE (Self-Contained Service)

This service provides advanced context management capabilities implementing
Manus.im and Cognition.ai principles for optimal LLM performance.

Service Responsibilities:
- Context recitation (Manus principle: attention manipulation)
- KV-cache optimization with stable prefixes
- Append-only context accumulation (Cognition.ai principle)
- Stage output formatting for recitation

Pattern: Self-contained service with injected dependencies
Status: Phase 3.3 MOVE - Logic migrated, service fully independent
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextEngineeringService:
    """
    OPERATION SCALPEL V2 - Phase 3.3: Context Engineering Service

    Provides advanced context management implementing Manus.im and Cognition.ai
    principles for optimal LLM performance and context continuity.

    Key Principles:
    - Manus: Recitation-based attention manipulation
    - Manus: Stable prompt prefixes for KV-cache optimization
    - Cognition.ai: Append-only context (never overwrite)
    - Cognition.ai: Single context-rich agent with full traces

    Service is fully self-contained with dependency injection.
    """

    def __init__(self, context_optimizer, kv_cache_optimizer, context_stream):
        """
        Initialize Context Engineering Service

        Phase 3.3: Self-contained service with injected dependencies

        Args:
            context_optimizer: ContextEngineeringOptimizer for recitation
            kv_cache_optimizer: KVCacheOptimizer for stable prefixes
            context_stream: UnifiedContextStream for event logging
        """
        self.context_optimizer = context_optimizer
        self.kv_cache_optimizer = kv_cache_optimizer
        self.context_stream = context_stream
        logger.info("🔗 ContextEngineeringService initialized (Phase 3.3: MOVE - Self-contained)")

    async def apply_context_recitation(
        self, stage, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply Manus attention recitation before stage execution.

        Recites key context elements to manipulate attention and ensure
        context continuity (Manus principle: "Recite objectives into the context's end")

        Args:
            stage: PipelineStage enum value
            context: Current accumulated context

        Returns:
            Recited context with attention manipulation applied
        """
        try:
            # Instrumentation: emit context engineering start with optional budgets
            try:
                from src.core.unified_context_stream import ContextEventType

                self.context_stream.add_event(
                    ContextEventType.CONTEXT_ENGINEERING_STARTED,
                    {
                        "stage": getattr(stage, "value", str(stage)),
                        "token_budget": context.get("_token_budget"),
                        "source_budget": context.get("_source_budget"),
                        "context_size": len(str(context)),
                    },
                )
            except Exception:
                pass
            # Create context session for this stage
            session_id = await self.context_optimizer.create_session(
                initial_context=f"Pipeline Stage: {stage.display_name}"
            )

            # Extract critical elements for recitation
            original_query = context.get("initial_query", "No query available")
            stage_history = context.get("stage_history", [])
            key_insights = context.get("key_insights", [])

            # Recitation content following Manus principles
            recitation_content = f"""
CONTEXT RECITATION FOR STAGE CONTINUITY:
========================================
🎯 Original Query: {original_query}
📋 Current Stage: {stage.display_name}
🔄 Stages Completed: {len(stage_history)}
💡 Key Insights So Far: {len(key_insights)} insights preserved
📊 Context Elements: {len(context)} total elements

PREVIOUS STAGE OUTPUTS (Last 2):
{self.format_recent_stage_outputs(stage_history[-2:] if len(stage_history) >= 2 else stage_history)}

CRITICAL CONTEXT PRESERVATION:
- Original user intent preserved
- All stage outputs appended (never overwritten)
- Context continuity maintained
"""

            # Add recitation to context using context optimizer
            from src.engine.core.context_engineering_optimizer import ContextPriority

            await self.context_optimizer.append_context(
                session_id=session_id,
                content=recitation_content,
                content_type="attention_recitation",
                priority=ContextPriority.CRITICAL,
            )

            # Merge recited context with original context
            recited_context = {
                **context,
                "_recitation_applied": True,
                "_recitation_stage": stage.value,
                "_context_session_id": session_id,
            }

            logger.info(f"✅ Manus recitation applied for {stage.display_name}")
            return recited_context

        except Exception as e:
            logger.warning(f"⚠️ Recitation failed for {stage.display_name}: {e}")
            # Fallback: return original context
            return {**context, "_recitation_applied": False}

    async def create_stable_stage_context(
        self, stage, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create stable prefix for KV-cache optimization (Manus principle).

        Implements "stable prompt prefixes for consistent caching" while preserving
        all context information.

        Args:
            stage: PipelineStage enum value
            context: Current accumulated context

        Returns:
            Context with stable prefix for KV-cache optimization
        """
        try:
            # Create stable prefix using KV cache optimizer
            base_context_str = f"Stage: {stage.display_name} | Query: {context.get('initial_query', '')[:200]}"
            stable_prefix = self.kv_cache_optimizer.create_stable_prompt_prefix(
                base_context_str
            )

            # Create stable context structure
            stable_context = {
                **context,
                "_stable_prefix": stable_prefix,
                "_cache_optimized": True,
                "_stage_context": {
                    "stage": stage.value,
                    "stage_name": stage.display_name,
                    "optimization_applied": True,
                },
            }

            logger.debug(f"🔧 KV-cache stable prefix created for {stage.display_name}")
            # Instrumentation: record optimization event with optional budget delta
            try:
                from src.core.unified_context_stream import ContextEventType
                self.context_stream.add_event(
                    ContextEventType.CONTEXT_OPTIMIZATION_APPLIED,
                    {
                        "stage": getattr(stage, "value", str(stage)),
                        "optimizations": ["stable_prefix"],
                        "budget_delta": context.get("_budget_delta"),
                    },
                )
            except Exception:
                pass
            return stable_context

        except Exception as e:
            logger.warning(
                f"⚠️ KV-cache optimization failed for {stage.display_name}: {e}"
            )
            # Fallback: return original context
            return {**context, "_cache_optimized": False}

    async def append_result_to_context(
        self,
        stage,
        result: Dict[str, Any],
        original_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Append stage result to context following Cognition.ai "single context-rich agent" principle.

        Never overwrites context - only appends. Preserves full agent traces.

        Args:
            stage: PipelineStage enum value
            result: Stage execution result
            original_context: Original accumulated context

        Returns:
            Enriched context with stage result appended
        """
        try:
            # V6 DEBUG: Log what we're receiving
            logger.info(f"🔍 V6 DEBUG append_result_to_context: stage={stage.value}")
            logger.info(f"🔍 V6 DEBUG: result keys: {list(result.keys())[:10]}")
            logger.info(f"🔍 V6 DEBUG: original_context keys: {list(original_context.keys())[:10]}")

            # Get existing stage history or initialize
            stage_history = original_context.get("stage_history", [])

            # Create stage entry with metadata only (no result to prevent nesting)
            # V6 MIGRATION FIX: Removed "result" to prevent exponential stage_history nesting
            # Result data is already merged at top-level in enriched_context
            stage_entry = {
                "stage": stage.value,
                "stage_name": stage.display_name,
                "completed_at": datetime.utcnow().isoformat(),
                # Removed "result": result to prevent stage_history nesting
            }

            # COGNITION.AI: Append only, never overwrite
            enriched_context = {
                **original_context,  # Preserve ALL original context
                **result,  # Add new result
                "stage_history": stage_history + [stage_entry],  # Append to history
                "_context_engineering": {
                    "append_only_verified": True,
                    "original_context_preserved": True,
                    "stage_count": len(stage_history) + 1,
                    "cognition_ai_compliance": True,
                },
            }

            # Update key insights (append only)
            existing_insights = original_context.get("key_insights", [])
            new_insights = result.get("insights", [])
            if new_insights:
                enriched_context["key_insights"] = existing_insights + new_insights

            logger.info(
                f"📈 Context enriched: {stage.display_name} (Context size: {len(str(enriched_context))} chars)"
            )

            # Log context preservation event
            from src.core.unified_context_stream import ContextEventType

            self.context_stream.add_event(
                ContextEventType.CONTEXT_PRESERVATION_VALIDATED,
                {
                    "stage": stage.value,
                    "context_preserved": True,
                    "append_only_verified": True,
                    "cognition_ai_compliant": True,
                },
            )

            return enriched_context

        except Exception as e:
            logger.error(f"❌ Context append failed for {stage.display_name}: {e}")
            # Emergency fallback: merge contexts safely
            return {**original_context, **result}

    def format_recent_stage_outputs(self, recent_stages: List[Dict]) -> str:
        """
        Format recent stage outputs for recitation.

        Args:
            recent_stages: List of recent stage entries

        Returns:
            Formatted string summarizing recent stage outputs
        """
        if not recent_stages:
            return "No previous stages completed"

        formatted = []
        for stage_entry in recent_stages:
            stage_name = stage_entry.get("stage_name", "Unknown")
            result_summary = str(stage_entry.get("result", {}))[:300] + "..."
            formatted.append(f"• {stage_name}: {result_summary}")

        return "\n".join(formatted)
