"""
Legacy call_llm() Pre-Pipeline Logic (ARCHIVED)
===============================================

**Archive Date**: 2025-10-18
**Removed From**: src/integrations/llm/unified_client.py, lines 614-751
**Reason**: Replaced by pipeline architecture
**Status**: PRESERVED FOR EMERGENCY ROLLBACK

This file contains the legacy pre-LLM and provider-specific logic that was
removed from call_llm() after pipeline refactoring completion.

DO NOT IMPORT OR USE THIS CODE IN PRODUCTION.

This is an archive for reference and emergency rollback only.
"""

# ==============================================================================
# LEGACY PRE-LLM LOGIC (Lines 614-654)
# ==============================================================================
#
# This logic was executed when self.pipeline_enabled = False
# All functionality is now handled by pipeline stages:
#   - InjectionFirewallStage
#   - PIIRedactionStage
#   - OutputContractStage
#   - RAGContextInjectionStage
#   - SensitivityRoutingStage
#
# ==============================================================================

def legacy_pre_llm_processing(
    self,
    messages,
    provider,
    output_contract,
    sensitivity_override,
    kwargs
):
    """
    ARCHIVED: Pre-LLM processing logic (replaced by pipeline)

    Original location: unified_client.py lines 614-654
    """
    if not self.pipeline_enabled:
        # ENTERPRISE SECURITY (Feature 3/6): Injection Firewall
        if self.injection_firewall_enabled and self.injection_firewall:
            messages = self._check_injection_firewall(messages)

        # ENTERPRISE SECURITY (Phase 6): PII Redaction
        if self.pii_redaction_enabled and self.pii_engine:
            messages = self._redact_pii_from_messages(messages)

        # ENTERPRISE ROBUSTNESS (Feature 2/6): Append contract prompt if specified
        if self.output_contracts_enabled and output_contract:
            messages = self._append_contract_prompt(messages, output_contract)

        # Phase 2: Optional RAG context injection via Memory V2 (behind flag)
        try:
            import os as _os
            if _os.getenv("FF_RAG_DECAY_RETRIEVAL", "false").lower() in ("1", "true", "yes", "on"):
                user_msg = next((m for m in messages if m.get("role") == "user" and m.get("content")), None)
                if user_msg:
                    from src.engine.retrieval.context_injector import build_context_system_message
                    ctx = build_context_system_message(user_msg.get("content", ""), k=int(_os.getenv("RAG_K", "3")))
                    if ctx:
                        sys_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
                        if sys_idx is not None:
                            messages[sys_idx]["content"] = (messages[sys_idx].get("content", "") + "\n\n" + ctx).strip()
                        else:
                            messages = ([{"role": "system", "content": ctx}] + messages)
        except Exception:
            pass

        # ENTERPRISE SECURITY (Phase 6): Sensitivity Routing
        if self.sensitivity_routing_enabled and self.sensitivity_router:
            provider = self._apply_sensitivity_routing(
                messages, provider, sensitivity_override, kwargs
            )

        # Initialize call_kwargs for legacy path
        call_kwargs = kwargs.copy()

    return messages, provider, call_kwargs


# ==============================================================================
# LEGACY PROVIDER-SPECIFIC LOGIC (Lines 700-751)
# ==============================================================================
#
# This logic was executed when self.pipeline_enabled = False
# All functionality is now handled by pipeline stages:
#   - ProviderAdapterStage (parameter filtering)
#   - ReasoningModeStage (OpenRouter reasoning mode)
#
# ==============================================================================

def legacy_provider_specific_logic(
    self,
    provider_key,
    functions,
    response_format,
    call_kwargs
):
    """
    ARCHIVED: Provider-specific parameter filtering (replaced by pipeline)

    Original location: unified_client.py lines 700-751
    """
    if not self.pipeline_enabled:
        # Provider-specific parameter filtering
        if provider_key == "deepseek":
            if functions is not None:
                call_kwargs["functions"] = functions
            if response_format is not None:
                call_kwargs["response_format"] = response_format

        elif provider_key == "anthropic":
            self.logger.info(
                "🎭 Claude provider: functions and response_format parameters excluded"
            )

        elif provider_key == "openai":
            if functions is not None:
                call_kwargs["functions"] = functions
            if response_format is not None:
                call_kwargs["response_format"] = response_format

        elif provider_key == "openrouter":
            self.logger.info(
                "🚀 OpenRouter provider: Using Grok-4-Fast for strategic analysis"
            )

            if response_format is not None:
                call_kwargs["response_format"] = response_format

            # Reasoning mode selection
            try:
                from src.services.reasoning_mode_selector import get_reasoning_mode_selector
                selector = get_reasoning_mode_selector()
                task_type = call_kwargs.get("task_type", "general")
                prompt_length = len(messages[0].get("content", "")) if messages else 0
                requires_multi_step = call_kwargs.get("requires_multi_step", False)
                stakeholder_impact = call_kwargs.get("stakeholder_impact", "medium")
                explicit_override = call_kwargs.get("reasoning_enabled_override")

                if explicit_override is not None:
                    reasoning_enabled = explicit_override
                else:
                    reasoning_enabled = selector.should_enable_reasoning(
                        task_type=task_type,
                        prompt_length=prompt_length,
                        requires_multi_step=requires_multi_step,
                        stakeholder_impact=stakeholder_impact
                    )

                call_kwargs["reasoning_enabled"] = reasoning_enabled
            except Exception as e:
                self.logger.warning(f"⚠️ Reasoning mode selector failed: {e}")
                call_kwargs["reasoning_enabled"] = True

    return call_kwargs


# ==============================================================================
# RESURRECTION INSTRUCTIONS
# ==============================================================================
#
# IF YOU NEED TO RESTORE THIS CODE:
#
# 1. Set self.pipeline_enabled = False in unified_client.py __init__
# 2. Copy the code above back into call_llm() method:
#    - Pre-LLM logic: Insert at line ~614 (before provider normalization)
#    - Provider logic: Insert at line ~700 (before provider.call_llm())
# 3. Ensure indentation matches (inside call_llm method)
# 4. Verify all helper methods still exist (_check_injection_firewall, etc.)
# 5. Run tests: pytest tests/integrations/llm/ -v
# 6. Deploy with extensive monitoring
# 7. FILE BUG REPORT explaining why pipeline failed
#
# ==============================================================================

# ==============================================================================
# MIGRATION NOTES
# ==============================================================================
#
# This code was removed as part of Operation Lean (2025-10-18)
#
# BEFORE (unified_client.py):
#   - 467 lines in call_llm()
#   - CC = 81 (cyclomatic complexity)
#   - Duplicate logic: pipeline + legacy fallback
#
# AFTER (unified_client.py):
#   - ~160 lines in call_llm() (65% reduction)
#   - CC = ~5 (pipeline delegation)
#   - Single source of truth (pipeline only)
#
# BENEFITS:
#   - Easier to test (test pipeline stages independently)
#   - Easier to extend (add new pipeline stages)
#   - Easier to understand (clean separation of concerns)
#   - No more confusion about which code path to update
#
# RISKS:
#   - Pipeline bugs could break all LLM calls
#   - This is why we keep legacy code archived for 2 sprints
#
# ==============================================================================
