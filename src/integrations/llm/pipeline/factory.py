"""
Pipeline Factory

Creates configured LLM call pipelines with all stages wired together.

This is the integration point for unified_client.py refactoring.
"""

from typing import Optional, List
import logging

from .pipeline import LLMCallPipeline
from .stages import (
    InjectionFirewallStage,
    PIIRedactionStage,
    SensitivityRoutingStage,
    OutputContractStage,
    RAGContextInjectionStage,
    ProviderAdapterStage,
    ReasoningModeStage,
    StyleGateStage,
    ConfidenceEscalationStage,
)


logger = logging.getLogger(__name__)


def create_llm_pipeline(
    # Stage dependencies (from unified_client)
    injection_firewall=None,
    pii_engine=None,
    sensitivity_router=None,
    get_contract_prompt_func=None,
    build_rag_context_func=None,
    reasoning_selector=None,
    style_scorer=None,
    style_gate=None,
    escalate_func=None,
    # Configuration
    available_providers: Optional[List[str]] = None,
    rag_k: int = 3,
    confidence_threshold: float = 0.85,
    # Enable/disable stages
    enable_injection_firewall: bool = True,
    enable_pii_redaction: bool = True,
    enable_sensitivity_routing: bool = True,
    enable_output_contracts: bool = True,
    enable_rag_context: bool = True,
    enable_provider_adapter: bool = True,
    enable_reasoning_mode: bool = True,
    enable_style_gate: bool = True,
    enable_confidence_escalation: bool = True,
) -> LLMCallPipeline:
    """
    Create a fully configured LLM call pipeline.

    This factory wires together all 9 pipeline stages in the correct order.

    Args:
        Stage dependencies from unified_client (typically from get_* functions)
        Configuration parameters (rag_k, thresholds, etc.)
        Enable/disable flags per stage

    Returns:
        LLMCallPipeline with all stages configured

    Usage:
        ```python
        # In unified_client.py __init__
        self.pipeline = create_llm_pipeline(
            injection_firewall=self.injection_firewall,
            pii_engine=self.pii_engine,
            # ... other dependencies
        )

        # In call_llm() method
        context = LLMCallContext(messages, model, provider, kwargs)
        result_context = await self.pipeline.execute(context)
        ```

    Stage Order (as per ADR-001):
    1. InjectionFirewall - Block/sanitize attacks
    2. PIIRedaction - Redact PII
    3. SensitivityRouting - Route based on sensitivity
    4. OutputContract - Inject contract prompt
    5. RAGContext - Inject retrieved context
    6. ProviderAdapter - Filter provider params
    7. ReasoningMode - Select reasoning mode (OpenRouter)
    [LLM CALL HAPPENS IN unified_client after pipeline]
    8. StyleGate - Score and gate response
    9. ConfidenceEscalation - Escalate if low confidence
    """

    stages = []

    # Stage 1: Injection Firewall
    if enable_injection_firewall:
        stages.append(InjectionFirewallStage(
            injection_firewall=injection_firewall,
            enabled=True
        ))
        logger.debug("Added InjectionFirewallStage")

    # Stage 2: PII Redaction
    if enable_pii_redaction:
        stages.append(PIIRedactionStage(
            pii_engine=pii_engine,
            enabled=True
        ))
        logger.debug("Added PIIRedactionStage")

    # Stage 3: Sensitivity Routing
    if enable_sensitivity_routing:
        stages.append(SensitivityRoutingStage(
            sensitivity_router=sensitivity_router,
            available_providers=available_providers or [],
            enabled=True
        ))
        logger.debug("Added SensitivityRoutingStage")

    # Stage 4: Output Contract
    if enable_output_contracts:
        stages.append(OutputContractStage(
            get_contract_prompt_func=get_contract_prompt_func,
            enabled=True
        ))
        logger.debug("Added OutputContractStage")

    # Stage 5: RAG Context Injection
    if enable_rag_context:
        stages.append(RAGContextInjectionStage(
            build_context_func=build_rag_context_func,
            rag_k=rag_k,
            enabled=True
        ))
        logger.debug("Added RAGContextInjectionStage")

    # Stage 6: Provider Adapter
    if enable_provider_adapter:
        stages.append(ProviderAdapterStage(
            enabled=True
        ))
        logger.debug("Added ProviderAdapterStage")

    # Stage 7: Reasoning Mode (OpenRouter only)
    if enable_reasoning_mode:
        stages.append(ReasoningModeStage(
            reasoning_selector=reasoning_selector,
            enabled=True
        ))
        logger.debug("Added ReasoningModeStage")

    # NOTE: LLM CALL happens in unified_client between Stage 7 and 8

    # Stage 8: Style Gate (post-LLM)
    if enable_style_gate:
        stages.append(StyleGateStage(
            style_scorer=style_scorer,
            style_gate=style_gate,
            enabled=True
        ))
        logger.debug("Added StyleGateStage")

    # Stage 9: Confidence Escalation (post-LLM)
    if enable_confidence_escalation:
        stages.append(ConfidenceEscalationStage(
            escalate_func=escalate_func,
            threshold=confidence_threshold,
            enabled=True
        ))
        logger.debug("Added ConfidenceEscalationStage")

    # Create pipeline
    pipeline = LLMCallPipeline(stages=stages)

    logger.info(
        f"✅ Created LLM pipeline with {len(stages)} stages "
        f"({len(pipeline.get_enabled_stages())} enabled)"
    )

    return pipeline
