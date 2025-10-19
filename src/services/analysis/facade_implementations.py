# src/services/analysis/facade_implementations.py
from typing import Dict, Any, List
from .contracts import (
    IPromptBuilder,
    IConsultantRunner,
    IResultAggregator,
    IEvidenceEmitter,
)


class V1PromptBuilder(IPromptBuilder):
    """Facade that allows the orchestrator to call into a prompt builder seam.
    For PR-02b this service is fully responsible for building all prompt sections and final prompt.
    """

    # ===== Private helpers (extracted from orchestrator) =====
    def _get_consultant_database(self) -> Dict[str, Dict[str, Any]]:
        return {
            "strategic_analyst": {
                "mental_models": [
                    "Porter's Five Forces",
                    "Blue Ocean Strategy",
                    "SWOT Analysis",
                    "BCG Matrix",
                ]
            },
            "market_researcher": {
                "mental_models": [
                    "Market Segmentation",
                    "Customer Journey Mapping",
                    "TAM/SAM/SOM",
                    "Competitive Benchmarking",
                ]
            },
            "financial_analyst": {
                "mental_models": [
                    "DCF Analysis",
                    "Scenario Modeling",
                    "Break-even Analysis",
                    "NPV/IRR",
                ]
            },
            "operations_expert": {
                "mental_models": [
                    "Lean Six Sigma",
                    "Value Stream Mapping",
                    "Theory of Constraints",
                    "Kaizen",
                ]
            },
            "implementation_specialist": {
                "mental_models": [
                    "Kotter's 8-Step Process",
                    "ADKAR Model",
                    "Agile Methodology",
                    "Stakeholder Mapping",
                ]
            },
            "innovation_consultant": {
                "mental_models": [
                    "Design Thinking",
                    "Lean Startup",
                    "Jobs-to-be-Done",
                    "Innovation Pipeline",
                ]
            },
            "technology_advisor": {
                "mental_models": [
                    "Technology Adoption Curve",
                    "Digital Maturity Model",
                    "Systems Thinking",
                    "Cloud Architecture",
                ]
            },
            "crisis_manager": {
                "mental_models": [
                    "Crisis Response Framework",
                    "Risk Matrix",
                    "Incident Command System",
                    "Business Continuity Planning",
                ]
            },
            "turnaround_specialist": {
                "mental_models": [
                    "Turnaround Management",
                    "Performance Improvement",
                    "Cost Reduction",
                    "Cash Flow Management",
                ]
            },
        }

    def _build_context_section(
        self, consultant: Dict[str, Any], dispatch_info: Dict[str, Any]
    ) -> str:
        specialization = consultant.get("specialization", "")
        assigned_dimensions = consultant.get("assigned_dimensions", [])
        pattern_name = (dispatch_info or {}).get("pattern_name", "")
        interaction_strategy = (dispatch_info or {}).get("interaction_strategy", "")
        return f"""You are analyzing a strategic business challenge that requires your specialized expertise in {specialization}.

Your assigned analytical dimensions:
{chr(10).join([f"• {dim}" for dim in assigned_dimensions])}

Interaction Pattern: {pattern_name}
Team Strategy: {interaction_strategy}

Your analysis will be combined with insights from other specialized consultants to provide comprehensive strategic guidance."""

    def _build_role_section(self, consultant: Dict[str, Any]) -> str:
        consultant_db = self._get_consultant_database()
        consultant_data = consultant_db.get(consultant.get("consultant_id", ""), {})
        mental_models = consultant_data.get("mental_models", [])
        consultant_type = consultant.get("consultant_type", "")
        specialization = consultant.get("specialization", "")
        return f"""You are a {consultant_type} with deep expertise in {specialization}.

Your analytical toolkit includes these proven frameworks:
{chr(10).join([f"• {model}" for model in mental_models])}

Apply your specialized knowledge and analytical frameworks to provide insights that only an expert in your field would identify. Focus on aspects that generalists might miss."""

    def _build_framework_section(self, consultant: Dict[str, Any]) -> str:
        assigned_dimensions = consultant.get("assigned_dimensions", [])
        return f"""Apply your expertise to analyze these specific dimensions:

{chr(10).join([f"**{dim}**" + chr(10) + "   - Conduct deep analysis specific to your expertise" + chr(10) + "   - Identify key insights and implications" + chr(10) + "   - Provide actionable recommendations" for dim in assigned_dimensions])}

Use your specialized mental models and frameworks to ensure analysis depth that matches your expertise level."""

    def _build_execution_instructions(self, consultant: Dict[str, Any]) -> str:
        return """1. **Deep Analysis**: Apply your specialized knowledge to thoroughly examine each assigned dimension
        
2. **Research Integration**: Where relevant, reference industry data, benchmarks, and best practices from your field

3. **Framework Application**: Use your specialized mental models to structure analysis and generate insights

4. **Actionable Recommendations**: Provide specific, implementable recommendations based on your expertise

5. **Risk Assessment**: Identify potential risks and mitigation strategies from your professional perspective

6. **Success Metrics**: Define how success should be measured in your area of expertise"""

    def _build_output_format_section(self) -> str:
        return """Structure your response as JSON:

```json
{
    "executive_summary": "Key findings and recommendations in 2-3 sentences",
    "detailed_analysis": "Comprehensive analysis addressing your assigned dimensions",
    "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
    "recommendations": ["Actionable recommendation 1", "Recommendation 2", "Recommendation 3"],
    "mental_models_applied": ["Framework 1", "Framework 2"],
    "research_citations": ["Source 1", "Source 2"],
    "risk_factors": ["Risk 1", "Risk 2"],
    "success_metrics": ["Metric 1", "Metric 2"],
    "confidence_level": 0.85
}
```

Provide thorough, professional analysis worthy of a premium consulting engagement."""

    # ===== Public API =====
    def build_context_section(
        self, consultant: Dict[str, Any], dispatch_info: Dict[str, Any]
    ) -> str:
        return self._build_context_section(consultant, dispatch_info)

    def build(self, consultant: Dict[str, Any], dispatch_info: Dict[str, Any]) -> str:
        """Build the 5-part base prompt exactly as the legacy orchestrator did."""
        ctx = self._build_context_section(consultant, dispatch_info)
        role = self._build_role_section(consultant)
        framework = self._build_framework_section(consultant)
        exec_instructions = self._build_execution_instructions(consultant)
        output_format = self._build_output_format_section()

        # IMPORTANT: Preserve formatting and spacing to keep golden-master outputs identical
        base_prompt = f"""# STRATEGIC CONSULTANT ANALYSIS REQUEST

## CONTEXT & PROBLEM
{ctx}

## YOUR ROLE & EXPERTISE
{role}

## ANALYTICAL FRAMEWORK
{framework}

## EXECUTION INSTRUCTIONS
{exec_instructions}

## REQUIRED OUTPUT FORMAT
{output_format}

Execute this analysis with the depth and rigor expected of a top-tier consulting firm. 
Provide actionable insights backed by logical reasoning and industry best practices."""
        return base_prompt


class V1ConsultantRunner(IConsultantRunner):
    """Facade seam for running a single consultant analysis.
    PR-03: This now contains the extracted execution logic from the orchestrator.
    """

    def __init__(self, context_stream=None) -> None:
        # Inject shared context stream for eventing/governance. Defaults to global singleton.
        try:
            from src.core.unified_context_stream import get_unified_context_stream

            self.context_stream = context_stream or get_unified_context_stream()
        except Exception:
            self.context_stream = None
        import logging as _logging

        self._logger = _logging.getLogger(__name__)

    async def run(
        self, consultant: Dict[str, Any], prompt: str, context: Dict[str, Any]
    ) -> Any:
        """Execute individual consultant analysis with V2 CoreOps execution paths.
        Returns ConsultantAnalysisResult or None on graceful degradation.
        """
        import time
        from datetime import datetime
        from src.orchestration.contracts import ConsultantAnalysisResult
        from src.core.unified_context_stream import ContextEventType

        start_time = time.time()
        logger = self._logger

        try:
            consultant_id = consultant.get("consultant_id")
            logger.info(f"📊 Analyzing with {consultant_id}...")

            # GOVERNANCE V2: Set agent context for this specific consultant
            agent_meta = (context or {}).get("agent_meta", {})
            if self.context_stream:
                self.context_stream.set_agent_context(
                    agent_contract_id=agent_meta.get("contract_id"),
                    agent_instance_id=agent_meta.get("instance_id"),
                )

                # DEEP INSTRUMENTATION: Log agent instantiation event
                self.context_stream.add_event(
                    ContextEventType.AGENT_INSTANTIATED,
                    data={
                        "consultant_id": consultant_id,
                        "consultant_type": consultant.get("consultant_type"),
                        "specialization": consultant.get("specialization"),
                        "assigned_dimensions": consultant.get(
                            "assigned_dimensions", []
                        ),
                        "instantiation_timestamp": datetime.now().isoformat(),
                        "execution_mode": "parallel_analysis_v2",
                        "station": "station_5_parallel_analysis",
                    },
                    metadata={
                        "agent_contract_id": agent_meta.get("contract_id"),
                        "agent_instance_id": agent_meta.get("instance_id"),
                        "forensic_instrumentation": True,
                    },
                )

            # V2 DEFAULT FOR STRATEGIST: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "mckinsey_strategist@1.0":
                from src.orchestration.cognitive_core_adapter import (
                    CognitiveCoreAdapter,
                )

                adapter = CognitiveCoreAdapter()
                payload = await adapter.execute_analysis("mckinsey_strategist@1.0")
                arguments = payload.get("arguments", [])
                analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                processing_time = time.time() - start_time
                result = ConsultantAnalysisResult(
                    consultant_id=consultant_id,
                    analysis_content=analysis_text,
                    mental_models_applied=["Outside View", "Competitive Dynamics"],
                    confidence_level=0.85,
                    key_insights=[a.claim for a in arguments[:3]],
                    recommendations=["Pilot entry under bounded conditions"],
                    research_citations=[],
                    processing_time_seconds=processing_time,
                    llm_tokens_used=0,
                    llm_cost_usd=0.0,
                )
                # Emit COREOPS_RUN_SUMMARY
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter as _AdapterRef,
                    )

                    program_path = _AdapterRef.PROGRAM_MAP.get(
                        "mckinsey_strategist@1.0"
                    )
                    if self.context_stream:
                        self.context_stream.add_event(
                            ContextEventType.COREOPS_RUN_SUMMARY,
                            {
                                "consultant_id": consultant_id,
                                "system_contract_id": "mckinsey_strategist@1.0",
                                "program_path": program_path,
                                "trace_id": payload.get("trace_id"),
                                "step_count": len(arguments),
                                "argument_examples": [a.claim for a in arguments[:3]],
                                "processing_time_seconds": processing_time,
                            },
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Could not emit COREOPS_RUN_SUMMARY (strategist): {e}"
                    )
                logger.info(
                    f"✅ V2 Strategist analysis completed via CoreOps for {consultant_id}"
                )
                return result

            # V2 DEFAULT FOR RISK ASSESSOR: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "risk_assessor@1.0":
                from src.orchestration.cognitive_core_adapter import (
                    CognitiveCoreAdapter,
                )

                adapter = CognitiveCoreAdapter()
                payload = await adapter.execute_analysis("risk_assessor@1.0")
                arguments = payload.get("arguments", [])
                analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                processing_time = time.time() - start_time
                result = ConsultantAnalysisResult(
                    consultant_id=consultant_id,
                    analysis_content=analysis_text,
                    mental_models_applied=["Risk Matrix", "Mitigation Planning"],
                    confidence_level=0.85,
                    key_insights=[a.claim for a in arguments[:3]],
                    recommendations=["Apply mitigations; monitor kill criteria"],
                    research_citations=[],
                    processing_time_seconds=processing_time,
                    llm_tokens_used=0,
                    llm_cost_usd=0.0,
                )
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter as _AdapterRef,
                    )

                    program_path = _AdapterRef.PROGRAM_MAP.get("risk_assessor@1.0")
                    if self.context_stream:
                        self.context_stream.add_event(
                            ContextEventType.COREOPS_RUN_SUMMARY,
                            {
                                "consultant_id": consultant_id,
                                "system_contract_id": "risk_assessor@1.0",
                                "program_path": program_path,
                                "trace_id": payload.get("trace_id"),
                                "step_count": len(arguments),
                                "argument_examples": [a.claim for a in arguments[:3]],
                                "processing_time_seconds": processing_time,
                            },
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Could not emit COREOPS_RUN_SUMMARY (risk_assessor): {e}"
                    )
                logger.info(
                    f"✅ V2 Risk Assessor analysis completed via CoreOps for {consultant_id}"
                )
                return result

            # V2 DEFAULT FOR OPERATIONS EXPERT: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "operations_expert@1.0":
                from src.orchestration.cognitive_core_adapter import (
                    CognitiveCoreAdapter,
                )

                adapter = CognitiveCoreAdapter()
                payload = await adapter.execute_analysis("operations_expert@1.0")
                arguments = payload.get("arguments", [])
                analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                processing_time = time.time() - start_time
                result = ConsultantAnalysisResult(
                    consultant_id=consultant_id,
                    analysis_content=analysis_text,
                    mental_models_applied=["TOC", "Lean"],
                    confidence_level=0.85,
                    key_insights=[a.claim for a in arguments[:3]],
                    recommendations=["Eliminate bottleneck; implement control plan"],
                    research_citations=[],
                    processing_time_seconds=processing_time,
                    llm_tokens_used=0,
                    llm_cost_usd=0.0,
                )
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter as _AdapterRef,
                    )

                    program_path = _AdapterRef.PROGRAM_MAP.get("operations_expert@1.0")
                    if self.context_stream:
                        self.context_stream.add_event(
                            ContextEventType.COREOPS_RUN_SUMMARY,
                            {
                                "consultant_id": consultant_id,
                                "system_contract_id": "operations_expert@1.0",
                                "program_path": program_path,
                                "trace_id": payload.get("trace_id"),
                                "step_count": len(arguments),
                                "argument_examples": [a.claim for a in arguments[:3]],
                                "processing_time_seconds": processing_time,
                            },
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Could not emit COREOPS_RUN_SUMMARY (operations_expert): {e}"
                    )
                logger.info(
                    f"✅ V2 Operations Expert analysis completed via CoreOps for {consultant_id}"
                )
                return result

            # V2 DEFAULT FOR MARKET RESEARCHER: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "market_researcher@1.0":
                from src.orchestration.cognitive_core_adapter import (
                    CognitiveCoreAdapter,
                )

                adapter = CognitiveCoreAdapter()
                payload = await adapter.execute_analysis("market_researcher@1.0")
                arguments = payload.get("arguments", [])
                analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                processing_time = time.time() - start_time
                result = ConsultantAnalysisResult(
                    consultant_id=consultant_id,
                    analysis_content=analysis_text,
                    mental_models_applied=["Market Sizing", "Segmentation", "JTBD"],
                    confidence_level=0.85,
                    key_insights=[a.claim for a in arguments[:3]],
                    recommendations=["Prioritize top segments; validate JTBD signals"],
                    research_citations=[],
                    processing_time_seconds=processing_time,
                    llm_tokens_used=0,
                    llm_cost_usd=0.0,
                )
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter as _AdapterRef,
                    )

                    program_path = _AdapterRef.PROGRAM_MAP.get("market_researcher@1.0")
                    if self.context_stream:
                        self.context_stream.add_event(
                            ContextEventType.COREOPS_RUN_SUMMARY,
                            {
                                "consultant_id": consultant_id,
                                "system_contract_id": "market_researcher@1.0",
                                "program_path": program_path,
                                "trace_id": payload.get("trace_id"),
                                "step_count": len(arguments),
                                "argument_examples": [a.claim for a in arguments[:3]],
                                "processing_time_seconds": processing_time,
                            },
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Could not emit COREOPS_RUN_SUMMARY (market_researcher): {e}"
                    )
                logger.info(
                    f"✅ V2 Market Researcher analysis completed via CoreOps for {consultant_id}"
                )
                return result

            # V2 DEFAULT FOR FINANCIAL ANALYST: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "financial_analyst@1.0":
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter,
                    )

                    adapter = CognitiveCoreAdapter()
                    payload = await adapter.execute_analysis("financial_analyst@1.0")
                    arguments = payload.get("arguments", [])
                    analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                    processing_time = time.time() - start_time
                    result = ConsultantAnalysisResult(
                        consultant_id=consultant_id,
                        analysis_content=analysis_text,
                        mental_models_applied=[
                            "DCF",
                            "Scenario Analysis",
                            "Sensitivity",
                        ],
                        confidence_level=0.85,
                        key_insights=[a.claim for a in arguments[:3]],
                        recommendations=[
                            "Focus on scenarios; test downside protections"
                        ],
                        research_citations=[],
                        processing_time_seconds=processing_time,
                        llm_tokens_used=0,
                        llm_cost_usd=0.0,
                    )
                    try:
                        from src.orchestration.cognitive_core_adapter import (
                            CognitiveCoreAdapter as _AdapterRef,
                        )

                        program_path = _AdapterRef.PROGRAM_MAP.get(
                            "financial_analyst@1.0"
                        )
                        if self.context_stream:
                            self.context_stream.add_event(
                                ContextEventType.COREOPS_RUN_SUMMARY,
                                {
                                    "consultant_id": consultant_id,
                                    "system_contract_id": "financial_analyst@1.0",
                                    "program_path": program_path,
                                    "trace_id": payload.get("trace_id"),
                                    "step_count": len(arguments),
                                    "argument_examples": [
                                        a.claim for a in arguments[:3]
                                    ],
                                    "processing_time_seconds": processing_time,
                                },
                            )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not emit COREOPS_RUN_SUMMARY (financial_analyst): {e}"
                        )
                    logger.info(
                        f"✅ V2 Financial Analyst analysis completed via CoreOps for {consultant_id}"
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"⚠️ V2 Financial Analyst path failed, falling back to legacy for {consultant_id}: {e}"
                    )

            # V2 DEFAULT FOR STRATEGIC ANALYST: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "strategic_analyst@1.0":
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter,
                    )

                    adapter = CognitiveCoreAdapter()
                    payload = await adapter.execute_analysis("strategic_analyst@1.0")
                    arguments = payload.get("arguments", [])
                    analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                    processing_time = time.time() - start_time
                    result = ConsultantAnalysisResult(
                        consultant_id=consultant_id,
                        analysis_content=analysis_text,
                        mental_models_applied=[
                            "Porter's Five Forces",
                            "Blue Ocean Strategy",
                            "SWOT Analysis",
                        ],
                        confidence_level=0.85,
                        key_insights=[a.claim for a in arguments[:3]],
                        recommendations=[
                            "Focus on competitive positioning and strategic options"
                        ],
                        research_citations=[],
                        processing_time_seconds=processing_time,
                        llm_tokens_used=0,
                        llm_cost_usd=0.0,
                    )
                    logger.info(
                        f"✅ V2 Strategic Analyst analysis completed via CoreOps for {consultant_id}"
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"⚠️ V2 Strategic Analyst path failed, falling back to legacy for {consultant_id}: {e}"
                    )

            # V2 DEFAULT FOR TECHNOLOGY ADVISOR: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "technology_advisor@1.0":
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter,
                    )

                    adapter = CognitiveCoreAdapter()
                    payload = await adapter.execute_analysis("technology_advisor@1.0")
                    arguments = payload.get("arguments", [])
                    analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                    processing_time = time.time() - start_time
                    result = ConsultantAnalysisResult(
                        consultant_id=consultant_id,
                        analysis_content=analysis_text,
                        mental_models_applied=[
                            "Architecture",
                            "Scalability",
                            "Security",
                        ],
                        confidence_level=0.85,
                        key_insights=[a.claim for a in arguments[:3]],
                        recommendations=[
                            "Adopt reference architecture; plan phased rollout"
                        ],
                        research_citations=[],
                        processing_time_seconds=processing_time,
                        llm_tokens_used=0,
                        llm_cost_usd=0.0,
                    )
                    try:
                        from src.orchestration.cognitive_core_adapter import (
                            CognitiveCoreAdapter as _AdapterRef,
                        )

                        program_path = _AdapterRef.PROGRAM_MAP.get(
                            "technology_advisor@1.0"
                        )
                        if self.context_stream:
                            self.context_stream.add_event(
                                ContextEventType.COREOPS_RUN_SUMMARY,
                                {
                                    "consultant_id": consultant_id,
                                    "system_contract_id": "technology_advisor@1.0",
                                    "program_path": program_path,
                                    "trace_id": payload.get("trace_id"),
                                    "step_count": len(arguments),
                                    "argument_examples": [
                                        a.claim for a in arguments[:3]
                                    ],
                                    "processing_time_seconds": processing_time,
                                },
                            )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not emit COREOPS_RUN_SUMMARY (technology_advisor): {e}"
                        )
                    logger.info(
                        f"✅ V2 Technology Advisor analysis completed via CoreOps for {consultant_id}"
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"⚠️ V2 Technology Advisor path failed, falling back to legacy for {consultant_id}: {e}"
                    )

            # V2 DEFAULT FOR IMPLEMENTATION SPECIALIST: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "implementation_specialist@1.0":
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter,
                    )

                    adapter = CognitiveCoreAdapter()
                    payload = await adapter.execute_analysis(
                        "implementation_specialist@1.0"
                    )
                    arguments = payload.get("arguments", [])
                    analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                    processing_time = time.time() - start_time
                    result = ConsultantAnalysisResult(
                        consultant_id=consultant_id,
                        analysis_content=analysis_text,
                        mental_models_applied=[
                            "Change Management",
                            "Stakeholder Plan",
                            "Milestones",
                        ],
                        confidence_level=0.85,
                        key_insights=[a.claim for a in arguments[:3]],
                        recommendations=["Define adoption KPIs; secure sponsorship"],
                        research_citations=[],
                        processing_time_seconds=processing_time,
                        llm_tokens_used=0,
                        llm_cost_usd=0.0,
                    )
                    try:
                        from src.orchestration.cognitive_core_adapter import (
                            CognitiveCoreAdapter as _AdapterRef,
                        )

                        program_path = _AdapterRef.PROGRAM_MAP.get(
                            "implementation_specialist@1.0"
                        )
                        if self.context_stream:
                            self.context_stream.add_event(
                                ContextEventType.COREOPS_RUN_SUMMARY,
                                {
                                    "consultant_id": consultant_id,
                                    "system_contract_id": "implementation_specialist@1.0",
                                    "program_path": program_path,
                                    "trace_id": payload.get("trace_id"),
                                    "step_count": len(arguments),
                                    "argument_examples": [
                                        a.claim for a in arguments[:3]
                                    ],
                                    "processing_time_seconds": processing_time,
                                },
                            )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not emit COREOPS_RUN_SUMMARY (implementation_specialist): {e}"
                        )
                    logger.info(
                        f"✅ V2 Implementation Specialist analysis completed via CoreOps for {consultant_id}"
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"⚠️ V2 Implementation Specialist path failed, falling back to legacy for {consultant_id}: {e}"
                    )

            # V2 DEFAULT FOR INNOVATION CONSULTANT: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "innovation_consultant@1.0":
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter,
                    )

                    adapter = CognitiveCoreAdapter()
                    payload = await adapter.execute_analysis(
                        "innovation_consultant@1.0"
                    )
                    arguments = payload.get("arguments", [])
                    analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                    processing_time = time.time() - start_time
                    result = ConsultantAnalysisResult(
                        consultant_id=consultant_id,
                        analysis_content=analysis_text,
                        mental_models_applied=[
                            "Divergent/Convergent",
                            "Portfolio",
                            "Experimentation",
                        ],
                        confidence_level=0.85,
                        key_insights=[a.claim for a in arguments[:3]],
                        recommendations=["Prioritize H2 bets; define kill criteria"],
                        research_citations=[],
                        processing_time_seconds=processing_time,
                        llm_tokens_used=0,
                        llm_cost_usd=0.0,
                    )
                    try:
                        from src.orchestration.cognitive_core_adapter import (
                            CognitiveCoreAdapter as _AdapterRef,
                        )

                        program_path = _AdapterRef.PROGRAM_MAP.get(
                            "innovation_consultant@1.0"
                        )
                        if self.context_stream:
                            self.context_stream.add_event(
                                ContextEventType.COREOPS_RUN_SUMMARY,
                                {
                                    "consultant_id": consultant_id,
                                    "system_contract_id": "innovation_consultant@1.0",
                                    "program_path": program_path,
                                    "trace_id": payload.get("trace_id"),
                                    "step_count": len(arguments),
                                    "argument_examples": [
                                        a.claim for a in arguments[:3]
                                    ],
                                    "processing_time_seconds": processing_time,
                                },
                            )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not emit COREOPS_RUN_SUMMARY (innovation_consultant): {e}"
                        )
                    logger.info(
                        f"✅ V2 Innovation Consultant analysis completed via CoreOps for {consultant_id}"
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"⚠️ V2 Innovation Consultant path failed, falling back to legacy for {consultant_id}: {e}"
                    )

            # V2 DEFAULT FOR CRISIS MANAGER: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "crisis_manager@1.0":
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter,
                    )

                    adapter = CognitiveCoreAdapter()
                    payload = await adapter.execute_analysis("crisis_manager@1.0")
                    arguments = payload.get("arguments", [])
                    analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                    processing_time = time.time() - start_time
                    result = ConsultantAnalysisResult(
                        consultant_id=consultant_id,
                        analysis_content=analysis_text,
                        mental_models_applied=["ICS", "Risk Controls", "Comms"],
                        confidence_level=0.85,
                        key_insights=[a.claim for a in arguments[:3]],
                        recommendations=["Execute playbook; run lessons learned"],
                        research_citations=[],
                        processing_time_seconds=processing_time,
                        llm_tokens_used=0,
                        llm_cost_usd=0.0,
                    )
                    try:
                        from src.orchestration.cognitive_core_adapter import (
                            CognitiveCoreAdapter as _AdapterRef,
                        )

                        program_path = _AdapterRef.PROGRAM_MAP.get("crisis_manager@1.0")
                        if self.context_stream:
                            self.context_stream.add_event(
                                ContextEventType.COREOPS_RUN_SUMMARY,
                                {
                                    "consultant_id": consultant_id,
                                    "system_contract_id": "crisis_manager@1.0",
                                    "program_path": program_path,
                                    "trace_id": payload.get("trace_id"),
                                    "step_count": len(arguments),
                                    "argument_examples": [
                                        a.claim for a in arguments[:3]
                                    ],
                                    "processing_time_seconds": processing_time,
                                },
                            )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not emit COREOPS_RUN_SUMMARY (crisis_manager): {e}"
                        )
                    logger.info(
                        f"✅ V2 Crisis Manager analysis completed via CoreOps for {consultant_id}"
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"⚠️ V2 Crisis Manager path failed, falling back to legacy for {consultant_id}: {e}"
                    )

            # V2 DEFAULT FOR TURNAROUND SPECIALIST: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "turnaround_specialist@1.0":
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter,
                    )

                    adapter = CognitiveCoreAdapter()
                    payload = await adapter.execute_analysis(
                        "turnaround_specialist@1.0"
                    )
                    arguments = payload.get("arguments", [])
                    analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                    processing_time = time.time() - start_time
                    result = ConsultantAnalysisResult(
                        consultant_id=consultant_id,
                        analysis_content=analysis_text,
                        mental_models_applied=["Liquidity", "ZBB", "PMO"],
                        confidence_level=0.85,
                        key_insights=[a.claim for a in arguments[:3]],
                        recommendations=["Stand up TO; track value weekly"],
                        research_citations=[],
                        processing_time_seconds=processing_time,
                        llm_tokens_used=0,
                        llm_cost_usd=0.0,
                    )
                    try:
                        from src.orchestration.cognitive_core_adapter import (
                            CognitiveCoreAdapter as _AdapterRef,
                        )

                        program_path = _AdapterRef.PROGRAM_MAP.get(
                            "turnaround_specialist@1.0"
                        )
                        if self.context_stream:
                            self.context_stream.add_event(
                                ContextEventType.COREOPS_RUN_SUMMARY,
                                {
                                    "consultant_id": consultant_id,
                                    "system_contract_id": "turnaround_specialist@1.0",
                                    "program_path": program_path,
                                    "trace_id": payload.get("trace_id"),
                                    "step_count": len(arguments),
                                    "argument_examples": [
                                        a.claim for a in arguments[:3]
                                    ],
                                    "processing_time_seconds": processing_time,
                                },
                            )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not emit COREOPS_RUN_SUMMARY (turnaround_specialist): {e}"
                        )
                    logger.info(
                        f"✅ V2 Turnaround Specialist analysis completed via CoreOps for {consultant_id}"
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"⚠️ V2 Turnaround Specialist path failed, falling back to legacy for {consultant_id}: {e}"
                    )

            # V2 DEFAULT FOR COMPETITIVE ANALYST: CoreOps + CognitiveCoreService
            if agent_meta.get("contract_id") == "competitive_analyst@1.0":
                try:
                    from src.orchestration.cognitive_core_adapter import (
                        CognitiveCoreAdapter,
                    )

                    adapter = CognitiveCoreAdapter()
                    payload = await adapter.execute_analysis("competitive_analyst@1.0")
                    arguments = payload.get("arguments", [])
                    analysis_text = "\n".join([f"- {a.claim}" for a in arguments])
                    processing_time = time.time() - start_time
                    result = ConsultantAnalysisResult(
                        consultant_id=consultant_id,
                        analysis_content=analysis_text,
                        mental_models_applied=["Five Forces", "Positioning", "Moats"],
                        confidence_level=0.85,
                        key_insights=[a.claim for a in arguments[:3]],
                        recommendations=["Close parity gaps; reinforce moats"],
                        research_citations=[],
                        processing_time_seconds=processing_time,
                        llm_tokens_used=0,
                        llm_cost_usd=0.0,
                    )
                    try:
                        from src.orchestration.cognitive_core_adapter import (
                            CognitiveCoreAdapter as _AdapterRef,
                        )

                        program_path = _AdapterRef.PROGRAM_MAP.get(
                            "competitive_analyst@1.0"
                        )
                        if self.context_stream:
                            self.context_stream.add_event(
                                ContextEventType.COREOPS_RUN_SUMMARY,
                                {
                                    "consultant_id": consultant_id,
                                    "system_contract_id": "competitive_analyst@1.0",
                                    "program_path": program_path,
                                    "trace_id": payload.get("trace_id"),
                                    "step_count": len(arguments),
                                    "argument_examples": [
                                        a.claim for a in arguments[:3]
                                    ],
                                    "processing_time_seconds": processing_time,
                                },
                            )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not emit COREOPS_RUN_SUMMARY (competitive_analyst): {e}"
                        )
                    logger.info(
                        f"✅ V2 Competitive Analyst analysis completed via CoreOps for {consultant_id}"
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"⚠️ V2 Competitive Analyst path failed, falling back to legacy for {consultant_id}: {e}"
                    )

            # Legacy V1 path removed: V2 CoreOps is now the canonical execution.
            return None

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"❌ {consultant.get('consultant_id')} analysis failed after {processing_time:.1f}s: {e}"
            )
            return None  # Graceful degradation
        finally:
            # Clear agent context after finishing this consultant
            if self.context_stream:
                self.context_stream.clear_agent_context()


class V1ResultAggregator(IResultAggregator):
    """Facade seam for result aggregation. Returns an empty aggregation in PR-01."""

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {}


class V1EvidenceEmitter(IEvidenceEmitter):
    """Facade seam for emitting evidence. No-ops in PR-01 to preserve behavior."""

    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        # Intentionally no-op to avoid duplicate emissions in PR-01
        return
