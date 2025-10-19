#!/usr/bin/env python3
"""
Mental Model Analysis Engine - METIS Proprietary Cognitive Advantage
Replaces generic micro-steps with sophisticated mental model applications

This is the core competitive differentiator:
- 100+ Mental Models from Supabase
- Big 4 Consulting Framework Integration
- N-way Pattern Matching
- Pyramid Principle Synthesis
"""

import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

from src.engine.adapters.llm_client import ResilientLLMClient, CognitiveCallContext  # Migrated
from src.v4.core.v4_supabase_adapter import V4SupabaseAdapter


class ProblemType(Enum):
    """Classification of business problems for mental model selection"""

    MARKET_ENTRY = "market_entry"
    ORGANIZATIONAL_CHANGE = "organizational_change"
    STRATEGIC_PLANNING = "strategic_planning"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    INNOVATION_STRATEGY = "innovation_strategy"
    FINANCIAL_ANALYSIS = "financial_analysis"
    CUSTOMER_RETENTION = "customer_retention"
    COMPETITIVE_STRATEGY = "competitive_strategy"
    DIGITAL_TRANSFORMATION = "digital_transformation"
    GROWTH_STRATEGY = "growth_strategy"


class ConsultingFramework(Enum):
    """Big 4 consulting frameworks"""

    MCKINSEY_7S = "mckinsey_7s"
    MCKINSEY_ISSUE_TREES = "mckinsey_issue_trees"
    BCG_GROWTH_SHARE = "bcg_growth_share"
    BCG_EXPERIENCE_CURVE = "bcg_experience_curve"
    BAIN_NET_PROMOTER = "bain_net_promoter"
    DELOITTE_DIGITAL_MATURITY = "deloitte_digital_maturity"
    PORTER_FIVE_FORCES = "porter_five_forces"
    BLUE_OCEAN = "blue_ocean_strategy"


@dataclass
class MentalModelApplication:
    """Mental model application result with tracking data"""

    model_id: str
    model_name: str
    category: str
    application_result: str
    confidence_score: float
    effectiveness_score: float
    reasoning_steps: List[Dict[str, Any]]
    cost_usd: float
    execution_time_ms: int
    n_way_patterns_used: List[str]


@dataclass
class ConsultingFrameworkMapping:
    """Mapping between problems and consulting frameworks"""

    framework: ConsultingFramework
    mental_models: List[str]
    analysis_steps: List[str]
    prompt_template: str
    complexity_multiplier: float


class CognitiveProblemClassifier:
    """Classifies business problems and selects appropriate mental models"""

    # Big 4 Framework Mappings
    FRAMEWORK_MAPPINGS = {
        ProblemType.MARKET_ENTRY: ConsultingFrameworkMapping(
            framework=ConsultingFramework.PORTER_FIVE_FORCES,
            mental_models=[
                "porter_competitive_analysis",
                "blue_ocean",
                "scenario_analysis",
            ],
            analysis_steps=[
                "competitive_intensity",
                "supplier_power",
                "buyer_power",
                "threat_substitutes",
                "barriers_entry",
            ],
            prompt_template="Apply Porter's Five Forces using {mental_model} to analyze market entry for {problem_context}. Focus on: {analysis_step}",
            complexity_multiplier=0.8,
        ),
        ProblemType.ORGANIZATIONAL_CHANGE: ConsultingFrameworkMapping(
            framework=ConsultingFramework.MCKINSEY_7S,
            mental_models=[
                "ackoff_problem_dissolution",
                "kotter_8_steps",
                "systems_thinking",
            ],
            analysis_steps=[
                "strategy",
                "structure",
                "systems",
                "shared_values",
                "style",
                "staff",
                "skills",
            ],
            prompt_template="Apply McKinsey 7S model using {mental_model} to analyze organizational change in {problem_context}. Focus on: {analysis_step}",
            complexity_multiplier=0.9,
        ),
        ProblemType.STRATEGIC_PLANNING: ConsultingFrameworkMapping(
            framework=ConsultingFramework.MCKINSEY_ISSUE_TREES,
            mental_models=["munger_inversion", "mece_principle", "jobs_to_be_done"],
            analysis_steps=[
                "problem_disaggregation",
                "hypothesis_generation",
                "fact_gathering",
                "solution_design",
            ],
            prompt_template="Build MECE issue tree using {mental_model} to structure strategic planning for {problem_context}. Focus on: {analysis_step}",
            complexity_multiplier=0.9,
        ),
        ProblemType.CUSTOMER_RETENTION: ConsultingFrameworkMapping(
            framework=ConsultingFramework.BAIN_NET_PROMOTER,
            mental_models=[
                "customer_journey_mapping",
                "value_proposition_canvas",
                "root_cause_analysis",
            ],
            analysis_steps=[
                "nps_analysis",
                "customer_segmentation",
                "loyalty_drivers",
                "retention_strategies",
            ],
            prompt_template="Apply Net Promoter methodology using {mental_model} to improve customer retention for {problem_context}. Focus on: {analysis_step}",
            complexity_multiplier=0.7,
        ),
        ProblemType.FINANCIAL_ANALYSIS: ConsultingFrameworkMapping(
            framework=ConsultingFramework.BCG_EXPERIENCE_CURVE,
            mental_models=["dcf_modeling", "real_options", "sensitivity_analysis"],
            analysis_steps=[
                "cost_curve_analysis",
                "competitive_advantage",
                "pricing_strategy",
                "investment_evaluation",
            ],
            prompt_template="Apply financial analysis using {mental_model} for {problem_context}. Focus on: {analysis_step}",
            complexity_multiplier=0.6,
        ),
    }

    # N-way Pattern Keywords (from your 21 patterns)
    NWAY_PATTERNS = {
        "combinatorial_effects": ["multiple", "interaction", "synergy", "combined"],
        "latticework_thinking": [
            "framework",
            "model",
            "multidisciplinary",
            "interconnected",
        ],
        "inversion_analysis": ["what could go wrong", "failure", "reverse", "opposite"],
        "systems_thinking": ["system", "structure", "relationships", "holistic"],
        "bias_detection": ["assumption", "bias", "cognitive", "rational"],
    }

    def __init__(self, supabase_adapter: V4SupabaseAdapter):
        self.supabase_adapter = supabase_adapter
        self.logger = logging.getLogger(__name__)

    async def classify_problem_type(
        self, enhanced_query: Dict[str, Any]
    ) -> ProblemType:
        """Classify business problem using N-way pattern matching"""

        query_text = enhanced_query.get("enhanced_query", "").lower()
        industry = enhanced_query.get("industry", "").lower()

        # Pattern matching for problem classification
        if any(
            keyword in query_text
            for keyword in ["market", "entry", "expansion", "new market"]
        ):
            return ProblemType.MARKET_ENTRY
        elif any(
            keyword in query_text
            for keyword in ["organization", "change", "transformation", "restructure"]
        ):
            return ProblemType.ORGANIZATIONAL_CHANGE
        elif any(
            keyword in query_text
            for keyword in ["strategy", "planning", "direction", "growth"]
        ):
            return ProblemType.STRATEGIC_PLANNING
        elif any(
            keyword in query_text
            for keyword in ["customer", "retention", "churn", "loyalty"]
        ):
            return ProblemType.CUSTOMER_RETENTION
        elif any(
            keyword in query_text
            for keyword in ["efficiency", "operations", "cost", "process"]
        ):
            return ProblemType.OPERATIONAL_EFFICIENCY
        elif any(
            keyword in query_text
            for keyword in ["innovation", "product", "development", "r&d"]
        ):
            return ProblemType.INNOVATION_STRATEGY
        elif any(
            keyword in query_text
            for keyword in ["financial", "revenue", "profit", "investment"]
        ):
            return ProblemType.FINANCIAL_ANALYSIS
        elif any(
            keyword in query_text for keyword in ["digital", "technology", "automation"]
        ):
            return ProblemType.DIGITAL_TRANSFORMATION
        else:
            # Default to strategic planning for complex problems
            return ProblemType.STRATEGIC_PLANNING

    async def identify_nway_patterns(self, enhanced_query: Dict[str, Any]) -> List[str]:
        """Identify relevant N-way interaction patterns"""

        query_text = enhanced_query.get("enhanced_query", "").lower()
        relevant_patterns = []

        for pattern_name, keywords in self.NWAY_PATTERNS.items():
            if any(keyword in query_text for keyword in keywords):
                relevant_patterns.append(pattern_name)

        # Always include latticework thinking as base pattern
        if "latticework_thinking" not in relevant_patterns:
            relevant_patterns.append("latticework_thinking")

        self.logger.info(f"🎯 Identified N-way patterns: {relevant_patterns}")
        return relevant_patterns

    def get_consulting_framework(
        self, problem_type: ProblemType
    ) -> ConsultingFrameworkMapping:
        """Get appropriate consulting framework for problem type"""

        framework_mapping = self.FRAMEWORK_MAPPINGS.get(
            problem_type,
            self.FRAMEWORK_MAPPINGS[ProblemType.STRATEGIC_PLANNING],  # Default
        )

        self.logger.info(
            f"🎯 Selected framework: {framework_mapping.framework.value} for {problem_type.value}"
        )
        return framework_mapping


class MentalModelSelector:
    """Selects optimal mental models from Supabase based on problem context"""

    def __init__(self, supabase_adapter: V4SupabaseAdapter):
        self.supabase_adapter = supabase_adapter
        self.logger = logging.getLogger(__name__)

    async def load_mental_models_for_framework(
        self, framework_mapping: ConsultingFrameworkMapping
    ) -> List[Dict[str, Any]]:
        """Load mental models from Supabase for specific framework"""

        try:
            # Load mental models by IDs specified in framework
            mental_models = []
            for model_id in framework_mapping.mental_models:
                try:
                    response = (
                        await self.supabase_adapter.client.table("mental_models")
                        .select("*")
                        .eq("model_id", model_id)
                        .execute()
                    )
                    if response.data:
                        mental_models.append(
                            response.data[0] if response.data else None
                        )
                        self.logger.debug(f"✅ Loaded mental model: {model_id}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load model {model_id}: {e}")

            # If specific models not found, load by category
            if not mental_models:
                self.logger.warning("⚠️ Specific models not found, loading by category")
                result = (
                    await self.supabase_adapter.client.table("mental_models")
                    .select("*")
                    .eq("is_active", True)
                    .limit(5)
                    .execute()
                )
                mental_models = result.data if result.data else []

            self.logger.info(
                f"📚 Loaded {len(mental_models)} mental models for framework"
            )
            return mental_models

        except Exception as e:
            self.logger.error(f"❌ Failed to load mental models: {e}")
            # Return fallback models
            return await self._get_fallback_mental_models()

    async def _get_fallback_mental_models(self) -> List[Dict[str, Any]]:
        """Fallback mental models if Supabase fails"""
        return [
            {
                "id": "fallback_1",
                "model_id": "munger_inversion",
                "name": "Munger Inversion Thinking",
                "category": "analytical",
                "prompt_integration_guide": "Apply inversion thinking by asking: 'What could cause this strategy to fail catastrophically?' Use this to identify hidden risks and strengthen the approach.",
                "implementation_steps": [
                    "Identify desired outcome",
                    "List failure modes",
                    "Analyze risks",
                    "Design safeguards",
                ],
            }
        ]


class MentalModelAnalysisEngine:
    """
    Core engine that replaces generic micro-steps with sophisticated mental model applications
    This is METIS's proprietary competitive advantage
    """

    def __init__(
        self,
        llm_client: ResilientLLMClient,
        supabase_adapter: V4SupabaseAdapter,
        engagement_id: str,
    ):
        self.llm_client = llm_client
        self.supabase_adapter = supabase_adapter
        self.engagement_id = engagement_id

        # Initialize components
        self.problem_classifier = CognitiveProblemClassifier(supabase_adapter)
        self.model_selector = MentalModelSelector(supabase_adapter)

        self.logger = logging.getLogger(__name__)
        self.logger.info("🧠 Mental Model Analysis Engine initialized")

    async def execute_mental_model_analysis(
        self, enhanced_query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute sophisticated mental model-based analysis
        Replaces the generic 5 micro-steps with proprietary cognitive intelligence
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting mental model-based analysis")

        # Phase 1: Problem Classification & Framework Selection
        problem_type = await self.problem_classifier.classify_problem_type(
            enhanced_query
        )
        nway_patterns = await self.problem_classifier.identify_nway_patterns(
            enhanced_query
        )
        consulting_framework = self.problem_classifier.get_consulting_framework(
            problem_type
        )

        self.logger.info(
            f"📊 Problem: {problem_type.value}, Framework: {consulting_framework.framework.value}"
        )

        # Phase 2: Mental Model Selection
        mental_models = await self.model_selector.load_mental_models_for_framework(
            consulting_framework
        )

        # Phase 3: Execute Mental Model Applications
        model_applications = []
        total_cost = 0.0

        for i, model in enumerate(mental_models):
            for j, analysis_step in enumerate(consulting_framework.analysis_steps):
                self.logger.info(
                    f"🔍 Applying {model.get('name', 'Unknown')} to {analysis_step}"
                )

                application = await self._execute_single_model_application(
                    model=model,
                    analysis_step=analysis_step,
                    consulting_framework=consulting_framework,
                    enhanced_query=enhanced_query,
                    nway_patterns=nway_patterns,
                )

                model_applications.append(application)
                total_cost += application.cost_usd

                # Limit to prevent excessive costs (max 10 applications)
                if len(model_applications) >= 10:
                    break

            if len(model_applications) >= 10:
                break

        # Phase 4: Synthesis and Structuring
        synthesis = await self._synthesize_model_applications(
            model_applications, consulting_framework
        )

        # Phase 5: Store Applications for Learning
        await self._store_model_applications(model_applications)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        analysis_result = {
            "problem_type": problem_type.value,
            "consulting_framework": consulting_framework.framework.value,
            "nway_patterns": nway_patterns,
            "mental_models_applied": len(model_applications),
            "model_applications": model_applications,
            "synthesis": synthesis,
            "total_cost": total_cost,
            "execution_time_ms": execution_time,
            "competitive_advantage": "mental_model_based",
        }

        self.logger.info(
            f"✅ Mental model analysis complete: {len(model_applications)} applications, ${total_cost:.4f}"
        )
        return analysis_result

    async def _execute_single_model_application(
        self,
        model: Dict[str, Any],
        analysis_step: str,
        consulting_framework: ConsultingFrameworkMapping,
        enhanced_query: Dict[str, Any],
        nway_patterns: List[str],
    ) -> MentalModelApplication:
        """Execute single mental model application with sophisticated prompting"""

        start_time = datetime.now()

        # Build sophisticated prompt using model's integration guide
        model_prompt = self._build_model_specific_prompt(
            model=model,
            analysis_step=analysis_step,
            consulting_framework=consulting_framework,
            enhanced_query=enhanced_query,
            nway_patterns=nway_patterns,
        )

        # Execute with DeepSeek reasoning mode (high complexity)
        cognitive_context = CognitiveCallContext(
            engagement_id=self.engagement_id,
            phase="analysis",
            task_type=f"mental_model_{model.get('model_id', 'unknown')}",
            complexity_score=0.8
            * consulting_framework.complexity_multiplier,  # High complexity for mental models
            time_constraints="thorough",  # Allow reasoning time
            quality_threshold=0.9,  # High quality requirement
            cost_sensitivity="normal",  # Accept reasonable costs for quality
        )

        try:
            result = await self.llm_client.execute_cognitive_call(
                prompt=model_prompt, context=cognitive_context
            )

            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # Calculate effectiveness score based on output quality
            effectiveness_score = self._calculate_effectiveness_score(result, model)

            return MentalModelApplication(
                model_id=model.get("model_id", "unknown"),
                model_name=model.get("name", "Unknown Model"),
                category=model.get("category", "general"),
                application_result=result.content,
                confidence_score=result.confidence,
                effectiveness_score=effectiveness_score,
                reasoning_steps=getattr(result, "reasoning_steps", []),
                cost_usd=result.cost_usd,
                execution_time_ms=execution_time,
                n_way_patterns_used=nway_patterns,
            )

        except Exception as e:
            self.logger.error(f"❌ Mental model application failed: {e}")
            # Return minimal result to prevent pipeline failure
            return MentalModelApplication(
                model_id=model.get("model_id", "unknown"),
                model_name=model.get("name", "Unknown Model"),
                category=model.get("category", "general"),
                application_result=f"Application failed: {str(e)}",
                confidence_score=0.1,
                effectiveness_score=0.1,
                reasoning_steps=[],
                cost_usd=0.0,
                execution_time_ms=0,
                n_way_patterns_used=nway_patterns,
            )

    def _build_model_specific_prompt(
        self,
        model: Dict[str, Any],
        analysis_step: str,
        consulting_framework: ConsultingFrameworkMapping,
        enhanced_query: Dict[str, Any],
        nway_patterns: List[str],
    ) -> str:
        """Build sophisticated prompt using mental model integration guide with NWAY clusters"""

        base_template = consulting_framework.prompt_template
        model_integration_guide = model.get(
            "prompt_integration_guide", "Apply systematic analysis"
        )
        implementation_steps = model.get("implementation_steps", [])

        # Get relevant NWAY clusters (simulated - in full implementation would use DeepSeekNWayEngine)
        nway_cluster_context = self._get_nway_cluster_context(
            enhanced_query, nway_patterns
        )

        # Enhanced prompt with NWAY cluster integration
        sophisticated_prompt = f"""
<mental_model_analysis>
FRAMEWORK: {consulting_framework.framework.value.replace('_', ' ').title()}
MENTAL MODEL: {model.get('name', 'Unknown')}
ANALYSIS FOCUS: {analysis_step}

<problem_context>
{enhanced_query.get('enhanced_query', 'Business problem analysis')}
Industry: {enhanced_query.get('industry', 'General')}
Complexity: {enhanced_query.get('complexity', 'Medium')}
</problem_context>

<mental_model_application>
{model_integration_guide}

Implementation Steps:
{chr(10).join(f"- {step}" for step in implementation_steps)}
</mental_model_application>

<nway_cognitive_integration>
Active NWAY Patterns: {', '.join(nway_patterns)}
{nway_cluster_context}

**NWAY DIRECTIVE**: Apply combinatorial thinking where multiple mental models interact synergistically. 
Let the mental model clusters generate emergent insights beyond individual model analysis.
</nway_cognitive_integration>

<consulting_rigor>
Apply Big 4 consulting standards enhanced with NWAY sophistication:
- Executive-level clarity and structure
- Evidence-based recommendations with NWAY depth
- Risk mitigation considering model interactions
- Implementation practicality with mental model synergies
- Glass-box transparency of cognitive processes
</consulting_rigor>

<ultrathink_optimization>
Take adequate time for deep cognitive processing. Apply the 70/30 principle:
- 70% stable core analysis using proven frameworks
- 30% adaptive NWAY cluster synergies for breakthrough insights
</ultrathink_optimization>

Execute {analysis_step} analysis using {model.get('name')} mental model with NWAY-enhanced consulting rigor.
Provide structured analysis with clear reasoning, synergistic insights, and actionable recommendations.
</mental_model_analysis>
"""

        return sophisticated_prompt

    def _get_nway_cluster_context(
        self, enhanced_query: Dict[str, Any], nway_patterns: List[str]
    ) -> str:
        """Get NWAY cluster context for the prompt (enhanced integration point)"""

        # In full implementation, this would integrate with DeepSeekNWayEngine
        # For now, provide structured context based on patterns

        context_mapping = {
            "latticework_thinking": "Apply cross-disciplinary mental model combinations for comprehensive analysis",
            "inversion_analysis": "Use inversion thinking to identify failure modes and strengthen recommendations",
            "systems_thinking": "Consider system-wide effects and feedback loops in your analysis",
            "bias_detection": "Actively identify and mitigate cognitive biases throughout the analysis",
            "combinatorial_effects": "Look for synergistic effects when multiple factors interact",
        }

        if not nway_patterns:
            return "Standard mental model application - no specific NWAY clusters activated."

        context = "NWAY Cluster Context:\n"
        for pattern in nway_patterns:
            if pattern in context_mapping:
                context += f"• {pattern.replace('_', ' ').title()}: {context_mapping[pattern]}\n"
            else:
                context += f"• {pattern.replace('_', ' ').title()}: Advanced cognitive pattern application\n"

        return context

    def _calculate_effectiveness_score(self, result, model: Dict[str, Any]) -> float:
        """Calculate mental model application effectiveness"""

        # Base score from LLM confidence
        base_score = result.confidence

        # Bonus for model-specific indicators in the output
        content = result.content.lower()
        model_keywords = model.get("key_concepts", [])

        keyword_bonus = 0.0
        for keyword in model_keywords:
            if keyword.lower() in content:
                keyword_bonus += 0.02  # 2% per relevant keyword

        # Bonus for reasoning depth
        reasoning_bonus = min(len(result.reasoning_steps) * 0.01, 0.1)  # Max 10% bonus

        # Penalty for very short outputs (likely poor quality)
        length_penalty = 0.0
        if len(content) < 200:
            length_penalty = 0.15  # 15% penalty for shallow analysis

        effectiveness = base_score + keyword_bonus + reasoning_bonus - length_penalty
        return max(0.0, min(1.0, effectiveness))  # Clamp to [0, 1]

    async def _synthesize_model_applications(
        self,
        model_applications: List[MentalModelApplication],
        consulting_framework: ConsultingFrameworkMapping,
    ) -> Dict[str, Any]:
        """Synthesize mental model applications using Pyramid Principle"""

        # Extract key insights from all applications
        insights = []
        for app in model_applications:
            if app.effectiveness_score > 0.6:  # Only include effective applications
                insights.append(
                    {
                        "mental_model": app.model_name,
                        "insight": app.application_result[
                            :500
                        ],  # First 500 chars as summary
                        "confidence": app.confidence_score,
                        "effectiveness": app.effectiveness_score,
                    }
                )

        # Group insights by effectiveness
        high_impact = [i for i in insights if i["effectiveness"] > 0.8]
        medium_impact = [i for i in insights if 0.6 < i["effectiveness"] <= 0.8]

        synthesis = {
            "executive_summary": f"Analysis completed using {consulting_framework.framework.value} framework with {len(model_applications)} mental model applications",
            "key_insights": high_impact,
            "supporting_analysis": medium_impact,
            "framework_used": consulting_framework.framework.value,
            "total_applications": len(model_applications),
            "average_effectiveness": (
                sum(app.effectiveness_score for app in model_applications)
                / len(model_applications)
                if model_applications
                else 0
            ),
        }

        return synthesis

    async def _store_model_applications(
        self, model_applications: List[MentalModelApplication]
    ):
        """Store mental model applications for continuous learning"""

        for application in model_applications:
            try:
                await self.supabase_adapter.client.table(
                    "mental_model_applications"
                ).insert(
                    {
                        "engagement_id": self.engagement_id,
                        "mental_model_id": str(
                            application.model_id
                        ),  # Ensure string for UUID compatibility
                        "phase": "analysis",
                        "application_trigger": f"framework_analysis_{application.model_id}",
                        "application_result": application.application_result[
                            :1000
                        ],  # Truncate for storage
                        "effectiveness_score": application.effectiveness_score,
                        "outcome_impact": (
                            "positive"
                            if application.effectiveness_score > 0.7
                            else "neutral"
                        ),
                        "execution_time_ms": application.execution_time_ms,
                        "tokens_consumed": len(
                            application.application_result.split()
                        ),  # Rough token estimate
                    }
                ).execute()

                self.logger.debug(f"📊 Stored application: {application.model_id}")

            except Exception as e:
                self.logger.error(f"❌ Failed to store model application: {e}")
                # Continue with other applications even if one fails


# Factory function for easy integration
def create_mental_model_analysis_engine(
    llm_client: ResilientLLMClient,
    supabase_adapter: V4SupabaseAdapter,
    engagement_id: str,
) -> MentalModelAnalysisEngine:
    """Factory function to create Mental Model Analysis Engine"""
    return MentalModelAnalysisEngine(llm_client, supabase_adapter, engagement_id)
