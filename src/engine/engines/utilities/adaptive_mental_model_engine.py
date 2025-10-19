#!/usr/bin/env python3
"""
Adaptive Mental Model Engine - ArXiv 2402.18252v1 Implementation
Separates constructive analysis from critical validation mental models

ARCHITECTURAL PRINCIPLES:
1. Constructive models for analysis (build up insights)
2. Critical models for validation (challenge assumptions)
3. Adaptive model selection based on problem type
4. Meta-cognitive reflection on effectiveness
5. Multi-disciplinary model integration
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.core.resilient_llm_client import ResilientLLMClient, CognitiveCallContext
from src.v4.core.v4_supabase_adapter import V4SupabaseAdapter


class ModelPhase(Enum):
    """Mental model application phases"""

    CONSTRUCTIVE_ANALYSIS = "constructive_analysis"
    CRITICAL_VALIDATION = "critical_validation"


class ModelDiscipline(Enum):
    """Multi-disciplinary model categories"""

    BUSINESS_STRATEGY = "business_strategy"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    SCIENTIFIC_METHOD = "scientific_method"
    BEHAVIORAL_PSYCHOLOGY = "behavioral_psychology"
    SYSTEMS_THINKING = "systems_thinking"


@dataclass
class MentalModelProfile:
    """Enhanced mental model with phase and discipline classification"""

    model_id: str
    name: str
    discipline: ModelDiscipline
    phase: ModelPhase
    description: str
    application_contexts: List[str]
    prompt_template: str
    effectiveness_score: float = 0.0
    usage_count: int = 0


@dataclass
class ModelApplicationResult:
    """Result of applying a specific mental model"""

    model_profile: MentalModelProfile
    application_context: str
    analysis_output: str
    confidence_score: float
    reasoning_quality: float
    execution_time_ms: int
    cost_usd: float
    insights_generated: List[str]


class AdaptiveMentalModelSelector:
    """
    Implements ArXiv 2402.18252v1 adaptive model selection
    Autonomously selects appropriate mental models for different problem types
    """

    # Constructive Models for Analysis Phase
    CONSTRUCTIVE_MODELS = {
        "market_analysis": [
            MentalModelProfile(
                model_id="porter_five_forces",
                name="Porter's Five Forces",
                discipline=ModelDiscipline.BUSINESS_STRATEGY,
                phase=ModelPhase.CONSTRUCTIVE_ANALYSIS,
                description="Analyze competitive forces shaping industry structure",
                application_contexts=[
                    "market_entry",
                    "competitive_strategy",
                    "industry_analysis",
                ],
                prompt_template="""Apply Porter's Five Forces framework to analyze: {problem_context}

Systematic analysis of competitive forces:
1. **Threat of New Entrants**: Barriers to entry, capital requirements, economies of scale
2. **Bargaining Power of Suppliers**: Supplier concentration, switching costs, forward integration
3. **Bargaining Power of Buyers**: Buyer concentration, price sensitivity, backward integration  
4. **Threat of Substitutes**: Alternative solutions, relative performance, switching costs
5. **Competitive Rivalry**: Number of competitors, industry growth, differentiation

For each force, assess:
- Current intensity (Low/Medium/High)
- Key drivers and underlying factors
- Strategic implications and opportunities
- Actionable recommendations

Provide structured analysis with evidence-based conclusions.""",
            ),
            MentalModelProfile(
                model_id="blue_ocean_strategy",
                name="Blue Ocean Strategy",
                discipline=ModelDiscipline.BUSINESS_STRATEGY,
                phase=ModelPhase.CONSTRUCTIVE_ANALYSIS,
                description="Identify uncontested market spaces and value innovation opportunities",
                application_contexts=[
                    "innovation_strategy",
                    "market_creation",
                    "differentiation",
                ],
                prompt_template="""Apply Blue Ocean Strategy framework to: {problem_context}

Value Innovation Analysis:
1. **Eliminate**: What factors should be eliminated that the industry takes for granted?
2. **Reduce**: What factors should be reduced well below industry standard?
3. **Raise**: What factors should be raised well above industry standard? 
4. **Create**: What factors should be created that the industry has never offered?

Strategic Canvas:
- Map current competitive factors and their levels
- Identify value curve opportunities
- Design new value proposition

Focus on creating uncontested market space through differentiation AND low cost.
Provide specific recommendations for value innovation.""",
            ),
        ],
        "business_modeling": [
            MentalModelProfile(
                model_id="value_proposition_canvas",
                name="Value Proposition Canvas",
                discipline=ModelDiscipline.BUSINESS_STRATEGY,
                phase=ModelPhase.CONSTRUCTIVE_ANALYSIS,
                description="Align products/services with customer jobs, pains, and gains",
                application_contexts=[
                    "customer_retention",
                    "product_development",
                    "market_fit",
                ],
                prompt_template="""Apply Value Proposition Canvas to: {problem_context}

Customer Profile Analysis:
1. **Jobs-to-be-Done**: What jobs are customers trying to get done?
   - Functional jobs (tasks, problems to solve)
   - Emotional jobs (feelings, status)
   - Social jobs (how others perceive them)

2. **Pains**: What frustrates customers before, during, after the job?
   - Undesired outcomes and problems
   - Obstacles preventing job completion
   - Risks of poor outcomes

3. **Gains**: What outcomes do customers want?
   - Required benefits and features
   - Expected outcomes and benefits  
   - Desired outcomes and benefits
   - Unexpected benefits that delight

Value Map Design:
- Products & Services offered
- Pain Relievers (how you address pains)
- Gain Creators (how you create gains)

Analyze fit between value proposition and customer profile.
Identify gaps and optimization opportunities.""",
            ),
            MentalModelProfile(
                model_id="jobs_to_be_done",
                name="Jobs-to-be-Done Framework",
                discipline=ModelDiscipline.BEHAVIORAL_PSYCHOLOGY,
                phase=ModelPhase.CONSTRUCTIVE_ANALYSIS,
                description="Understand the fundamental job customers hire your product to do",
                application_contexts=[
                    "customer_retention",
                    "product_development",
                    "innovation",
                ],
                prompt_template="""Apply Jobs-to-be-Done framework to: {problem_context}

Job Story Format: "When I [situation], I want to [motivation], so I can [expected outcome]"

Job Analysis:
1. **Functional Job**: What task is the customer trying to accomplish?
2. **Emotional Job**: How does the customer want to feel or be perceived?
3. **Social Job**: How does this relate to others' perceptions?

Job Execution Analysis:
- Current solutions customers use
- Workarounds and compensating behaviors
- Unmet needs in job execution
- Success metrics for job completion

Competition Analysis:
- What else competes for this job?
- Why do customers fire existing solutions?
- What triggers customer search for alternatives?

Provide insights on better job execution and competitive advantages.""",
            ),
        ],
        "strategic_planning": [
            MentalModelProfile(
                model_id="mece_structuring",
                name="MECE Problem Structuring",
                discipline=ModelDiscipline.MATHEMATICAL_REASONING,
                phase=ModelPhase.CONSTRUCTIVE_ANALYSIS,
                description="Structure problems in Mutually Exclusive, Collectively Exhaustive way",
                application_contexts=[
                    "strategic_planning",
                    "problem_solving",
                    "analysis",
                ],
                prompt_template="""Apply MECE (Mutually Exclusive, Collectively Exhaustive) structuring to: {problem_context}

Problem Decomposition:
1. **Issue Tree Development**: Break down the problem into component parts
   - Each branch should be mutually exclusive (no overlap)
   - All branches together should be collectively exhaustive (covers everything)
   - Use 2-4 branches per level for clarity

2. **Hypothesis Formation**: For each branch, develop testable hypotheses
   - What could be driving this issue?
   - How can we validate or disprove each hypothesis?

3. **Priority Analysis**: Assess each branch for:
   - Impact potential (High/Medium/Low)
   - Feasibility of intervention (Easy/Medium/Hard)
   - Evidence strength (Strong/Weak)

4. **Action Planning**: For high-impact, high-feasibility areas:
   - Specific interventions needed
   - Success metrics and timelines
   - Resource requirements

Ensure logical flow and no gaps in analysis structure.""",
            )
        ],
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_effectiveness_cache = {}

    def select_constructive_models(
        self, problem_context: Dict[str, Any]
    ) -> List[MentalModelProfile]:
        """
        Select 2-3 constructive models for analysis phase
        Based on problem type, industry, and historical effectiveness
        """

        problem_type = problem_context.get("problem_type", "strategic_planning")
        industry = problem_context.get("industry", "general")
        complexity = problem_context.get("complexity", 0.5)

        selected_models = []

        # Map problem types to model categories
        if problem_type in ["market_entry", "competitive_strategy"]:
            selected_models.extend(
                self.CONSTRUCTIVE_MODELS["market_analysis"][:1]
            )  # Porter 5 Forces

        if problem_type in ["customer_retention", "product_development"]:
            selected_models.extend(
                self.CONSTRUCTIVE_MODELS["business_modeling"]
            )  # Value Prop + JTBD

        # Always include MECE for complex problems
        if complexity > 0.6:
            selected_models.extend(
                self.CONSTRUCTIVE_MODELS["strategic_planning"]
            )  # MECE

        # Limit to 3 models maximum to avoid overload
        selected_models = selected_models[:3]

        self.logger.info(
            f"🎯 Selected {len(selected_models)} constructive models for {problem_type}"
        )
        return selected_models

    def get_model_by_id(self, model_id: str) -> Optional[MentalModelProfile]:
        """Get specific model profile by ID"""
        for category in self.CONSTRUCTIVE_MODELS.values():
            for model in category:
                if model.model_id == model_id:
                    return model
        return None


class AdaptiveMentalModelEngine:
    """
    Main engine implementing adaptive mental model selection and application
    Separates constructive analysis from critical validation
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

        self.model_selector = AdaptiveMentalModelSelector()
        self.logger = logging.getLogger(__name__)

        self.logger.info("🧠 Adaptive Mental Model Engine initialized")

    async def execute_constructive_analysis(
        self, problem_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute constructive analysis phase using appropriate mental models
        NO CRITICAL MODELS HERE - only constructive frameworks
        """

        start_time = datetime.now()
        self.logger.info(
            "🏗️ Starting constructive analysis with adaptive model selection"
        )

        # Select appropriate constructive models
        selected_models = self.model_selector.select_constructive_models(
            problem_context
        )

        if not selected_models:
            self.logger.warning("⚠️ No constructive models selected, using default MECE")
            selected_models = [self.model_selector.get_model_by_id("mece_structuring")]

        # Apply models in parallel
        model_applications = []
        for model in selected_models:
            if model:
                application = await self._apply_constructive_model(
                    model, problem_context
                )
                model_applications.append(application)

        # Synthesize constructive insights
        synthesis = await self._synthesize_constructive_insights(model_applications)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        result = {
            "analysis_type": "adaptive_constructive",
            "models_applied": len(model_applications),
            "model_profiles": [app.model_profile.name for app in model_applications],
            "disciplines_covered": list(
                set(app.model_profile.discipline.value for app in model_applications)
            ),
            "constructive_insights": synthesis,
            "model_applications": model_applications,
            "execution_time_ms": execution_time,
            "total_cost": sum(app.cost_usd for app in model_applications),
            "competitive_advantage": True,
            "phase": ModelPhase.CONSTRUCTIVE_ANALYSIS.value,
        }

        self.logger.info(
            f"✅ Constructive analysis complete: {len(model_applications)} models, ${result['total_cost']:.4f}"
        )
        return result

    async def _apply_constructive_model(
        self, model: MentalModelProfile, problem_context: Dict[str, Any]
    ) -> ModelApplicationResult:
        """Apply a single constructive mental model"""

        start_time = datetime.now()

        # Build sophisticated prompt using model template
        formatted_prompt = model.prompt_template.format(
            problem_context=problem_context.get("enhanced_query", "Business analysis"),
            industry=problem_context.get("industry", "general"),
            complexity=problem_context.get("complexity", "medium"),
        )

        # Add meta-cognitive context
        meta_prompt = f"""<adaptive_mental_model_analysis>
SELECTED MODEL: {model.name}
DISCIPLINE: {model.discipline.value}
PHASE: {model.phase.value}

<model_context>
{model.description}
Application contexts: {', '.join(model.application_contexts)}
</model_context>

<problem_context>
{problem_context.get('enhanced_query', 'Business analysis')}
Industry: {problem_context.get('industry', 'general')}
Complexity: {problem_context.get('complexity', 'medium')}
</problem_context>

{formatted_prompt}

<meta_cognitive_reflection>
Why is this mental model appropriate for this problem?
What insights might this model reveal that others would miss?
How does this contribute to comprehensive analysis?
</meta_cognitive_reflection>
</adaptive_mental_model_analysis>"""

        # Execute with DeepSeek
        cognitive_context = CognitiveCallContext(
            engagement_id=self.engagement_id,
            phase="constructive_analysis",
            task_type=f"mental_model_{model.model_id}",
            complexity_score=0.8,
            time_constraints="thorough",
            quality_threshold=0.9,
        )

        try:
            result = await self.llm_client.execute_cognitive_call(
                meta_prompt, cognitive_context
            )

            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # Extract insights from the analysis
            insights = self._extract_insights(result.content, model)

            application_result = ModelApplicationResult(
                model_profile=model,
                application_context=problem_context.get("problem_type", "general"),
                analysis_output=result.content,
                confidence_score=result.confidence,
                reasoning_quality=self._assess_reasoning_quality(result.content),
                execution_time_ms=execution_time,
                cost_usd=result.cost_usd,
                insights_generated=insights,
            )

            self.logger.debug(
                f"✅ Applied {model.name}: {len(insights)} insights, {result.confidence:.2f} confidence"
            )
            return application_result

        except Exception as e:
            self.logger.error(f"❌ Failed to apply {model.name}: {e}")
            # Return minimal result to prevent pipeline failure
            return ModelApplicationResult(
                model_profile=model,
                application_context=problem_context.get("problem_type", "general"),
                analysis_output=f"Model application failed: {str(e)}",
                confidence_score=0.1,
                reasoning_quality=0.1,
                execution_time_ms=0,
                cost_usd=0.0,
                insights_generated=[],
            )

    def _extract_insights(
        self, analysis_output: str, model: MentalModelProfile
    ) -> List[str]:
        """Extract key insights from model application output"""

        insights = []
        lines = analysis_output.split("\n")

        # Look for structured insights, recommendations, key findings
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for actionable insights
            if any(
                keyword in line.lower()
                for keyword in [
                    "recommendation:",
                    "insight:",
                    "key finding:",
                    "opportunity:",
                    "strategic implication:",
                    "action:",
                    "priority:",
                ]
            ):
                insights.append(line)

            # Look for numbered recommendations
            elif line.startswith(("1.", "2.", "3.", "4.", "5.")):
                if len(line) > 10:  # Substantial content
                    insights.append(line)

        # If no structured insights found, extract first few substantial sentences
        if not insights:
            sentences = [
                s.strip() for s in analysis_output.split(".") if len(s.strip()) > 50
            ]
            insights = sentences[:3]

        return insights[:5]  # Max 5 insights per model

    def _assess_reasoning_quality(self, analysis_output: str) -> float:
        """Assess the reasoning quality of the analysis"""

        quality_indicators = {
            "structured_thinking": ["1.", "2.", "first", "second", "next"],
            "evidence_based": ["data", "evidence", "research", "study", "analysis"],
            "actionable": ["recommend", "action", "implement", "strategy", "approach"],
            "comprehensive": [
                "consider",
                "factor",
                "aspect",
                "dimension",
                "perspective",
            ],
            "specific": ["specific", "concrete", "measurable", "timeline", "metric"],
        }

        content_lower = analysis_output.lower()
        quality_score = 0.5  # Base score

        for category, indicators in quality_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                quality_score += 0.1

        # Length bonus (more comprehensive analysis)
        if len(analysis_output) > 1000:
            quality_score += 0.05
        if len(analysis_output) > 2000:
            quality_score += 0.05

        return min(1.0, quality_score)

    async def _synthesize_constructive_insights(
        self, model_applications: List[ModelApplicationResult]
    ) -> Dict[str, Any]:
        """Synthesize insights from multiple constructive models"""

        if not model_applications:
            return {"error": "No model applications to synthesize"}

        # Collect all insights
        all_insights = []
        for app in model_applications:
            all_insights.extend(app.insights_generated)

        # Group by themes/topics
        disciplines_covered = list(
            set(app.model_profile.discipline.value for app in model_applications)
        )
        models_applied = [app.model_profile.name for app in model_applications]

        # Calculate synthesis metrics
        avg_confidence = sum(app.confidence_score for app in model_applications) / len(
            model_applications
        )
        avg_quality = sum(app.reasoning_quality for app in model_applications) / len(
            model_applications
        )

        synthesis = {
            "executive_summary": f"Constructive analysis using {len(model_applications)} mental models from {len(disciplines_covered)} disciplines",
            "models_applied": models_applied,
            "disciplines_covered": disciplines_covered,
            "key_insights": all_insights[:10],  # Top 10 insights
            "synthesis_quality": {
                "average_confidence": avg_confidence,
                "average_reasoning_quality": avg_quality,
                "insight_count": len(all_insights),
                "model_diversity": len(disciplines_covered),
            },
            "next_phase": "Ready for critical validation using Red Team Council",
        }

        return synthesis


# Factory function
def create_adaptive_mental_model_engine(
    llm_client: ResilientLLMClient,
    supabase_adapter: V4SupabaseAdapter,
    engagement_id: str,
) -> AdaptiveMentalModelEngine:
    """Create adaptive mental model engine"""
    return AdaptiveMentalModelEngine(llm_client, supabase_adapter, engagement_id)
