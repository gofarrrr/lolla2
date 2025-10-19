"""
METIS V2 Problem Structuring Agent
The first stage of V2 Augmented Core architecture after SocraticCognitiveForge.

This agent transforms an EnhancedQuery into a robust StructuredAnalyticalFramework
using the NWAY_PROBLEM_DECONSTRUCTION_023 Core N-Way model.
"""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

# V5 Integration
from src.engine.adapters.context_stream import UnifiedContextStream, ContextEventType  # Migrated
from src.engine.core.tool_decision_framework import (
    ToolSelectionDecision,
    get_tool_decision_framework,
)

# V2 Contracts
from src.contracts.frameworks import StructuredAnalyticalFramework, FrameworkChunk

# Import EnhancedQuery from the correct location
try:
    from src.contracts.socratic_contracts import EnhancedQuery
except ImportError:
    from src.engine.engines.core.socratic_cognitive_forge import EnhancedQuery

# LLM Integration
from src.integrations.llm.unified_client import UnifiedLLMClient

# Database
try:
    from supabase import create_client, Client
except ImportError:
    print("Supabase client not available - using mock implementation")
    Client = None


@dataclass
class PSAResult:
    """Result of Problem Structuring Agent processing"""

    success: bool
    framework: Optional[StructuredAnalyticalFramework]
    processing_time_ms: int
    confidence_score: float
    error_message: Optional[str] = None
    fallback_used: bool = False


class ProblemStructuringAgent:
    """
    V2 Problem Structuring Agent - MECE Framework Generation

    Uses NWAY_PROBLEM_DECONSTRUCTION_023 to transform vague business problems
    into robust, hypothesis-driven analytical frameworks.
    """

    MIN_FRAMEWORK_CHUNKS = 4
    MAX_FRAMEWORK_ATTEMPTS = 3  # Increased from 2 to ensure 4+ dimensions

    def __init__(
        self,
        supabase_client: Optional[Client] = None,
        context_stream: Optional[UnifiedContextStream] = None,
    ):
        # V5 Integration
        self.context_stream = context_stream or UnifiedContextStream(max_events=50000)
        self.tool_framework = get_tool_decision_framework(
            context_stream=self.context_stream
        )

        # Database connection
        self.supabase = supabase_client

        # LLM Client
        self.llm_client = UnifiedLLMClient()

        # PSA Configuration
        self.agent_id = "problem_structuring_agent"
        self.core_nway_model_id = "NWAY_PROBLEM_DECONSTRUCTION_023"

        # Cache for Core model
        self._core_model_cache: Optional[Dict[str, Any]] = None

    def _log_context_event(
        self,
        event_type: ContextEventType,
        event_data: Dict[str, Any],
        engagement_id: Optional[str] = None,
    ):
        """Log event to unified context stream"""
        try:
            self.context_stream.add_event(
                event_type=event_type,
                data=event_data,
                metadata={
                    "agent": "problem_structuring_agent",
                    "phase": "v2_problem_structuring",
                    "engagement_id": engagement_id,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            print(f"Warning: Failed to log context event: {e}")

    async def _get_core_model(self) -> Optional[Dict[str, Any]]:
        """Retrieve the Core N-Way model from database with caching"""
        if self._core_model_cache:
            return self._core_model_cache

        if not self.supabase:
            print("⚠️ No Supabase client - using mock Core model")
            # Return mock for testing
            return {
                "interaction_id": "NWAY_PROBLEM_DECONSTRUCTION_023",
                "instructional_cue_apce": self._get_mock_instructions(),
                "emergent_effect_summary": "Mock Core model for testing",
            }

        try:
            # Try V2 query with nway_type column first
            result = (
                self.supabase.table("nway_interactions")
                .select(
                    "interaction_id, instructional_cue_apce, emergent_effect_summary, type"
                )
                .eq("interaction_id", self.core_nway_model_id)
                .eq("nway_type", "CORE")
                .single()
                .execute()
            )

            if result.data:
                self._core_model_cache = result.data
                return result.data
            else:
                print(f"❌ Core model {self.core_nway_model_id} not found in database")

        except Exception as e:
            print(f"❌ Error retrieving Core model with V2 schema: {e}")

            # Fallback: Try V1 query without nway_type column
            try:
                result = (
                    self.supabase.table("nway_interactions")
                    .select(
                        "interaction_id, instructional_cue_apce, emergent_effect_summary, type"
                    )
                    .eq("interaction_id", self.core_nway_model_id)
                    .single()
                    .execute()
                )

                if result.data:
                    print("✅ Retrieved Core model using V1 fallback schema")
                    self._core_model_cache = result.data
                    return result.data

            except Exception as e2:
                print(f"❌ V1 fallback also failed: {e2}")

        # Final fallback: use mock data
        print("⚠️ Using mock Core model for testing")
        return {
            "interaction_id": "NWAY_PROBLEM_DECONSTRUCTION_023",
            "instructional_cue_apce": self._get_mock_instructions(),
            "emergent_effect_summary": "MECE analytical framework generation with five-step protocol",
            "type": "CORE",
        }

    def _get_mock_instructions(self) -> str:
        """Mock instructions for testing when database is unavailable"""
        return """You are a world-class MECE agent, a 'Problem Structuring Agent' trained in the methodologies of top-tier strategy consulting. Your sole purpose is to take the user's enhanced query and deconstruct it into a robust, MECE, and hypothesis-driven analytical framework. You must execute the following five-step protocol:
1. **Challenge the Premise:** First, use Inversion to question the validity of the user's stated problem. Is it a symptom of a deeper issue?
2. **Deconstruct to First Principles:** Use the 'Five Whys' to strip the problem down to its core, undeniable facts.
3. **Map the System:** Briefly identify the primary 'stocks', 'flows', and 'feedback loops' at play.
4. **Build the Issue Tree:** Based on your findings, construct a clear, hierarchical, and MECE Issue Tree with 3-5 primary branches.
5. **Formulate Key Hypotheses:** For the top 2-3 most critical branches of your tree, state a clear, falsifiable hypothesis that the subsequent analysis should test.

Return your response as a structured JSON with the following format:
{
  "refined_problem_statement": "Enhanced problem statement after your analysis",
  "framework_chunks": [
    {
      "part_number": 1,
      "title": "Framework chunk title",
      "description": "Description of this analytical area",
      "assigned_nway_clusters": ["NWAY_CLUSTER_ID_1", "NWAY_CLUSTER_ID_2"],
      "key_hypotheses_to_test": ["Hypothesis 1", "Hypothesis 2"]
    }
  ]
}"""

    def _create_psa_prompt(
        self, enhanced_query: EnhancedQuery, core_model: Dict[str, Any], seed_dimensions: Optional[List[str]] = None
    ) -> str:
        """Create the complete PSA prompt using Core model instructions"""
        instructions = core_model.get(
            "instructional_cue_apce", self._get_mock_instructions()
        )

        seed_block = ""
        if seed_dimensions:
            try:
                seeds = "\n".join(f"- {s}" for s in seed_dimensions[:8])
                seed_block = f"\n\n## Seed Dimensions (domain-informed)\nThe framework should strongly consider covering these dimensions if relevant:\n{seeds}\n"
            except Exception:
                seed_block = ""

        prompt = f"""# Problem Structuring Agent - METIS V2 Core Methodology

## Your Mission
{instructions}

## Enhanced Query to Structure
**Original Problem Statement:** {enhanced_query.original_statement}

**Enhanced Context:** {enhanced_query.enhanced_statement}

**Additional Context:**
- Quality Level: {enhanced_query.quality_level}
- Confidence Score: {enhanced_query.confidence_score}
- User Context: {json.dumps(enhanced_query.context_enrichment, indent=2) if enhanced_query.context_enrichment else 'None provided'}

## Required Output Format
You must return a valid JSON response with this exact structure:

```json
{{
  "refined_problem_statement": "Your enhanced and refined problem statement after MECE analysis",
  "framework_chunks": [
    {{
      "part_number": 1,
      "title": "Analytical Framework Title",
      "description": "Clear description of what this chunk will analyze",
      "assigned_nway_clusters": ["NWAY_MARKET_ANALYSIS_001", "NWAY_COMPETITIVE_DYNAMICS_003"],
      "key_hypotheses_to_test": ["Specific falsifiable hypothesis 1", "Specific falsifiable hypothesis 2"]
    }}
  ]
}}
```

## Critical Requirements
1. ⚠️ MANDATORY: Create EXACTLY 4-5 framework chunks (NOT 2 or 3 - you MUST produce at least 4 distinct analytical dimensions)
2. Each hypothesis must be specific and falsifiable
3. Assign relevant N-Way cluster IDs (these will be used for Dynamic model selection)
4. Use MECE principles - chunks should be Mutually Exclusive and Collectively Exhaustive
5. Each chunk MUST focus on a genuinely different strategic dimension (e.g., Market Dynamics, Operational Execution, Financial Viability, Risk Management, Stakeholder Alignment)
6. CRITICAL: You MUST respond with a single, valid JSON object (no markdown, no code blocks, no explanatory text, no reasoning - ONLY raw JSON as specified above)
{seed_block}
Begin your MECE deconstruction now and return ONLY the JSON object:"""

        return prompt

    def _build_min_chunk_directive(self, previous_count: int, min_chunks: int) -> str:
        """Construct reinforcement directive to guarantee minimum framework coverage"""

        return f"""

🚨 CRITICAL STRUCTURING COMPLIANCE - OUTPUT REJECTED 🚨
- Your previous output contained only {previous_count} primary framework_chunks.
- This is INSUFFICIENT for comprehensive MECE analysis.
- Regenerate the JSON with EXACTLY {min_chunks} to 5 mutually exclusive chunks.
- Each chunk must concentrate on a genuinely DIFFERENT strategic dimension:
  * Market & Competitive Dynamics
  * Operational Execution & Capabilities
  * Financial Viability & Economics
  * Risk Management & Governance
  * Stakeholder Alignment & Change Management
- DO NOT create overlapping or generic dimensions
- Titles and descriptions must be unique and non-overlapping
- Maintain strict JSON-only output exactly as specified above
- FAILURE TO PRODUCE {min_chunks}+ CHUNKS WILL RESULT IN SYSTEM FALLBACK TO GENERIC TEMPLATES

REGENERATE NOW WITH {min_chunks}+ DISTINCT DIMENSIONS:
"""

    async def _execute_structuring_prompt(
        self,
        prompt: str,
        engagement_id: str,
        attempt_label: str = "primary",
    ) -> Optional[StructuredAnalyticalFramework]:
        """Execute core structuring prompt with logging and parsing"""

        self._log_context_event(
            event_type=ContextEventType.LLM_PROVIDER_REQUEST,
            event_data={
                "model_used": self.core_nway_model_id,
                "prompt_length": len(prompt),
                "reasoning_mode": "structured_analysis",
                "attempt": attempt_label,
            },
            engagement_id=engagement_id,
        )

        llm_response = await self.llm_client.call_llm(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-reasoner",
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )

        if not llm_response.content or len(llm_response.content.strip()) == 0:
            self._log_context_event(
                event_type=ContextEventType.ERROR_OCCURRED,
                event_data={
                    "error": "LLM call returned empty response",
                    "step": "llm_execution",
                    "attempt": attempt_label,
                },
                engagement_id=engagement_id,
            )
            return None

        parse_start_time = datetime.now()
        self._log_context_event(
            event_type=ContextEventType.TOOL_CALL_START,
            event_data={
                "tool": "framework_parsing",
                "tool_name": "framework_parsing",
                "llm_response_length": len(llm_response.content),
                "llm_tokens_used": getattr(llm_response, "tokens_used", 0),
                "llm_cost_usd": getattr(llm_response, "cost_usd", 0),
                "attempt": attempt_label,
            },
            engagement_id=engagement_id,
        )

        framework = self._parse_llm_response(llm_response.content, engagement_id)

        if framework:
            parse_latency_ms = int((datetime.now() - parse_start_time).total_seconds() * 1000)
            self._log_context_event(
                event_type=ContextEventType.TOOL_CALL_COMPLETE,
                event_data={
                    "tool": "framework_parsing_success",
                    "tool_name": "framework_parsing_success",
                    "latency_ms": parse_latency_ms,
                    "framework_chunks": len(framework.framework_chunks),
                    "refined_statement_length": len(framework.refined_problem_statement),
                    "total_hypotheses": sum(len(chunk.key_hypotheses_to_test) for chunk in framework.framework_chunks),
                    "nway_clusters_assigned": sum(len(chunk.assigned_nway_clusters) for chunk in framework.framework_chunks),
                    "attempt": attempt_label,
                },
                engagement_id=engagement_id,
            )

        else:
            self._log_context_event(
                event_type=ContextEventType.ERROR_OCCURRED,
                event_data={
                    "error": "Failed to parse LLM response into structured framework",
                    "attempt": attempt_label,
                    "raw_response": llm_response.content[:500],
                },
                engagement_id=engagement_id,
            )

        return framework

    def _parse_llm_response(
        self, response: str, engagement_id: str
    ) -> Optional[StructuredAnalyticalFramework]:
        """
        OPERATION CONSISTENCY: Parse structured JSON response from LLM.

        Battle-tested approach (Cognition.ai/Manus.im):
        - LLM returns deterministic JSON via response_format
        - No fragile regex extraction
        - Direct json.loads() with graceful fallback
        """
        try:
            # OPERATION CONSISTENCY: Direct JSON parsing (no regex)
            # The response_format parameter guarantees valid JSON output
            response = response.strip()
            parsed = json.loads(response)

            # Validate required fields
            if (
                "refined_problem_statement" not in parsed
                or "framework_chunks" not in parsed
            ):
                raise ValueError("Missing required fields in response")

            # Convert to FrameworkChunk objects
            framework_chunks = []
            for chunk_data in parsed["framework_chunks"]:
                chunk = FrameworkChunk(
                    part_number=chunk_data.get(
                        "part_number", len(framework_chunks) + 1
                    ),
                    title=chunk_data.get("title", ""),
                    description=chunk_data.get("description", ""),
                    assigned_nway_clusters=chunk_data.get("assigned_nway_clusters", []),
                    key_hypotheses_to_test=chunk_data.get("key_hypotheses_to_test", []),
                )
                framework_chunks.append(chunk)

            # CRITICAL FIX: If no framework chunks were created, fall back to reasoning extraction
            if len(framework_chunks) == 0:
                print("⚠️ JSON parsed successfully but contained 0 framework chunks - falling back to reasoning extraction")
                return self._create_framework_from_reasoning(response, engagement_id)

            # Create StructuredAnalyticalFramework
            framework = StructuredAnalyticalFramework(
                engagement_id=engagement_id,
                refined_problem_statement=parsed["refined_problem_statement"],
                framework_chunks=framework_chunks,
                core_nway_model_used=self.core_nway_model_id,
                confidence_score=0.85,  # Default confidence for successful parsing
            )

            return framework

        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response was: {response[:500]}...")
            return None
        except Exception as e:
            print(f"❌ Framework parsing error: {e}")
            return None

    def _create_framework_from_reasoning(
        self, reasoning_content: str, engagement_id: str
    ) -> StructuredAnalyticalFramework:
        """Create structured framework from DeepSeek's reasoning content when no JSON is present"""

        # Extract key insights from the reasoning content
        lines = reasoning_content.split("\n")

        # Generate refined problem statement from first substantial paragraph
        problem_statement_lines = []
        for line in lines[:10]:  # Look at first 10 lines
            line = line.strip()
            if len(line) > 50 and not line.startswith(
                "*"
            ):  # Substantial content, not markdown
                problem_statement_lines.append(line)
                if len(problem_statement_lines) >= 2:
                    break

        refined_problem_statement = (
            " ".join(problem_statement_lines)
            if problem_statement_lines
            else "Comprehensive strategic framework required to address the stated decision."
        )

        query_excerpt = " ".join(line.strip() for line in lines if line.strip())[:120]

        # Create default MECE framework chunks with broadly applicable dimensions
        chunks = [
            FrameworkChunk(
                part_number=1,
                title="Strategic Outcomes & Success Metrics",
                description="Define the explicit goals, value metrics, and time horizons guiding this decision.",
                assigned_nway_clusters=[
                    "NWAY_STRATEGIST_CLUSTER_009",
                    "NWAY_ANALYST_CLUSTER_007",
                ],
                key_hypotheses_to_test=[
                    f"Success metrics for '{query_excerpt}' are agreed and measurable.",
                    "Stakeholders share a clear view of desired outcomes and timing.",
                ],
            ),
            FrameworkChunk(
                part_number=2,
                title="Demand, Stakeholder & Market Dynamics",
                description="Surface customer demand drivers, stakeholder needs, and external benchmarks that influence the decision.",
                assigned_nway_clusters=[
                    "NWAY_DECOMPOSITION_003",
                    "NWAY_DECISION_TRILEMMA_004",
                ],
                key_hypotheses_to_test=[
                    "Critical stakeholder segments and their success criteria are fully understood.",
                    "External benchmarks reveal material lessons or constraints for this situation.",
                ],
            ),
            FrameworkChunk(
                part_number=3,
                title="Economic Engine & Resource Model",
                description="Assess financial viability, resource requirements, and trade-offs across key options.",
                assigned_nway_clusters=[
                    "NWAY_DIAGNOSTIC_SOLVING_014",
                    "NWAY_OUTLIER_ANALYSIS_017",
                ],
                key_hypotheses_to_test=[
                    "Unit economics and investment profile support the targeted outcomes.",
                    "Resource allocation scenarios are resilient under plausible downside cases.",
                ],
            ),
            FrameworkChunk(
                part_number=4,
                title="Risk, Governance & Execution Readiness",
                description="Identify execution risks, contingency plans, governance cadence, and leading indicators for course correction.",
                assigned_nway_clusters=[
                    "NWAY_BIAS_MITIGATION_019",
                    "NWAY_ETHICAL_GOVERNANCE_FRAMEWORK_026",
                ],
                key_hypotheses_to_test=[
                    "Material risks and assumption tests are documented with mitigation owners.",
                    "Governance and leading metrics provide early warning if the plan drifts off track.",
                ],
            ),
        ]

        return StructuredAnalyticalFramework(
            engagement_id=engagement_id,
            refined_problem_statement=refined_problem_statement,
            framework_chunks=chunks,
            core_nway_model_used=self.core_nway_model_id,
            confidence_score=0.75,  # Lower confidence since created from reasoning rather than structured output
        )

    def _calculate_confidence_score(
        self, framework: StructuredAnalyticalFramework
    ) -> float:
        """Calculate confidence score based on framework quality"""
        score = 0.0

        # Base score for successful creation
        score += 0.3

        # Points for problem statement quality
        if len(framework.refined_problem_statement) > 50:
            score += 0.2

        # Points for framework chunks quality
        if 3 <= len(framework.framework_chunks) <= 5:  # MECE optimal range
            score += 0.2
        else:
            score += 0.1

        # Points for hypotheses quality
        total_hypotheses = sum(
            len(chunk.key_hypotheses_to_test) for chunk in framework.framework_chunks
        )
        if total_hypotheses >= 4:
            score += 0.2
        else:
            score += 0.1

        # Points for N-Way cluster assignments
        total_clusters = sum(
            len(chunk.assigned_nway_clusters) for chunk in framework.framework_chunks
        )
        if total_clusters >= 6:
            score += 0.1
        else:
            score += 0.05

        return min(score, 1.0)

    async def process_query(self, enhanced_query: EnhancedQuery, seed_dimensions: Optional[List[str]] = None) -> PSAResult:
        """
        Main processing method: Transform EnhancedQuery into StructuredAnalyticalFramework
        """
        start_time = datetime.now()
        engagement_id = (
            getattr(enhanced_query, "engagement_id", None)
            or f"psa_{int(datetime.now().timestamp())}"
        )

        # Log start of PSA processing
        self._log_context_event(
            event_type=ContextEventType.ENGAGEMENT_STARTED,
            event_data={
                "agent": "problem_structuring_agent",
                "input_query": enhanced_query.enhanced_statement[:200] + "...",
                "core_model": self.core_nway_model_id,
            },
            engagement_id=engagement_id,
        )

        try:
            # Step 1: Retrieve Core N-Way model
            core_model = await self._get_core_model()
            if not core_model:
                error_msg = f"Failed to retrieve Core model {self.core_nway_model_id}"
                self._log_context_event(
                    event_type=ContextEventType.ERROR_OCCURRED,
                    event_data={"error": error_msg, "step": "core_model_retrieval"},
                    engagement_id=engagement_id,
                )

                processing_time = int(
                    (datetime.now() - start_time).total_seconds() * 1000
                )
                return PSAResult(
                    success=False,
                    framework=None,
                    processing_time_ms=processing_time,
                    confidence_score=0.0,
                    error_message=error_msg,
                    fallback_used=False,
                )

            # Step 2: Create PSA prompt
            prompt = self._create_psa_prompt(enhanced_query, core_model, seed_dimensions)

            # Step 3: Record tool decision
            tool_decision = ToolSelectionDecision(
                tool_name="unified_llm_client",
                confidence=0.9,
                reasoning="Core methodology requires sophisticated reasoning capabilities",
                alternative_tools=["direct_template", "rule_based_parser"],
                context_requirements=[
                    "Core N-Way model NWAY_PROBLEM_DECONSTRUCTION_023"
                ],
                expected_output_type="structured_analytical_framework",
            )

            # Tool decision recording for audit (disabled for now)
            # self.tool_framework.record_decision would be called here if method existed

            fallback_used = False

            framework = await self._execute_structuring_prompt(
                prompt,
                engagement_id=engagement_id,
                attempt_label="primary",
            )

            attempts = 1
            while (
                framework
                and len(framework.framework_chunks) < self.MIN_FRAMEWORK_CHUNKS
                and attempts < self.MAX_FRAMEWORK_ATTEMPTS
            ):
                self._log_context_event(
                    event_type=ContextEventType.ERROR_OCCURRED,
                    event_data={
                        "error": f"Insufficient framework coverage ({len(framework.framework_chunks)} chunks)",
                        "step": "framework_validation",
                        "attempt": f"reinforcement_{attempts + 1}",
                    },
                    engagement_id=engagement_id,
                )

                attempts += 1
                reinforcement_prompt = (
                    self._create_psa_prompt(enhanced_query, core_model, seed_dimensions)
                    + self._build_min_chunk_directive(
                        len(framework.framework_chunks), self.MIN_FRAMEWORK_CHUNKS
                    )
                )

                framework = await self._execute_structuring_prompt(
                    reinforcement_prompt,
                    engagement_id=engagement_id,
                    attempt_label=f"reinforcement_{attempts}",
                )

            if not framework or len(framework.framework_chunks) < self.MIN_FRAMEWORK_CHUNKS:
                self._log_context_event(
                    event_type=ContextEventType.ERROR_OCCURRED,
                    event_data={
                        "error": "Falling back to default framework template",
                        "step": "framework_validation",
                        "attempt": "fallback_template",
                    },
                    engagement_id=engagement_id,
                )
                fallback_used = True
                framework = self._create_framework_from_reasoning(
                    enhanced_query.enhanced_statement,
                    engagement_id,
                )

            # Step 6: Calculate confidence and finalize
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            confidence_score = self._calculate_confidence_score(framework)

            # Update framework with processing metadata
            framework.psa_processing_time_ms = processing_time
            framework.confidence_score = confidence_score

            # Log successful completion
            self._log_context_event(
                event_type=ContextEventType.ENGAGEMENT_COMPLETED,
                event_data={
                    "framework_chunks": len(framework.framework_chunks),
                    "confidence_score": confidence_score,
                    "processing_time_ms": processing_time,
                    "refined_statement": framework.refined_problem_statement[:100]
                    + "...",
                },
                engagement_id=engagement_id,
            )

            return PSAResult(
                success=True,
                framework=framework,
                processing_time_ms=processing_time,
                confidence_score=confidence_score,
                error_message=None,
                fallback_used=fallback_used,
            )

        except Exception as e:
            error_msg = f"PSA processing failed: {str(e)}"
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

            self._log_context_event(
                event_type=ContextEventType.ERROR_OCCURRED,
                event_data={
                    "error": error_msg,
                    "step": "general_execution",
                    "exception_type": type(e).__name__,
                },
                engagement_id=engagement_id,
            )

            return PSAResult(
                success=False,
                framework=None,
                processing_time_ms=processing_time,
                confidence_score=0.0,
                error_message=error_msg,
                fallback_used=False,
            )

    async def structure_problem(
        self, questions: List[Any], answers: List[Any] = None, seed_dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Alias method for StatefulPipelineOrchestrator compatibility"""
        # Convert questions to enhanced query format
        if questions and len(questions) > 0:
            # Extract text from questions if they're dictionaries
            question_texts = []
            for q in questions:
                if isinstance(q, dict):
                    question_texts.append(q.get("text", str(q)))
                else:
                    question_texts.append(str(q))

            combined_statement = "Analysis request based on questions: " + "; ".join(
                question_texts
            )
        else:
            combined_statement = "General analytical framework request"

        # Create enhanced query
        enhanced_query = EnhancedQuery(
            original_statement=combined_statement,
            enhanced_statement=combined_statement,
            context_enrichment={"questions": questions, "answers": answers or []},
            user_responses=[],
            quality_level=75,  # Default quality level
            confidence_score=0.8,
        )

        # Process with existing method
        result = await self.process_query(enhanced_query, seed_dimensions=seed_dimensions)

        if result.success and result.framework and len(result.framework.framework_chunks) > 0:
            # Create compatible data for orchestration.contracts.StructuredAnalyticalFramework
            from src.orchestration.contracts import FrameworkType, AnalyticalDimension

            # Convert framework chunks to AnalyticalDimension objects
            primary_dimensions = []
            for chunk in result.framework.framework_chunks:
                dimension = AnalyticalDimension(
                    dimension_name=chunk.title,
                    key_questions=chunk.key_hypotheses_to_test[
                        :3
                    ],  # Take first 3 as questions
                    analysis_approach=chunk.description,
                    priority_level=chunk.part_number,
                )
                primary_dimensions.append(dimension)

            return {
                "framework_type": FrameworkType.STRATEGIC_ANALYSIS,
                "primary_dimensions": primary_dimensions,
                "secondary_considerations": [
                    hypothesis
                    for chunk in result.framework.framework_chunks
                    for hypothesis in chunk.key_hypotheses_to_test[
                        3:
                    ]  # Remaining hypotheses
                ],
                "analytical_sequence": [
                    chunk.title for chunk in result.framework.framework_chunks
                ],
                "complexity_assessment": result.framework.refined_problem_statement,
                "recommended_consultant_types": [
                    "strategic_analyst",
                    "domain_expert",
                    "operational_consultant",
                ],
                "processing_time_seconds": result.processing_time_ms
                / 1000.0,  # Convert to seconds
                # Additional fields for pipeline compatibility
                "confidence_score": result.confidence_score,
                "full_framework": result.framework.dict(),
            }
        else:
            # Return fallback structure compatible with StructuredAnalyticalFramework
            from src.orchestration.contracts import FrameworkType, AnalyticalDimension

            fallback_dimensions = [
                AnalyticalDimension(
                    dimension_name="Strategic Analysis",
                    key_questions=["What are the strategic priorities?"],
                    analysis_approach="Strategic assessment",
                    priority_level=1,
                ),
                AnalyticalDimension(
                    dimension_name="Operational Review",
                    key_questions=["What operational improvements are needed?"],
                    analysis_approach="Operational analysis",
                    priority_level=2,
                ),
            ]

            return {
                "framework_type": FrameworkType.STRATEGIC_ANALYSIS,
                "primary_dimensions": fallback_dimensions,
                "secondary_considerations": [
                    "Risk mitigation",
                    "Implementation planning",
                ],
                "analytical_sequence": ["Strategic Analysis", "Operational Review"],
                "complexity_assessment": "Medium complexity analytical framework",
                "recommended_consultant_types": [
                    "strategic_analyst",
                    "operational_consultant",
                ],
                "processing_time_seconds": 0.1,
                # Additional fields for pipeline compatibility
                "confidence_score": 0.3,
                "error_message": result.error_message if result else "Unknown error",
            }
