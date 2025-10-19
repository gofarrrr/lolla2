"""
Model Application Engine - Extracted from model_manager.py
Handles LLM integration and model-specific application strategies
"""

import logging
import re
from typing import Dict, Optional, Any
from datetime import datetime

from src.engine.models.data_contracts import MentalModelDefinition
from src.interfaces.model_manager_interface import IModelApplicationStrategy

# LLM integration
try:
    from src.core.llm_integration_adapter import get_unified_llm_adapter
except ImportError:
    get_unified_llm_adapter = None


# Legacy imports for backward compatibility
try:
    from src.integrations.claude_client import get_claude_client, LLMCallType

    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    CLAUDE_CLIENT_AVAILABLE = True
except ImportError:
    CLAUDE_CLIENT_AVAILABLE = False

# Decision capture integration
try:
    from src.core.decision_capture import (
        DecisionTree,
        DecisionCriteria,
        DecisionAlternative,
    )

    DECISION_CAPTURE_AVAILABLE = True
except ImportError:
    DECISION_CAPTURE_AVAILABLE = False


class ModelApplicationEngine(IModelApplicationStrategy):
    """
    Model-specific application strategies with LLM integration
    Handles the actual execution of mental models through language models
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        settings: Optional[Any] = None,
        decision_capture: Optional[Any] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.settings = settings
        self.decision_capture = decision_capture

    def _extract_thinking_process(self, response_content: str) -> Dict[str, str]:
        """Extract <thinking> content from LLM response for cognitive exhaust capture"""
        import re

        thinking_pattern = r"<thinking>(.*?)</thinking>"
        thinking_match = re.search(
            thinking_pattern, response_content, re.DOTALL | re.IGNORECASE
        )

        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
            # Remove the thinking block from the main response
            clean_response = re.sub(
                thinking_pattern, "", response_content, flags=re.DOTALL | re.IGNORECASE
            ).strip()

            return {
                "thinking_process": thinking_content,
                "clean_response": clean_response,
            }
        else:
            return {"thinking_process": None, "clean_response": response_content}

    async def apply_model(
        self,
        model: MentalModelDefinition,
        problem_statement: str,
        business_context: Dict[str, Any],
        step_id: str,
        current_nway_pattern: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Route model application to specific method based on model type
        """
        # Route to specialized application methods
        if model.model_id == "nway_systems_thinking":
            return await self.apply_systems_thinking(
                problem_statement, business_context, step_id, current_nway_pattern
            )
        elif model.model_id == "nway_critical_thinking":
            return await self.apply_critical_thinking(
                problem_statement, business_context, step_id, current_nway_pattern
            )
        elif model.model_id == "nway_mece_structuring":
            return await self.apply_mece_structuring(
                problem_statement, business_context, step_id, current_nway_pattern
            )
        elif model.model_id == "nway_hypothesis_testing":
            return await self.apply_hypothesis_testing(
                problem_statement, business_context, step_id, current_nway_pattern
            )
        elif model.model_id == "nway_decision_frameworks":
            return await self.apply_decision_framework(
                problem_statement, business_context, step_id
            )
        else:
            return await self.apply_generic_model(
                model, problem_statement, business_context, step_id
            )

    async def apply_systems_thinking(
        self,
        problem_statement: str,
        business_context: Dict[str, Any],
        step_id: str,
        current_nway_pattern: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply systems thinking mental model with N-WAY enhanced prompting"""

        if not CLAUDE_CLIENT_AVAILABLE:
            return self._create_error_result(
                step_id, "nway_systems_thinking", "CLAUDE CLIENT NOT AVAILABLE"
            )

        try:
            claude = await get_claude_client()

            # Enhanced N-WAY systems thinking prompt with cognitive exhaust capture
            prompt = f"""You are a systems thinking expert applying rigorous analytical frameworks.

BUSINESS CONTEXT:
{business_context.get('industry', 'General business') if hasattr(business_context, 'get') else getattr(business_context, 'industry', 'General business')} | {business_context.get('company_size', 'Unknown size') if hasattr(business_context, 'get') else getattr(business_context, 'company_size', 'Unknown size')} | {business_context.get('urgency', 'Standard timeline') if hasattr(business_context, 'get') else getattr(business_context, 'urgency', 'Standard timeline')}

PROBLEM STATEMENT:
{problem_statement}

<thinking>
Let me work through this systematically using systems thinking:

1. First, let me understand what system I'm looking at - what are the boundaries? What's inside vs outside the system?

2. Who are the key stakeholders and what are their motivations, constraints, and power dynamics?

3. What are the core processes, information flows, and resource flows in this system?

4. Where do I see potential feedback loops - both reinforcing (amplifying) and balancing (stabilizing)?

5. What mental models might the stakeholders be using that could create blind spots?

6. Where are the delays in the system that might cause people to overreact or underreact?

7. What are the unintended consequences I can anticipate from current patterns?

8. Where are the highest leverage intervention points - places where small changes could create big impacts?

9. What could go wrong with my proposed interventions? What are the second and third-order effects?
</thinking>

SYSTEMS THINKING ANALYSIS:
Apply comprehensive systems thinking to identify:

1. SYSTEM COMPONENTS:
   - Key stakeholders and their roles
   - Critical processes and workflows
   - Resources and constraints
   - Information flows

2. INTERDEPENDENCIES:
   - Direct relationships between components
   - Indirect effects and feedback loops
   - Cascade effects and amplification points
   - Bottlenecks and leverage points

3. DYNAMIC BEHAVIOR:
   - Current system behavior patterns
   - Unintended consequences
   - Reinforcing vs. balancing loops
   - Delays and time dependencies

4. INTERVENTION POINTS:
   - High-leverage intervention opportunities
   - Potential system interventions ranked by impact
   - Resource requirements for interventions
   - Risk assessment for each intervention

Provide concrete, actionable systems insights focused on business impact."""

            response = await claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            response_content = response.content[0].text if response.content else ""

            # Extract thinking process for cognitive exhaust capture
            parsed_response = self._extract_thinking_process(response_content)

            return {
                "step_id": step_id,
                "model_id": "nway_systems_thinking",
                "reasoning": parsed_response["clean_response"],
                "thinking_process": parsed_response["thinking_process"],
                "confidence": 0.85,
                "processing_time": 0.0,
                "token_usage": (
                    response.usage.input_tokens + response.usage.output_tokens
                    if hasattr(response, "usage")
                    else 0
                ),
                "nway_pattern": current_nway_pattern,
            }

        except Exception as e:
            self.logger.error(f"❌ Systems thinking application failed: {e}")
            return self._create_error_result(step_id, "nway_systems_thinking", str(e))

    async def apply_critical_thinking(
        self,
        problem_statement: str,
        business_context: Dict[str, Any],
        step_id: str,
        current_nway_pattern: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply critical thinking mental model for assumption validation"""

        if not CLAUDE_CLIENT_AVAILABLE:
            return self._create_error_result(
                step_id, "nway_critical_thinking", "CLAUDE CLIENT NOT AVAILABLE"
            )

        try:
            claude = await get_claude_client()

            # Enhanced N-WAY critical thinking prompt with cognitive exhaust capture
            prompt = f"""You are a critical thinking expert focused on rigorous assumption validation.

BUSINESS CONTEXT:
{business_context.get('industry', 'General business') if hasattr(business_context, 'get') else getattr(business_context, 'industry', 'General business')} | {business_context.get('company_size', 'Unknown size') if hasattr(business_context, 'get') else getattr(business_context, 'company_size', 'Unknown size')} | {business_context.get('urgency', 'Standard timeline') if hasattr(business_context, 'get') else getattr(business_context, 'urgency', 'Standard timeline')}

PROBLEM STATEMENT:
{problem_statement}

<thinking>
Let me apply critical thinking systematically to this problem:

1. What assumptions am I making just by reading this problem statement? What's being taken for granted?

2. How might the way this problem is framed be limiting my thinking? What if the real problem is something else entirely?

3. What evidence do I actually have versus what I'm inferring? Where am I filling in gaps with assumptions?

4. What cognitive biases might be at play here - both in the problem statement and in my own analysis?
   - Am I anchoring on the first explanation I thought of?
   - Am I looking for evidence that confirms what I already believe?
   - Am I being influenced by recent examples I can easily recall?

5. If I were arguing the opposite position, what would be my strongest points?

6. What would a skeptic or devil's advocate say about each of my conclusions?

7. What information would I need to see to change my mind about this?

8. What are the second and third-order consequences of the solutions I'm considering?

9. What assumptions about human behavior, market dynamics, or organizational behavior am I making?
</thinking>

CRITICAL THINKING ANALYSIS:
Conduct rigorous assumption validation and bias detection:

1. ASSUMPTION IDENTIFICATION:
   - Explicit assumptions stated in the problem
   - Implicit assumptions underlying the problem framing
   - Hidden assumptions about causation
   - Assumptions about stakeholder behavior

2. ASSUMPTION VALIDATION:
   - Evidence supporting each assumption
   - Evidence contradicting each assumption  
   - Assumptions that require verification
   - Assumptions that are likely false

3. COGNITIVE BIAS DETECTION:
   - Confirmation bias indicators
   - Anchoring effects
   - Availability heuristic influences
   - Survivorship bias considerations

4. ALTERNATIVE PERSPECTIVES:
   - How would competitors frame this problem?
   - What would skeptics argue?
   - What are we NOT considering?
   - What evidence would change our conclusion?

Focus on intellectual honesty and assumption rigor for business decision-making."""

            response = await claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            response_content = response.content[0].text if response.content else ""

            # Extract thinking process using helper method
            thinking_data = self._extract_thinking_process(response_content)

            return {
                "step_id": step_id,
                "model_id": "nway_critical_thinking",
                "reasoning": response_content,
                "confidence": 0.88,
                "processing_time": 0.0,
                "token_usage": (
                    response.usage.input_tokens + response.usage.output_tokens
                    if hasattr(response, "usage")
                    else 0
                ),
                "nway_pattern": current_nway_pattern,
                "thinking_process": thinking_data["thinking_content"],
                "cleaned_response": thinking_data["cleaned_response"],
            }

        except Exception as e:
            self.logger.error(f"❌ Critical thinking application failed: {e}")
            return self._create_error_result(step_id, "nway_critical_thinking", str(e))

    async def apply_mece_structuring(
        self,
        problem_statement: str,
        business_context: Dict[str, Any],
        step_id: str,
        current_nway_pattern: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply MECE (Mutually Exclusive, Collectively Exhaustive) structuring"""

        if not CLAUDE_CLIENT_AVAILABLE:
            return self._create_error_result(
                step_id, "nway_mece_structuring", "CLAUDE CLIENT NOT AVAILABLE"
            )

        try:
            claude = await get_claude_client()

            # Enhanced N-WAY MECE prompt with cognitive exhaust capture
            prompt = f"""You are a McKinsey-trained consultant expert in MECE (Mutually Exclusive, Collectively Exhaustive) problem structuring.

BUSINESS CONTEXT:
{business_context.get('industry', 'General business') if hasattr(business_context, 'get') else getattr(business_context, 'industry', 'General business')} | {business_context.get('company_size', 'Unknown size') if hasattr(business_context, 'get') else getattr(business_context, 'company_size', 'Unknown size')} | {business_context.get('urgency', 'Standard timeline') if hasattr(business_context, 'get') else getattr(business_context, 'urgency', 'Standard timeline')}

PROBLEM STATEMENT:
{problem_statement}

<thinking>
Let me approach this MECE structuring systematically:

1. What is the fundamental nature of this problem? Is it operational, strategic, financial, organizational, or market-related?

2. What are the natural fault lines or dimensions along which I can divide this problem?
   - Time-based (short/medium/long-term)?
   - Stakeholder-based (internal/external, different departments)?
   - Process-based (different stages of a workflow)?
   - Impact-based (cost/revenue/risk/growth)?

3. Let me brainstorm potential top-level categories and test them:
   - Are these mutually exclusive? Do any overlap?
   - Are they collectively exhaustive? Am I missing anything?
   - Are they at the same level of abstraction?

4. Which of these categories are most critical to solving the underlying problem?

5. For the most critical categories, what are the logical sub-breakdowns?

6. How can I validate that my structure actually helps solve the problem rather than just organizing it?

7. What would a skeptic say about my categorization? Where might it be weak or incomplete?
</thinking>

MECE STRUCTURAL ANALYSIS:
Create a rigorous MECE breakdown of this problem:

1. PRIMARY MECE BREAKDOWN:
   - Identify 3-5 mutually exclusive categories that together are collectively exhaustive
   - Ensure no overlap between categories
   - Ensure all aspects of the problem are covered
   - Validate that categories are at the same level of abstraction

2. SECONDARY BREAKDOWNS:
   - Break down 1-2 most critical primary categories into sub-components
   - Apply MECE principle to subcategories
   - Ensure actionable granularity

3. PRIORITY RANKING:
   - Rank categories by business impact
   - Rank categories by implementation difficulty
   - Identify high-impact, low-effort opportunities

4. VALIDATION CHECKS:
   - Verify mutual exclusivity (no overlaps)
   - Verify collective exhaustiveness (nothing missing)
   - Confirm business relevance and actionability

Focus on creating a clear, actionable structure that enables systematic problem-solving."""

            response = await claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            response_content = response.content[0].text if response.content else ""

            # Extract thinking process using helper method
            thinking_data = self._extract_thinking_process(response_content)

            return {
                "step_id": step_id,
                "model_id": "nway_mece_structuring",
                "reasoning": response_content,
                "confidence": 0.90,
                "processing_time": 0.0,
                "token_usage": (
                    response.usage.input_tokens + response.usage.output_tokens
                    if hasattr(response, "usage")
                    else 0
                ),
                "nway_pattern": current_nway_pattern,
                "thinking_process": thinking_data["thinking_content"],
                "cleaned_response": thinking_data["cleaned_response"],
            }

        except Exception as e:
            self.logger.error(f"❌ MECE structuring application failed: {e}")
            return self._create_error_result(step_id, "nway_mece_structuring", str(e))

    async def apply_hypothesis_testing(
        self,
        problem_statement: str,
        business_context: Dict[str, Any],
        step_id: str,
        current_nway_pattern: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply hypothesis testing for validation-driven analysis"""

        if not CLAUDE_CLIENT_AVAILABLE:
            return self._create_error_result(
                step_id, "nway_hypothesis_testing", "CLAUDE CLIENT NOT AVAILABLE"
            )

        try:
            claude = await get_claude_client()

            # Enhanced N-WAY hypothesis testing prompt with cognitive exhaust capture
            prompt = f"""You are a research methodology expert applying rigorous hypothesis testing to business problems.

BUSINESS CONTEXT:
{business_context.get('industry', 'General business') if hasattr(business_context, 'get') else getattr(business_context, 'industry', 'General business')} | {business_context.get('company_size', 'Unknown size') if hasattr(business_context, 'get') else getattr(business_context, 'company_size', 'Unknown size')} | {business_context.get('urgency', 'Standard timeline') if hasattr(business_context, 'get') else getattr(business_context, 'urgency', 'Standard timeline')}

PROBLEM STATEMENT:
{problem_statement}

<thinking>
Let me think through this hypothesis testing systematically:

1. What are the different possible explanations for this problem or different potential solutions?
   - What's the most obvious explanation? (This will be my primary hypothesis)
   - What alternative explanations could there be?
   - What would the "do nothing" scenario look like?

2. For each potential hypothesis, what would I expect to see if it were true?
   - What specific outcomes or indicators would support it?
   - What would contradict or refute it?

3. How could I actually test these hypotheses in practice?
   - What data would I need to collect?
   - What experiments could I run?
   - How long would testing take and what would it cost?

4. What are the risks of being wrong about each hypothesis?
   - What happens if I think a hypothesis is true but it's actually false? (Type I error)
   - What happens if I think a hypothesis is false but it's actually true? (Type II error)
   - Which type of error would be more costly for the business?

5. What confounding factors or alternative explanations might muddy the results?

6. How can I design tests that are as clean and definitive as possible?
</thinking>

HYPOTHESIS TESTING FRAMEWORK:
Apply systematic hypothesis testing methodology:

1. HYPOTHESIS GENERATION:
   - Primary hypothesis (most likely explanation/solution)
   - 2-3 alternative hypotheses
   - Null hypothesis (status quo/no change)
   - Specify what each hypothesis predicts

2. TESTABILITY ASSESSMENT:
   - What evidence would support each hypothesis?
   - What evidence would refute each hypothesis?  
   - What data/metrics are needed for validation?
   - Timeline and cost for testing each hypothesis

3. VALIDATION DESIGN:
   - Minimum viable tests for each hypothesis
   - Success/failure criteria
   - Statistical significance requirements
   - Confounding variables to control for

4. RISK ASSESSMENT:
   - Risk of false positives (Type I error)
   - Risk of false negatives (Type II error)
   - Business consequences of each error type
   - Mitigation strategies for high-risk scenarios

Focus on creating testable, falsifiable hypotheses with clear validation pathways."""

            response = await claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            response_content = response.content[0].text if response.content else ""

            # Extract thinking process using helper method
            thinking_data = self._extract_thinking_process(response_content)

            return {
                "step_id": step_id,
                "model_id": "nway_hypothesis_testing",
                "reasoning": response_content,
                "confidence": 0.87,
                "processing_time": 0.0,
                "token_usage": (
                    response.usage.input_tokens + response.usage.output_tokens
                    if hasattr(response, "usage")
                    else 0
                ),
                "nway_pattern": current_nway_pattern,
                "thinking_process": thinking_data["thinking_content"],
                "cleaned_response": thinking_data["cleaned_response"],
            }

        except Exception as e:
            self.logger.error(f"❌ Hypothesis testing application failed: {e}")
            return self._create_error_result(step_id, "nway_hypothesis_testing", str(e))

    async def apply_decision_framework(
        self, problem_statement: str, business_context: Dict[str, Any], step_id: str
    ) -> Dict[str, Any]:
        """Apply structured decision framework for complex business decisions"""

        if not CLAUDE_CLIENT_AVAILABLE:
            return self._create_error_result(
                step_id, "nway_decision_frameworks", "CLAUDE CLIENT NOT AVAILABLE"
            )

        try:
            claude = await get_claude_client()

            # Decision framework prompt with cognitive exhaust capture
            prompt = f"""You are a decision science expert applying systematic decision frameworks.

BUSINESS CONTEXT:
{business_context.get('industry', 'General business') if hasattr(business_context, 'get') else getattr(business_context, 'industry', 'General business')} | {business_context.get('company_size', 'Unknown size') if hasattr(business_context, 'get') else getattr(business_context, 'company_size', 'Unknown size')} | {business_context.get('urgency', 'Standard timeline') if hasattr(business_context, 'get') else getattr(business_context, 'urgency', 'Standard timeline')}

PROBLEM STATEMENT:
{problem_statement}

<thinking>
Let me approach this decision systematically:

1. What decision am I actually trying to make here? What's the core choice or tradeoff?

2. What are the key factors that should influence this decision?
   - Financial considerations (cost, revenue, ROI)?
   - Strategic fit with business objectives?
   - Risk factors and potential downsides?
   - Implementation feasibility and timeline?
   - Stakeholder impact and buy-in?

3. How should I weight these criteria? Which are most critical for this business context?

4. What are all the realistic alternatives I should consider?
   - What's the "do nothing" option and its implications?
   - What creative alternatives might I be missing?
   - Are there hybrid approaches that combine elements?

5. How do I objectively score each alternative against each criterion?
   - What evidence or benchmarks can I use for scoring?
   - Where am I making assumptions versus having data?

6. What are the key sensitivities in my analysis?
   - If I changed the weightings, would the recommendation change?
   - Which scores am I least confident about?
   - What external factors could affect these scores?

7. What could go wrong with my recommended approach, and how can I mitigate those risks?
</thinking>

STRUCTURED DECISION FRAMEWORK:
Apply comprehensive decision analysis:

CRITERIA:
- List 4-6 key decision criteria
- Weight each criterion by importance (must sum to 1.0)
- Explain rationale for each criterion

ALTERNATIVES:
- Generate 3-5 viable alternatives
- Ensure alternatives are mutually exclusive
- Include status quo as one alternative

EVALUATION MATRIX:
- Score each alternative on each criterion (1-10 scale)
- Provide brief rationale for each score
- Calculate weighted scores

SENSITIVITY ANALYSIS:
- Which criteria weights most impact the decision?
- What score changes would alter the recommendation?
- Key uncertainties affecting the analysis

RECOMMENDATION:
- Recommended alternative with rationale
- Implementation considerations
- Risk mitigation strategies
- Success metrics

Focus on analytical rigor and practical business application."""

            response = await claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            response_content = response.content[0].text if response.content else ""

            # Extract thinking process using helper method
            thinking_data = self._extract_thinking_process(response_content)

            # Parse structured decision data if decision capture is available
            decision_result = {
                "step_id": step_id,
                "model_id": "nway_decision_frameworks",
                "reasoning": response_content,
                "confidence": 0.85,
                "processing_time": 0.0,
                "token_usage": (
                    response.usage.input_tokens + response.usage.output_tokens
                    if hasattr(response, "usage")
                    else 0
                ),
                "thinking_process": thinking_data["thinking_content"],
                "cleaned_response": thinking_data["cleaned_response"],
            }

            # Enhanced decision parsing and capture integration
            if DECISION_CAPTURE_AVAILABLE and self.decision_capture:
                parsed_decision = self._parse_decision_response(response_content)
                if parsed_decision:
                    decision_result["structured_decision"] = parsed_decision

                    # Create decision tree in decision capture system
                    decision_tree = DecisionTree(
                        id=f"decision_{step_id}",
                        problem_statement=problem_statement,
                        context=business_context,
                        timestamp=datetime.utcnow(),
                    )

                    await self._integrate_decision_capture(
                        self.decision_capture, decision_tree, parsed_decision, step_id
                    )

            return decision_result

        except Exception as e:
            self.logger.error(f"❌ Decision framework application failed: {e}")
            return self._create_error_result(
                step_id, "nway_decision_frameworks", str(e)
            )

    async def apply_generic_model(
        self,
        model: MentalModelDefinition,
        problem_statement: str,
        business_context: Dict[str, Any],
        step_id: str,
    ) -> Dict[str, Any]:
        """Apply generic mental model using model definition"""

        if not CLAUDE_CLIENT_AVAILABLE:
            return self._create_error_result(
                step_id, model.model_id, "CLAUDE CLIENT NOT AVAILABLE"
            )

        try:
            claude = await get_claude_client()

            # Generic model application prompt
            prompt = f"""You are an expert applying the {model.name} mental model.

MENTAL MODEL: {model.name}
DESCRIPTION: {model.description}

BUSINESS CONTEXT:
{business_context.get('industry', 'General business')} | {business_context.get('company_size', 'Unknown size')}

PROBLEM STATEMENT:
{problem_statement}

Apply this mental model systematically:
1. Explain how this model applies to the problem
2. Work through the model's key concepts and frameworks
3. Generate specific insights using this model's perspective
4. Provide actionable recommendations based on the analysis

Focus on practical business application and actionable insights."""

            response = await claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            response_content = response.content[0].text if response.content else ""

            return {
                "step_id": step_id,
                "model_id": model.model_id,
                "reasoning": response_content,
                "confidence": 0.80,
                "processing_time": 0.0,
                "token_usage": (
                    response.usage.input_tokens + response.usage.output_tokens
                    if hasattr(response, "usage")
                    else 0
                ),
            }

        except Exception as e:
            self.logger.error(
                f"❌ Generic model application failed for {model.model_id}: {e}"
            )
            return self._create_error_result(step_id, model.model_id, str(e))

    def _create_error_result(
        self, step_id: str, model_id: str, error_message: str
    ) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "step_id": step_id,
            "model_id": model_id,
            "reasoning": f"Model application failed: {error_message}",
            "confidence": 0.0,
            "processing_time": 0.0,
            "token_usage": 0,
            "error": error_message,
        }

    def _parse_decision_response(
        self, response_content: str
    ) -> Optional[Dict[str, Any]]:
        """Parse structured decision response for decision capture integration"""
        parsed_data = {"criteria": [], "alternatives": [], "scoring": []}

        try:
            # Extract criteria section
            criteria_match = re.search(
                r"CRITERIA:?\s*(.*?)(?=ALTERNATIVES|$)",
                response_content,
                re.DOTALL | re.IGNORECASE,
            )
            if criteria_match:
                criteria_text = criteria_match.group(1)
                criteria_lines = [
                    line.strip() for line in criteria_text.split("\n") if line.strip()
                ]

                for i, line in enumerate(criteria_lines[:6]):  # Max 6 criteria
                    if ":" in line:
                        name, description = line.split(":", 1)
                        name = name.strip().replace("-", "").replace("*", "").strip()
                        description = description.strip()
                    else:
                        name = line.strip().replace("-", "").replace("*", "").strip()
                        description = f"Criterion: {name}"

                    if name:
                        parsed_data["criteria"].append(
                            {
                                "name": name[:50],
                                "description": description[:200],
                                "weight": 1.0 / min(6, len(criteria_lines)),
                            }
                        )

            # Extract alternatives section
            alternatives_match = re.search(
                r"ALTERNATIVES:?\s*(.*?)(?=WEIGHTS|SCORING|$)",
                response_content,
                re.DOTALL | re.IGNORECASE,
            )
            if alternatives_match:
                alternatives_text = alternatives_match.group(1)
                alt_lines = [
                    line.strip()
                    for line in alternatives_text.split("\n")
                    if line.strip()
                ]

                for i, line in enumerate(alt_lines[:5]):  # Max 5 alternatives
                    if ":" in line:
                        name, description = line.split(":", 1)
                        name = name.strip().replace("-", "").replace("*", "").strip()
                        description = description.strip()
                    else:
                        name = line.strip().replace("-", "").replace("*", "").strip()
                        description = f"Alternative: {name}"

                    if name:
                        parsed_data["alternatives"].append(
                            {"name": name[:50], "description": description[:200]}
                        )

            return (
                parsed_data
                if (parsed_data["criteria"] or parsed_data["alternatives"])
                else None
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to parse decision response: {e}")
            return None

    async def _integrate_decision_capture(
        self,
        decision_capture,
        decision_tree,
        parsed_decision: Dict[str, Any],
        step_id: str,
    ) -> None:
        """Integrate parsed decision data with decision capture system"""
        try:
            # Add criteria to decision tree
            if parsed_decision.get("criteria"):
                criteria_objects = []
                for criterion_data in parsed_decision["criteria"]:
                    criteria_objects.append(
                        DecisionCriteria(
                            name=criterion_data["name"],
                            description=criterion_data["description"],
                            weight=criterion_data["weight"],
                            measurement_type="quantitative",
                            preferred_direction="maximize",
                        )
                    )
                decision_capture.add_decision_criteria(
                    decision_tree.id, criteria_objects
                )

            # Add alternatives to decision tree
            if parsed_decision.get("alternatives"):
                alternative_objects = []
                for alt_data in parsed_decision["alternatives"]:
                    alternative_objects.append(
                        DecisionAlternative(
                            id=f"alt_{len(alternative_objects)+1}",
                            name=alt_data["name"],
                            description=alt_data["description"],
                            data_sources=[f"cognitive_engine_{step_id}"],
                            analysis_methods=["multi_criteria_decision_analysis"],
                        )
                    )
                decision_capture.add_decision_alternatives(
                    decision_tree.id, alternative_objects
                )

            self.logger.info(
                f"✅ Decision capture integration completed for {decision_tree.id}"
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Decision capture integration failed: {e}")
