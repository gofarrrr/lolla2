"""
Prompt Builder - Constructs consultant prompts with context injection.

Responsibilities:
- Build consultant-specific prompts from problem context
- Apply persona templates and specializations
- Inject MECE framework, depth packs, enhanced prompts
- Validate token budgets
- Generate audit fingerprints
"""

import hashlib
from typing import List, Dict, Any, Optional
from .interfaces import PromptBuilder
from .types import PromptSpec


# Approximate token estimation (1 token ≈ 4 characters)
CHARS_PER_TOKEN = 4

# Default model mappings - Grok 4 Fast for strategic depth, DeepSeek as emergency fallback
DEFAULT_MODEL = "grok-4-fast"
MODEL_BY_CONSULTANT_TYPE = {
    "strategic_analyst": "grok-4-fast",
    "financial_analyst": "grok-4-fast",
    "operations_expert": "grok-4-fast",
    "technical_architect": "grok-4-fast",
    "market_researcher": "grok-4-fast",
}

# Persona-specific temperatures for cognitive diversity
PERSONA_TEMPERATURES = {
    "strategic_analyst": 0.8,  # More creative for strategic thinking
    "financial_analyst": 0.5,  # More precise for financial analysis
    "operations_expert": 0.6,  # Balanced for operational planning
    "technical_architect": 0.5,  # Precise for technical decisions
    "market_researcher": 0.7,  # Moderate creativity for market insights
    "default": 0.7,
}


class StandardPromptBuilder(PromptBuilder):
    """
    Standard implementation of PromptBuilder interface.

    Builds prompts with:
    - Consultant persona and specialization
    - Problem context
    - MECE framework (if available)
    - Depth packs (if available)
    - Enhanced prompts (if available)
    - Token budget validation
    """

    def __init__(
        self,
        max_tokens_per_prompt: int = 8000,
        default_temperature: float = 0.7,
    ):
        """
        Initialize PromptBuilder.

        Args:
            max_tokens_per_prompt: Maximum tokens per prompt (including response)
            default_temperature: Default temperature if not persona-specific
        """
        self.max_tokens_per_prompt = max_tokens_per_prompt
        self.default_temperature = default_temperature

    def build(
        self,
        problem_context: str,
        consultant_blueprints: List[Dict[str, Any]],
        framework: Optional[Dict[str, Any]] = None,
        enhanced_prompts: Optional[List[str]] = None,
        depth_packs: Optional[Dict[str, str]] = None,
    ) -> List[PromptSpec]:
        """
        Build prompts for all consultants.

        Args:
            problem_context: User's problem/query
            consultant_blueprints: Selected consultant configurations
            framework: MECE framework from problem structuring (optional)
            enhanced_prompts: Additional prompt enhancements (optional)
            depth_packs: Stage 0 depth packs by consultant_id (optional)

        Returns:
            List of PromptSpec objects ready for execution

        Raises:
            ValueError: If token budget is exceeded
        """
        prompts = []

        for consultant in consultant_blueprints:
            consultant_id = consultant.get("consultant_id", "unknown")
            consultant_type = consultant.get("consultant_type", "general")
            specialization = consultant.get("specialization", "")
            assigned_dimensions = consultant.get("assigned_dimensions", [])

            # Get depth pack if available
            depth_pack = ""
            if depth_packs and consultant_id in depth_packs:
                depth_pack = depth_packs[consultant_id]

            # Build system prompt (persona + role)
            system_prompt = self._build_system_prompt(
                consultant_type=consultant_type,
                specialization=specialization,
                assigned_dimensions=assigned_dimensions,
            )

            # Build user prompt (problem + context + framework + depth)
            user_prompt = self._build_user_prompt(
                problem_context=problem_context,
                framework=framework,
                enhanced_prompts=enhanced_prompts or [],
                depth_pack=depth_pack,
            )

            # Determine model and temperature
            model = MODEL_BY_CONSULTANT_TYPE.get(consultant_type, DEFAULT_MODEL)
            temperature = PERSONA_TEMPERATURES.get(consultant_type, self.default_temperature)

            # Create prompt spec
            prompt_spec = PromptSpec(
                consultant_id=consultant_id,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=5000,  # OPERATION KEYSTONE: Generous budget for McKinsey-depth multi-paragraph analysis
                metadata={
                    "consultant_type": consultant_type,
                    "specialization": specialization,
                    "has_depth_pack": bool(depth_pack),
                    "has_framework": bool(framework),
                },
            )

            # Estimate tokens
            estimated = self.estimate_tokens(prompt_spec)
            prompt_spec.estimated_tokens = estimated

            # Validate token budget
            if estimated > self.max_tokens_per_prompt:
                raise ValueError(
                    f"Prompt for {consultant_id} exceeds token budget: "
                    f"{estimated} > {self.max_tokens_per_prompt}"
                )

            prompts.append(prompt_spec)

        return prompts

    def _build_system_prompt(
        self,
        consultant_type: str,
        specialization: str,
        assigned_dimensions: List[str],
    ) -> str:
        """Build consultant persona system prompt"""

        # Base persona by type
        persona_intros = {
            "strategic_analyst": "You are an expert strategic business consultant specializing in {specialization}. "
            "Your strength is high-level strategic thinking, market analysis, and long-term planning.",
            "financial_analyst": "You are a senior financial analyst with expertise in {specialization}. "
            "You excel at financial modeling, risk assessment, and investment analysis.",
            "operations_expert": "You are an operations excellence consultant specializing in {specialization}. "
            "You focus on process optimization, resource allocation, and operational efficiency.",
            "technical_architect": "You are a technical architecture expert with deep knowledge of {specialization}. "
            "You design scalable systems and evaluate technology decisions.",
            "market_researcher": "You are a market research specialist focusing on {specialization}. "
            "You analyze market dynamics, competitive landscapes, and customer insights.",
        }

        intro = persona_intros.get(
            consultant_type,
            "You are a business consultant with expertise in {specialization}. "
            "You provide strategic analysis and actionable recommendations.",
        )

        system_prompt = intro.format(specialization=specialization or "general business strategy")

        # Add cognitive dimensions if assigned
        if assigned_dimensions:
            dimensions_str = ", ".join(assigned_dimensions)
            system_prompt += f"\n\nYour assigned cognitive dimensions: {dimensions_str}. "
            system_prompt += "Focus your analysis through these lenses."

        # Analysis instructions - OPERATION KEYSTONE: McKinsey-depth standards
        system_prompt += """

Deliver McKinsey-caliber strategic analysis following this structure:

1. **Key Insights** (3-5 insights, each 800-1500 characters):
   Structure per insight:
   - **Lead Statement** (bold): Core insight in one sentence
   - **Strategic Analysis** (2-3 paragraphs, 500-800 words total):
     * WHY this matters: Explain causal mechanisms and strategic implications
     * EVIDENCE: Quantitative data, market dynamics, competitive positioning
     * CONTEXT: Connection to broader strategic themes and interdependencies
     * EXAMPLES: Specific scenarios, analogies, or precedents

   Follow McKinsey's What → So What → Now What progression:
   - WHAT: Present the finding with supporting data
   - SO WHAT: Synthesize implications for strategy
   - NOW WHAT: Connect to actionable recommendations

2. **Risk Factors** (2-3 risks, each 500-800 characters):
   Structure per risk:
   - **Risk Identification**: Specific threat + probability/timeline
   - **Impact Analysis**: Concrete scenarios with quantified impact
   - **Interconnections**: How this risk amplifies/triggers other risks
   - **Mitigation Framework**: 2-3 specific countermeasures with implementation guidance

   Apply "What Would You Have to Believe?" framework to spell out all assumptions.

3. **Opportunities** (2-3 opportunities, each 500-800 characters):
   Structure per opportunity:
   - **Strategic Value**: Describe opportunity and competitive advantage
   - **Market Timing**: Window of opportunity + first-mover advantages
   - **Quantified Upside**: ROI, market share, or strategic positioning gains
   - **Capture Strategy**: 3-5 MECE implementation steps

   Focus on opportunities under client control with measurable outcomes.

4. **Recommendations** (2-3 recommendations, each 800-1200 characters):
   Follow Pyramid Principle architecture:
   - **Governing Thought** (one sentence): Clear, actionable recommendation
   - **Detailed Rationale** (300-500 words):
     * Evidence synthesis from findings
     * Strategic logic and causal reasoning
     * Address potential objections preemptively
   - **Implementation Plan** (MECE breakdown):
     * 3-5 concrete steps with ownership and timeline
     * Success metrics (quantified outcomes)
     * Dependencies and critical path
   - **Risk Mitigation**: Early warning indicators + contingency plans

Quality Standards:
- Demonstrate deep domain expertise with concrete reasoning
- Use McKinsey 80/20 rule: Focus on 20% of factors yielding 80% of value
- Avoid "boiling the ocean" - depth over breadth
- Every claim must have evidence: data, market dynamics, or expert validation
- CITATIONS REQUIRED: Append inline bracketed citations immediately after any factual assertion or quantitative figure, using the form [source: URL] or [source: Publisher, Year]. Responses lacking inline citations will be rejected.
- Connect insights to actionable next steps under client control
- Use MECE principle: ideas must be Mutually Exclusive, Collectively Exhaustive"""

        return system_prompt

    def _build_user_prompt(
        self,
        problem_context: str,
        framework: Optional[Dict[str, Any]],
        enhanced_prompts: List[str],
        depth_pack: str,
    ) -> str:
        """Build user prompt with problem context and enhancements"""

        user_prompt = f"**Problem to Analyze:**\n{problem_context}\n"

        # Add MECE framework if available
        if framework:
            user_prompt += "\n**Structured Problem Framework:**\n"
            user_prompt += self._format_framework(framework)

        # Add depth pack (Stage 0 enrichment) if available
        if depth_pack:
            user_prompt += f"\n**Relevant Mental Models & Context:**\n{depth_pack}\n"

        # Add enhanced prompts if available
        if enhanced_prompts:
            user_prompt += "\n**Additional Context:**\n"
            for i, enhancement in enumerate(enhanced_prompts, 1):
                user_prompt += f"{i}. {enhancement}\n"

        # OPERATION STRUCTURED OUTPUT: Response Prefilling (Manus.im principle)
        # Guide the model to produce deterministic JSON matching our schema
        user_prompt += """

**CRITICAL: JSON Output Format**

You MUST respond with a single, valid JSON object (no markdown, no code blocks, just raw JSON) matching this exact structure:

{
  "consultant_id": "your_consultant_id",
  "key_insights": [
    "First insight as a complete, multi-paragraph analysis (800-1500 characters)...",
    "Second insight with full What/So What/Now What progression (800-1500 characters)...",
    "Third insight (if applicable)..."
  ],
  "risk_factors": [
    "First risk with impact analysis and mitigation (500-800 characters)...",
    "Second risk (if applicable)..."
  ],
  "opportunities": [
    "First opportunity with strategic value and capture plan (500-800 characters)...",
    "Second opportunity (if applicable)..."
  ],
  "recommendations": [
    "First recommendation with detailed rationale and implementation (800-1200 characters)...",
    "Second recommendation (if applicable)..."
  ],
  "confidence_level": "high",
  "analysis_quality": "excellent"
}

Each string field must be a COMPLETE, MULTI-PARAGRAPH analysis (NOT bullet points).
Confidence level: "high" | "medium" | "low"
Analysis quality: "excellent" | "good" | "adequate" | "poor"

Provide your McKinsey-caliber analysis now as valid JSON.

CRITICAL CITATION ENFORCEMENT:
- For each factual statement and every quantitative figure in any field above, include an inline bracketed citation in the text immediately following the claim (e.g., "... global CAGR is 12% [source: https://example.com/market-report-2024]").
- If a URL is unavailable, cite the publisher and year (e.g., [source: McKinsey, 2024]).
- Do not include markdown code fences; produce raw JSON with citation text embedded within the strings.
"""

        return user_prompt

    def _format_framework(self, framework: Dict[str, Any]) -> str:
        """Format MECE framework for prompt injection"""
        parts = []

        # Framework name and description
        name = framework.get("name", "Problem Structure")
        description = framework.get("description", "")
        if description:
            parts.append(f"**{name}:** {description}")

        # Dimensions
        dimensions = framework.get("dimensions", [])
        if dimensions:
            parts.append("\n**Key Dimensions:**")
            for dim in dimensions:
                dim_name = dim.get("dimension", "")
                considerations = dim.get("considerations", [])
                priority = dim.get("priority", 2)
                priority_label = {1: "🔴 High", 2: "🟡 Medium", 3: "🟢 Low"}.get(priority, "")

                parts.append(f"- {dim_name} {priority_label}")
                if considerations:
                    for consideration in considerations:
                        parts.append(f"  • {consideration}")

        # Core assumptions
        assumptions = framework.get("core_assumptions", [])
        if assumptions:
            parts.append("\n**Core Assumptions:**")
            for assumption in assumptions:
                parts.append(f"- {assumption}")

        # Critical constraints
        constraints = framework.get("critical_constraints", [])
        if constraints:
            parts.append("\n**Critical Constraints:**")
            for constraint in constraints:
                parts.append(f"- {constraint}")

        # Success criteria
        success_criteria = framework.get("success_criteria", [])
        if success_criteria:
            parts.append("\n**Success Criteria:**")
            for criterion in success_criteria:
                parts.append(f"- {criterion}")

        return "\n".join(parts)

    def estimate_tokens(self, prompt_spec: PromptSpec) -> int:
        """
        Estimate total tokens for a prompt.

        Args:
            prompt_spec: Prompt specification

        Returns:
            Estimated token count (system + user + expected response)
        """
        system_chars = len(prompt_spec.system_prompt)
        user_chars = len(prompt_spec.user_prompt)
        response_chars = (prompt_spec.max_tokens or 2000) * CHARS_PER_TOKEN

        total_chars = system_chars + user_chars + response_chars
        return total_chars // CHARS_PER_TOKEN

    def generate_fingerprint(self, prompt_spec: PromptSpec) -> str:
        """
        Generate audit fingerprint for prompt.

        Args:
            prompt_spec: Prompt specification

        Returns:
            SHA-256 hash of prompt content for traceability
        """
        # Combine system + user prompts + metadata for fingerprinting
        content = (
            f"{prompt_spec.system_prompt}|{prompt_spec.user_prompt}|"
            f"{prompt_spec.model}|{prompt_spec.temperature}"
        )

        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(content.encode("utf-8"))
        return hash_obj.hexdigest()[:16]  # First 16 chars for readability
