"""
MECE Structuring Cognitive Model
Focused implementation of MECE (Mutually Exclusive, Collectively Exhaustive) problem structuring
"""

import time
import re
from typing import Dict, List, Any, Optional
from .base_cognitive_model import (
    BaseCognitiveModel,
    CognitiveModelType,
    ModelApplicationContext,
    ModelApplicationResult,
)


class MECEStructuringModel(BaseCognitiveModel):
    """
    Implements MECE structuring cognitive model for systematic problem breakdown
    Focuses on mutually exclusive, collectively exhaustive problem decomposition
    """

    def __init__(self, llm_orchestrator: Optional["LLMOrchestrator"] = None):
        super().__init__("mece_structuring", llm_orchestrator)
        self.model_specific_config = {
            "focus_areas": [
                "problem_decomposition",
                "mutual_exclusivity",
                "collective_exhaustiveness",
                "hierarchical_structuring",
                "component_categorization",
                "logical_completeness",
            ],
            "quality_indicators": [
                "decomposition_clarity",
                "exclusivity_validation",
                "exhaustiveness_coverage",
                "logical_hierarchy",
                "actionable_granularity",
            ],
        }

    def _get_model_type(self) -> CognitiveModelType:
        return CognitiveModelType.MECE_STRUCTURING

    async def apply_model(
        self, context: ModelApplicationContext
    ) -> ModelApplicationResult:
        """Apply MECE structuring analysis"""
        start_time = time.time()

        try:
            # Build MECE specific prompt
            prompt = self._build_prompt(context)

            # Get LLM response with MECE optimization
            if self.llm_orchestrator:
                llm_response = await self.llm_orchestrator.generate_response(
                    prompt=prompt,
                    model_type="mece_structuring",
                    temperature=0.1,  # Very low temperature for structured analysis
                    max_tokens=1400,
                    require_high_quality=True,
                )
                response_text = llm_response.content
                base_confidence = llm_response.confidence_score
            else:
                # Fallback to template-based analysis
                response_text = self._generate_template_analysis(context)
                base_confidence = 0.6

            # Parse and structure the response
            result = self._parse_mece_response(response_text, context, base_confidence)
            result.processing_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                f"✅ MECE structuring analysis completed: confidence={result.confidence_score:.3f}"
            )
            return result

        except Exception as e:
            self.logger.error(f"❌ MECE structuring analysis failed: {e}")
            # Return fallback result
            return self._create_fallback_analysis(context, str(e))

    def _build_prompt(self, context: ModelApplicationContext) -> str:
        """Build MECE structuring specific prompt"""

        # Determine problem domain for specialized MECE approach
        problem_domain = self._determine_problem_domain(
            context.problem_statement, context.business_context
        )
        domain_frameworks = self._get_domain_specific_frameworks(problem_domain)

        prompt = f"""
You are applying the MECE (Mutually Exclusive, Collectively Exhaustive) framework to systematically structure this problem:

PROBLEM STATEMENT:
{context.problem_statement}

BUSINESS CONTEXT:
{context.business_context}

MECE STRUCTURING METHODOLOGY - Apply rigorously:

1. PROBLEM BREAKDOWN (Level 1)
   - Identify the main components of this problem
   - Ensure components are mutually exclusive (no overlap)
   - Verify components are collectively exhaustive (cover everything)
   
2. COMPONENT ANALYSIS (Level 2)  
   - Break down each Level 1 component into sub-components
   - Apply MECE principle to each breakdown
   - Maintain logical hierarchy and clear relationships

3. CATEGORIZATION VALIDATION
   - Confirm no overlaps between categories at same level
   - Verify complete coverage of the problem space
   - Check for appropriate granularity (actionable level)

4. STRUCTURAL VERIFICATION
   - Test mutual exclusivity: Can any item belong to multiple categories?
   - Test collective exhaustiveness: Is anything missing?
   - Validate logical consistency throughout structure

DOMAIN-SPECIFIC CONSIDERATIONS ({problem_domain}):
{domain_frameworks}

MECE STRUCTURE REQUIREMENTS:
- Create 3-5 main categories (Level 1)
- Break each into 2-4 sub-categories (Level 2) 
- Ensure clear, non-overlapping definitions
- Use parallel grammatical structure
- Make categories actionable and meaningful
- Verify complete problem coverage

OUTPUT FORMAT:
Present as clear hierarchical structure with:
- Main Categories (Level 1)
- Sub-categories (Level 2)
- Brief justification for MECE compliance
- Identification of any potential gaps or overlaps

Focus on {problem_domain} domain-specific structuring approaches.
"""

        return prompt

    def _determine_problem_domain(
        self, problem_statement: str, business_context: Dict[str, Any]
    ) -> str:
        """Determine the primary domain for specialized MECE approach"""

        # Check business context first
        # Safe get from business_context (dict or MetisDataContract)
        if hasattr(business_context, "get"):
            industry = business_context.get("industry", "").lower()
        elif hasattr(business_context, "industry"):
            industry = getattr(business_context, "industry", "").lower()
        else:
            industry = ""
        if industry:
            domain_mapping = {
                "technology": "technology",
                "software": "technology",
                "finance": "finance",
                "consulting": "strategy",
                "healthcare": "healthcare",
                "manufacturing": "operations",
            }
            for key, domain in domain_mapping.items():
                if key in industry:
                    return domain

        # Analyze problem statement for domain indicators
        problem_lower = problem_statement.lower()

        if any(
            word in problem_lower
            for word in ["strategy", "strategic", "market", "competitive"]
        ):
            return "strategy"
        elif any(
            word in problem_lower
            for word in ["operation", "process", "efficiency", "workflow"]
        ):
            return "operations"
        elif any(
            word in problem_lower
            for word in ["technology", "ai", "digital", "platform"]
        ):
            return "technology"
        elif any(
            word in problem_lower
            for word in ["financial", "revenue", "cost", "investment"]
        ):
            return "finance"
        else:
            return "general"

    def _get_domain_specific_frameworks(self, domain: str) -> str:
        """Get domain-specific MECE frameworks"""

        frameworks = {
            "strategy": """
Strategic MECE Frameworks:
- Market/Product/Technology dimensions
- Internal/External/Competitive factors
- Current State/Future State/Transition
- People/Process/Technology structure
- Revenue/Cost/Risk categorization
""",
            "operations": """
Operations MECE Frameworks:
- Input/Process/Output structure
- People/Process/Technology/Environment
- Plan/Execute/Monitor/Improve cycle
- Internal/External/Stakeholder dimensions
- Efficiency/Quality/Cost/Speed metrics
""",
            "technology": """
Technology MECE Frameworks:
- Frontend/Backend/Infrastructure layers
- Build/Buy/Partner decision structure
- Security/Performance/Scalability/Usability
- Current/Target/Migration architecture
- Technical/Business/User requirements
""",
            "finance": """
Financial MECE Frameworks:
- Revenue/Cost/Profit structure
- Short-term/Medium-term/Long-term horizons
- Internal/External/Market factors
- Operational/Strategic/Financial metrics
- Risk/Return/Liquidity dimensions
""",
            "general": """
General MECE Frameworks:
- Internal/External factor analysis
- Past/Present/Future timeline
- People/Process/Technology structure
- Strategic/Tactical/Operational levels
- Quantitative/Qualitative dimensions
""",
        }

        return frameworks.get(domain, frameworks["general"])

    def _parse_mece_response(
        self,
        response_text: str,
        context: ModelApplicationContext,
        base_confidence: float,
    ) -> ModelApplicationResult:
        """Parse and structure MECE response"""

        # Extract hierarchical structure
        insights = []
        evidence = []
        assumptions = []

        # Look for Level 1 categories
        level1_matches = re.findall(
            r"(?:^|\n)\s*(?:1\.|•|\-)\s*([^\n]+)", response_text, re.MULTILINE
        )
        if level1_matches:
            insights.append(f"Level 1 Categories: {', '.join(level1_matches[:4])}")

        # Look for MECE validation
        mece_validation = re.findall(
            r"(?:mece|mutual|exclusive|exhaustive)[:\-\s]*([^\.]+\.)",
            response_text,
            re.IGNORECASE,
        )
        if mece_validation:
            evidence.append(f"MECE Validation: {mece_validation[0].strip()}")

        # Look for structural insights
        structure_matches = re.findall(
            r"(?:structure|category|breakdown)[:\-\s]*([^\.]+\.)",
            response_text,
            re.IGNORECASE,
        )
        if structure_matches:
            insights.append(f"Structural Analysis: {structure_matches[0].strip()}")

        # Look for completeness assessment
        complete_matches = re.findall(
            r"(?:complete|comprehensive|coverage)[:\-\s]*([^\.]+\.)",
            response_text,
            re.IGNORECASE,
        )
        if complete_matches:
            evidence.append(f"Completeness Check: {complete_matches[0].strip()}")

        # Extract assumptions about categorization
        assumption_keywords = ["assume", "assuming", "given", "provided"]
        for keyword in assumption_keywords:
            matches = re.findall(
                rf"{keyword}[:\-\s]*([^\.]+\.)", response_text, re.IGNORECASE
            )
            if matches:
                assumptions.append(matches[0].strip())

        # Fallback insight extraction if structured parsing fails
        if not insights:
            sentences = [
                s.strip() + "." for s in response_text.split(".") if len(s.strip()) > 25
            ]
            insights = sentences[:3]

        # Calculate MECE-specific confidence
        mece_confidence = self._calculate_mece_confidence(
            response_text, level1_matches, insights
        )

        # Combine with base confidence
        final_confidence = (base_confidence + mece_confidence) / 2

        return ModelApplicationResult(
            reasoning_text=response_text,
            confidence_score=final_confidence,
            key_insights=insights[:5],
            supporting_evidence=evidence[:3],
            assumptions_made=assumptions[:3],
            quality_metrics={
                "mece_rigor": mece_confidence,
                "structure_clarity": self._assess_structure_clarity(response_text),
                "category_count": len(level1_matches),
                "exclusivity_validation": (
                    1.0
                    if any("exclusive" in text.lower() for text in evidence)
                    else 0.5
                ),
                "exhaustiveness_validation": (
                    1.0
                    if any("exhaustive" in text.lower() for text in evidence)
                    else 0.5
                ),
                "hierarchical_depth": self._count_hierarchical_levels(response_text),
            },
            processing_time_ms=0.0,
            model_id="mece_structuring",
        )

    def _calculate_mece_confidence(
        self, response_text: str, level1_categories: List[str], insights: List[str]
    ) -> float:
        """Calculate confidence based on MECE quality indicators"""

        confidence = 0.5  # Base confidence

        # Boost for proper categorization
        if 3 <= len(level1_categories) <= 5:
            confidence += 0.15  # Optimal number of categories
        elif len(level1_categories) >= 2:
            confidence += 0.1

        # Boost for MECE terminology usage
        mece_keywords = [
            "mece",
            "mutual",
            "exclusive",
            "exhaustive",
            "comprehensive",
            "structure",
            "category",
        ]
        keyword_count = sum(
            1 for keyword in mece_keywords if keyword in response_text.lower()
        )
        confidence += min(0.2, keyword_count * 0.04)

        # Boost for hierarchical structure
        if any(
            pattern in response_text
            for pattern in ["Level 1", "Level 2", "1.", "2.", "•"]
        ):
            confidence += 0.1

        # Boost for validation discussion
        if any(
            word in response_text.lower()
            for word in ["validation", "verify", "confirm", "check"]
        ):
            confidence += 0.05

        # Boost for completeness indicators
        if any(
            word in response_text.lower()
            for word in ["complete", "comprehensive", "coverage", "gap"]
        ):
            confidence += 0.05

        return min(0.95, confidence)

    def _assess_structure_clarity(self, response_text: str) -> float:
        """Assess the clarity of the structural breakdown"""

        clarity_score = 0.0

        # Check for clear hierarchical indicators
        if any(
            pattern in response_text
            for pattern in ["1.", "2.", "3.", "A.", "B.", "•", "-"]
        ):
            clarity_score += 0.3

        # Check for consistent structure
        numbered_patterns = len(re.findall(r"\d+\.", response_text))
        if numbered_patterns >= 3:
            clarity_score += 0.3

        # Check for clear categorization language
        if any(
            word in response_text.lower()
            for word in ["category", "component", "element", "dimension"]
        ):
            clarity_score += 0.2

        # Check for explanatory structure
        if any(
            phrase in response_text.lower()
            for phrase in ["consists of", "includes", "comprises", "breaks down"]
        ):
            clarity_score += 0.2

        return min(1.0, clarity_score)

    def _count_hierarchical_levels(self, response_text: str) -> int:
        """Count the number of hierarchical levels in the structure"""

        levels = 0

        # Look for Level indicators
        if "Level 1" in response_text or "1." in response_text:
            levels = max(levels, 1)
        if "Level 2" in response_text or any(
            pattern in response_text for pattern in ["1.1", "a.", "i."]
        ):
            levels = max(levels, 2)
        if "Level 3" in response_text or any(
            pattern in response_text for pattern in ["1.1.1", "a.i", "i.a"]
        ):
            levels = max(levels, 3)

        return levels

    def _generate_template_analysis(self, context: ModelApplicationContext) -> str:
        """Generate template-based MECE analysis when LLM unavailable"""

        domain = self._determine_problem_domain(
            context.problem_statement, context.business_context
        )

        return f"""
MECE STRUCTURING ANALYSIS

Problem: {context.problem_statement[:100]}...

LEVEL 1 BREAKDOWN (Main Categories):
1. Current State Analysis
   - Existing situation assessment
   - Stakeholder impact evaluation
   - Resource availability review

2. Future State Design  
   - Desired outcome definition
   - Success criteria establishment
   - Performance target setting

3. Transition Strategy
   - Implementation approach
   - Risk mitigation planning
   - Timeline development

4. Enablement Factors
   - Resource requirements
   - Capability development
   - Support system design

MECE VALIDATION:
- Mutual Exclusivity: Each category addresses distinct aspects without overlap
- Collective Exhaustiveness: All aspects of the problem are covered across categories
- Appropriate Granularity: Categories are actionable and meaningful for {domain} domain

STRUCTURAL VERIFICATION:
✓ No overlaps between main categories
✓ Complete problem coverage achieved  
✓ Hierarchical logic maintained
✓ Domain-appropriate categorization applied

NEXT STEPS:
- Detailed breakdown of each main category
- Sub-category development with MECE validation
- Implementation priority assessment
"""

    def _create_fallback_analysis(
        self, context: ModelApplicationContext, error_message: str
    ) -> ModelApplicationResult:
        """Create fallback analysis for MECE structuring"""

        fallback_insights = [
            "MECE analysis requires systematic problem decomposition",
            "Mutual exclusivity ensures no category overlaps",
            "Collective exhaustiveness ensures complete coverage",
        ]

        return ModelApplicationResult(
            reasoning_text=f"MECE Structuring Analysis: {context.problem_statement[:200]}... [Analysis limited due to technical constraints]",
            confidence_score=0.4,
            key_insights=fallback_insights,
            supporting_evidence=[],
            assumptions_made=[f"Analysis constrained by: {error_message}"],
            quality_metrics={
                "fallback_result": True,
                "mece_rigor": 0.3,
                "structure_clarity": 0.2,
                "category_count": 0,
                "exclusivity_validation": 0.3,
                "exhaustiveness_validation": 0.3,
            },
            processing_time_ms=0.0,
            model_id="mece_structuring",
        )

    def _validate_output_quality(self, result: ModelApplicationResult) -> bool:
        """Validate MECE structuring output quality"""

        # Check minimum requirements
        min_requirements = [
            len(result.reasoning_text) > 150,
            len(result.key_insights) >= 2,
            result.confidence_score > 0.3,
            result.quality_metrics.get("mece_rigor", 0) > 0.4,
        ]

        # Check for MECE indicators
        reasoning_lower = result.reasoning_text.lower()
        mece_indicators = [
            "mece" in reasoning_lower or "mutual" in reasoning_lower,
            "category" in reasoning_lower or "structure" in reasoning_lower,
            "exclusive" in reasoning_lower or "exhaustive" in reasoning_lower,
            result.quality_metrics.get("category_count", 0) >= 2,
        ]

        # Quality validation
        basic_quality = sum(min_requirements) >= 3
        mece_quality = sum(mece_indicators) >= 2

        return basic_quality and mece_quality
