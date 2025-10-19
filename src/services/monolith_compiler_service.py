"""
Lolla Proving Ground - Monolith Compiler Service
Service for compiling multi-station pipeline prompts into monolithic challengers

This service performs the core "DNA extraction" and "monolith weaving" operations
that transform our distributed 8-station cognitive architecture into a single,
comprehensive monolithic prompt for fair comparison testing.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StationDNA:
    """Represents the cognitive DNA extracted from a single station"""

    station_id: str
    station_name: str
    prompt_template: str
    persona: str
    protocols: List[str]
    key_instructions: str
    output_format: str
    cognitive_signature: str  # Unique fingerprint of the station's behavior


@dataclass
class PipelineExecutionResult:
    """Result of a pipeline dry run execution"""

    station_results: Dict[str, Dict[str, Any]]
    execution_metadata: Dict[str, Any]
    cognitive_dna: Dict[str, StationDNA]
    final_output: str


class MonolithCompilerService:
    """
    Service for compiling multi-station pipeline prompts into monolithic challengers
    """

    def __init__(self):
        self.station_mappings = {
            "STATION_1": "QUICKTHINK",
            "STATION_2": "DEEPTHINK",
            "STATION_3": "BLUETHINK",
            "STATION_4": "REDTHINK",
            "STATION_5": "GREYTHINK",
            "STATION_6": "ULTRATHINK",
            "STATION_7": "DIVERGENTTHINK",
            "STATION_8": "CONVERGENTTHINK",
        }

        # Monolith template structure
        self.monolith_template = """# COMPREHENSIVE ANALYSIS SYSTEM - COMPILED MONOLITH

## GLOBAL CONTEXT
{global_context}

## UNIFIED PERSONA
You are an advanced analytical system that combines the cognitive capabilities of multiple specialized thinking stations. Your analysis incorporates:
{combined_personas}

## INTEGRATED PROTOCOLS
Follow these unified protocols throughout your analysis:
{unified_protocols}

## SEQUENTIAL ANALYSIS TASKS

{sequential_tasks}

## OUTPUT REQUIREMENTS
{output_specifications}

## QUALITY STANDARDS
{quality_criteria}

## ANALYSIS INPUT
Now, analyze the following input comprehensively using all the above cognitive frameworks:

{input_data}

---
IMPORTANT: Execute all eight thinking modes in sequence, building upon each previous analysis to create a comprehensive, multi-dimensional assessment."""

    async def compile_perfect_monolith(
        self,
        golden_case_id: str,
        save_to_db: bool = True,
        custom_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main compilation method

        Steps:
        1. Execute dry run of 8-station pipeline
        2. Extract cognitive DNA from each station
        3. Weave into monolithic prompt
        4. Save to challenger_prompts table (if requested)

        Args:
            golden_case_id: Golden case to use for compilation
            save_to_db: Whether to save result to database
            custom_name: Optional custom name for the compiled prompt

        Returns:
            Dictionary containing compiled prompt and metadata
        """
        try:
            logger.info(
                f"Starting monolith compilation for golden case: {golden_case_id}"
            )

            # Step 1: Execute dry run of pipeline
            logger.info("Executing pipeline dry run...")
            pipeline_result = await self._execute_dry_run(golden_case_id)

            # Step 2: Extract cognitive DNA
            logger.info("Extracting cognitive DNA from stations...")
            cognitive_dna = self._extract_cognitive_dna(pipeline_result.station_results)

            # Step 3: Weave monolithic prompt
            logger.info("Weaving monolithic prompt...")
            monolithic_prompt = self._weave_monolithic_prompt(
                cognitive_dna, pipeline_result
            )

            # Step 4: Generate metadata
            compilation_metadata = {
                "compilation_type": "automated",
                "source_golden_case": golden_case_id,
                "compiled_at": datetime.utcnow().isoformat(),
                "stations_integrated": list(self.station_mappings.keys()),
                "cognitive_signatures": {
                    k: v.cognitive_signature for k, v in cognitive_dna.items()
                },
                "compilation_stats": {
                    "total_stations": len(cognitive_dna),
                    "total_protocols": sum(
                        len(dna.protocols) for dna in cognitive_dna.values()
                    ),
                    "prompt_length": len(monolithic_prompt),
                    "complexity_score": self._calculate_complexity_score(cognitive_dna),
                },
            }

            result = {
                "prompt_name": custom_name
                or f"Perfect Monolith - {golden_case_id[:8]}",
                "prompt_text": monolithic_prompt,
                "version": "1.0",
                "target_station": "FULL_PIPELINE",
                "golden_case_id": golden_case_id,
                "compilation_metadata": compilation_metadata,
                "cognitive_dna": {
                    k: self._serialize_dna(v) for k, v in cognitive_dna.items()
                },
            }

            logger.info(
                f"Monolith compilation completed - Length: {len(monolithic_prompt)} chars"
            )
            return result

        except Exception as e:
            logger.error(f"Error in monolith compilation: {e}")
            raise Exception(f"Failed to compile monolith: {str(e)}")

    async def _execute_dry_run(self, golden_case_id: str) -> PipelineExecutionResult:
        """
        Execute pipeline without side effects to capture all prompts

        This method simulates the execution of the 8-station pipeline,
        capturing the prompts, personas, and protocols used by each station.
        """
        logger.info("Simulating 8-station pipeline execution...")

        # This is a simulation - in real implementation, this would interface with actual pipeline
        mock_station_results = {}

        # Simulate each station's execution
        for station_id, station_name in self.station_mappings.items():
            station_result = await self._simulate_station_execution(
                station_id, station_name, golden_case_id
            )
            mock_station_results[station_id] = station_result

        # Extract cognitive DNA from simulated results
        cognitive_dna = {}
        for station_id, result in mock_station_results.items():
            dna = self._create_station_dna(station_id, result)
            cognitive_dna[station_id] = dna

        return PipelineExecutionResult(
            station_results=mock_station_results,
            execution_metadata={
                "execution_type": "dry_run",
                "golden_case_id": golden_case_id,
                "executed_at": datetime.utcnow().isoformat(),
                "total_stations": len(self.station_mappings),
            },
            cognitive_dna=cognitive_dna,
            final_output="Simulated final pipeline output",
        )

    async def _simulate_station_execution(
        self, station_id: str, station_name: str, golden_case_id: str
    ) -> Dict[str, Any]:
        """Simulate individual station execution to extract its cognitive signature"""

        # Station-specific prompt templates and behaviors
        station_templates = {
            "STATION_1": {
                "prompt": "You are QUICKTHINK, a rapid analysis agent. Perform initial assessment of the input, identifying key themes, immediate insights, and obvious patterns. Focus on speed and breadth over depth.",
                "persona": "Rapid analyst focused on identifying immediate patterns and key themes",
                "protocols": [
                    "Speed over perfection",
                    "Broad pattern recognition",
                    "Initial hypothesis formation",
                ],
                "cognitive_signature": "rapid_pattern_recognition",
            },
            "STATION_2": {
                "prompt": "You are DEEPTHINK, a comprehensive analysis agent. Perform thorough, detailed analysis of the input, exploring nuances, complex relationships, and deeper implications.",
                "persona": "Deep analytical thinker focused on comprehensive exploration",
                "protocols": [
                    "Thorough investigation",
                    "Complex relationship mapping",
                    "Nuanced understanding",
                ],
                "cognitive_signature": "comprehensive_analysis",
            },
            "STATION_3": {
                "prompt": "You are BLUETHINK, a conservative analysis agent. Apply conservative perspective to the analysis, identifying risks, potential downsides, and cautious approaches.",
                "persona": "Conservative analyst focused on risk assessment and cautious evaluation",
                "protocols": [
                    "Risk identification",
                    "Conservative estimation",
                    "Cautious recommendations",
                ],
                "cognitive_signature": "conservative_risk_assessment",
            },
            "STATION_4": {
                "prompt": "You are REDTHINK, a bold innovation agent. Think boldly and innovatively about the input, considering revolutionary approaches, disruptive solutions, and ambitious possibilities.",
                "persona": "Bold innovator focused on revolutionary thinking and ambitious solutions",
                "protocols": [
                    "Bold hypothesis generation",
                    "Disruptive thinking",
                    "Ambitious goal setting",
                ],
                "cognitive_signature": "bold_innovation",
            },
            "STATION_5": {
                "prompt": "You are GREYTHINK, a reality check agent. Apply practical reality checks and feasibility assessments to the analysis, grounding insights in practical constraints.",
                "persona": "Pragmatic evaluator focused on feasibility and practical constraints",
                "protocols": [
                    "Feasibility assessment",
                    "Practical constraint evaluation",
                    "Grounded recommendations",
                ],
                "cognitive_signature": "practical_feasibility",
            },
            "STATION_6": {
                "prompt": "You are ULTRATHINK, a deep synthesis agent. Synthesize all previous analyses into coherent, integrated insights with deep understanding of connections.",
                "persona": "Synthesis specialist focused on integration and coherent insight generation",
                "protocols": [
                    "Multi-perspective integration",
                    "Coherent synthesis",
                    "Deep connection mapping",
                ],
                "cognitive_signature": "deep_synthesis",
            },
            "STATION_7": {
                "prompt": "You are DIVERGENTTHINK, an alternative perspective agent. Explore alternative viewpoints, contrarian positions, and unconventional approaches to the analysis.",
                "persona": "Contrarian thinker focused on alternative perspectives and unconventional approaches",
                "protocols": [
                    "Contrarian analysis",
                    "Alternative viewpoint exploration",
                    "Unconventional approach generation",
                ],
                "cognitive_signature": "divergent_perspectives",
            },
            "STATION_8": {
                "prompt": "You are CONVERGENTTHINK, a final integration agent. Integrate all perspectives into final actionable recommendations with clear priorities and next steps.",
                "persona": "Integration specialist focused on actionable synthesis and clear recommendations",
                "protocols": [
                    "Multi-perspective convergence",
                    "Actionable recommendation generation",
                    "Clear prioritization",
                ],
                "cognitive_signature": "convergent_integration",
            },
        }

        template = station_templates.get(station_id, station_templates["STATION_1"])

        return {
            "station_id": station_id,
            "station_name": station_name,
            "input": f"Analysis input for {golden_case_id}",
            "prompt": template["prompt"],
            "persona": template["persona"],
            "protocols": template["protocols"],
            "output": f"Simulated output from {station_name} analysis...",
            "cognitive_signature": template["cognitive_signature"],
            "execution_time": 2.5,  # Mock execution time
            "metadata": {
                "golden_case_id": golden_case_id,
                "executed_at": datetime.utcnow().isoformat(),
            },
        }

    def _extract_cognitive_dna(
        self, station_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, StationDNA]:
        """
        Extract cognitive DNA from station results

        Parse and structure the essential elements from each station
        """
        cognitive_dna = {}

        for station_id, result in station_results.items():
            dna = self._create_station_dna(station_id, result)
            cognitive_dna[station_id] = dna

        return cognitive_dna

    def _create_station_dna(
        self, station_id: str, result: Dict[str, Any]
    ) -> StationDNA:
        """Create StationDNA object from station result"""
        return StationDNA(
            station_id=station_id,
            station_name=self.station_mappings.get(station_id, station_id),
            prompt_template=result.get("prompt", ""),
            persona=result.get("persona", ""),
            protocols=result.get("protocols", []),
            key_instructions=self._extract_key_instructions(result.get("prompt", "")),
            output_format=self._extract_output_format(result.get("prompt", "")),
            cognitive_signature=result.get("cognitive_signature", ""),
        )

    def _extract_key_instructions(self, prompt: str) -> str:
        """Extract key instructions from a prompt"""
        # Simple extraction - in real implementation, this would be more sophisticated
        lines = prompt.split("\n")
        instructions = []
        for line in lines:
            if any(
                keyword in line.lower()
                for keyword in ["perform", "analyze", "identify", "explore", "apply"]
            ):
                instructions.append(line.strip())
        return ". ".join(instructions[:3])  # Top 3 instructions

    def _extract_output_format(self, prompt: str) -> str:
        """Extract output format requirements from prompt"""
        if "format" in prompt.lower():
            return "Structured analytical output"
        return "Standard analytical format"

    def _weave_monolithic_prompt(
        self,
        cognitive_dna: Dict[str, StationDNA],
        pipeline_result: PipelineExecutionResult,
    ) -> str:
        """
        Combine all station DNA into single coherent prompt

        This is the core intelligence of the compiler - it creates a monolithic
        prompt that captures the essence of all 8 stations in a single execution context.
        """

        # Step 1: Generate global context
        global_context = self._generate_global_context(pipeline_result)

        # Step 2: Merge personas without redundancy
        combined_personas = self._merge_personas(cognitive_dna)

        # Step 3: Consolidate protocols, removing duplicates
        unified_protocols = self._merge_protocols(cognitive_dna)

        # Step 4: Create sequential task instructions
        sequential_tasks = self._format_sequential_tasks(cognitive_dna)

        # Step 5: Generate output specifications
        output_specifications = self._generate_output_specs(cognitive_dna)

        # Step 6: Define quality criteria
        quality_criteria = self._generate_quality_criteria(cognitive_dna)

        # Step 7: Assemble the monolithic prompt
        monolithic_prompt = self.monolith_template.format(
            global_context=global_context,
            combined_personas=combined_personas,
            unified_protocols=unified_protocols,
            sequential_tasks=sequential_tasks,
            output_specifications=output_specifications,
            quality_criteria=quality_criteria,
            input_data="{input_data}",  # Placeholder for actual input
        )

        return monolithic_prompt

    def _generate_global_context(self, pipeline_result: PipelineExecutionResult) -> str:
        """Generate global context section"""
        return f"""This comprehensive analysis system integrates {len(pipeline_result.cognitive_dna)} specialized cognitive stations into a unified analytical framework. Each station contributes unique perspectives and methodologies to create a multi-dimensional analysis that captures both broad patterns and deep insights.

The system combines rapid pattern recognition, comprehensive exploration, risk assessment, innovative thinking, feasibility evaluation, synthesis capabilities, alternative perspectives, and final integration into a coherent analytical process."""

    def _merge_personas(self, cognitive_dna: Dict[str, StationDNA]) -> str:
        """Merge personas without redundancy"""
        personas = []
        for station_id, dna in cognitive_dna.items():
            station_name = dna.station_name
            personas.append(f"• **{station_name}**: {dna.persona}")

        return "\n".join(personas)

    def _merge_protocols(self, cognitive_dna: Dict[str, StationDNA]) -> str:
        """Consolidate protocols, removing duplicates"""
        all_protocols = []
        seen_protocols = set()

        for dna in cognitive_dna.values():
            for protocol in dna.protocols:
                if protocol.lower() not in seen_protocols:
                    all_protocols.append(f"• {protocol}")
                    seen_protocols.add(protocol.lower())

        return "\n".join(all_protocols)

    def _format_sequential_tasks(self, cognitive_dna: Dict[str, StationDNA]) -> str:
        """Create sequential task instructions"""
        tasks = []

        for i, (station_id, dna) in enumerate(cognitive_dna.items(), 1):
            task_section = f"""### TASK {i}: {dna.station_name} - {dna.station_name.title()} Analysis
{dna.key_instructions}

**Cognitive Focus**: {dna.cognitive_signature.replace('_', ' ').title()}
**Expected Output**: {dna.output_format}"""

            tasks.append(task_section)

        return "\n\n".join(tasks)

    def _generate_output_specs(self, cognitive_dna: Dict[str, StationDNA]) -> str:
        """Generate output format specifications"""
        return """Provide a comprehensive analysis structured as follows:

1. **Executive Summary**: Key findings and recommendations
2. **Multi-Station Analysis**: Results from each cognitive station
3. **Synthesis**: Integration of all perspectives  
4. **Recommendations**: Prioritized, actionable next steps
5. **Risk Assessment**: Key risks and mitigation strategies
6. **Innovation Opportunities**: Bold possibilities and breakthrough potential

Ensure each section builds upon the previous analyses while maintaining clarity and actionability."""

    def _generate_quality_criteria(self, cognitive_dna: Dict[str, StationDNA]) -> str:
        """Generate quality standards"""
        return """Your analysis must meet these quality standards:

• **Comprehensiveness**: Address all aspects identified by each cognitive station
• **Coherence**: Maintain logical flow between different analytical perspectives  
• **Actionability**: Provide clear, implementable recommendations
• **Balance**: Integrate conservative and bold perspectives appropriately
• **Depth**: Demonstrate deep understanding of complex relationships
• **Clarity**: Present insights in accessible, well-structured format"""

    def _calculate_complexity_score(
        self, cognitive_dna: Dict[str, StationDNA]
    ) -> float:
        """Calculate complexity score for compiled prompt"""
        base_score = len(cognitive_dna) * 10  # Base score for number of stations

        # Add complexity for protocols
        protocol_score = sum(len(dna.protocols) for dna in cognitive_dna.values()) * 2

        # Add complexity for unique cognitive signatures
        unique_signatures = len(
            set(dna.cognitive_signature for dna in cognitive_dna.values())
        )
        signature_score = unique_signatures * 5

        total_score = base_score + protocol_score + signature_score
        return min(total_score / 100.0, 10.0)  # Normalize to 0-10 scale

    def _serialize_dna(self, dna: StationDNA) -> Dict[str, Any]:
        """Serialize StationDNA for storage"""
        return {
            "station_id": dna.station_id,
            "station_name": dna.station_name,
            "prompt_template": dna.prompt_template,
            "persona": dna.persona,
            "protocols": dna.protocols,
            "key_instructions": dna.key_instructions,
            "output_format": dna.output_format,
            "cognitive_signature": dna.cognitive_signature,
        }

    async def analyze_compilation_quality(self, compiled_prompt: str) -> Dict[str, Any]:
        """
        Analyze the quality of a compiled monolithic prompt

        Returns metrics about the compilation quality, completeness, and effectiveness
        """
        try:
            analysis = {
                "prompt_length": len(compiled_prompt),
                "section_completeness": self._analyze_sections(compiled_prompt),
                "station_integration": self._analyze_station_integration(
                    compiled_prompt
                ),
                "protocol_coverage": self._analyze_protocol_coverage(compiled_prompt),
                "complexity_metrics": self._analyze_complexity(compiled_prompt),
                "quality_score": 0.0,
            }

            # Calculate overall quality score
            analysis["quality_score"] = self._calculate_quality_score(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing compilation quality: {e}")
            return {"error": str(e)}

    def _analyze_sections(self, prompt: str) -> Dict[str, bool]:
        """Check if all required sections are present"""
        required_sections = [
            "GLOBAL CONTEXT",
            "UNIFIED PERSONA",
            "INTEGRATED PROTOCOLS",
            "SEQUENTIAL ANALYSIS TASKS",
            "OUTPUT REQUIREMENTS",
            "QUALITY STANDARDS",
        ]

        section_analysis = {}
        for section in required_sections:
            section_analysis[section] = section in prompt

        return section_analysis

    def _analyze_station_integration(self, prompt: str) -> Dict[str, Any]:
        """Analyze how well all 8 stations are integrated"""
        station_mentions = {}
        for station_id, station_name in self.station_mappings.items():
            station_mentions[station_name] = station_name in prompt

        integration_score = sum(station_mentions.values()) / len(station_mentions)

        return {
            "station_mentions": station_mentions,
            "integration_score": integration_score,
            "total_stations_integrated": sum(station_mentions.values()),
        }

    def _analyze_protocol_coverage(self, prompt: str) -> Dict[str, Any]:
        """Analyze protocol coverage in the compiled prompt"""
        common_protocols = [
            "analysis",
            "assessment",
            "evaluation",
            "synthesis",
            "integration",
            "recommendation",
            "exploration",
            "investigation",
        ]

        protocol_coverage = {}
        for protocol in common_protocols:
            protocol_coverage[protocol] = protocol.lower() in prompt.lower()

        coverage_score = sum(protocol_coverage.values()) / len(protocol_coverage)

        return {
            "protocol_coverage": protocol_coverage,
            "coverage_score": coverage_score,
        }

    def _analyze_complexity(self, prompt: str) -> Dict[str, Any]:
        """Analyze complexity metrics of the prompt"""
        return {
            "word_count": len(prompt.split()),
            "sentence_count": prompt.count(".") + prompt.count("!") + prompt.count("?"),
            "section_count": prompt.count("#"),
            "instruction_count": prompt.count("•") + prompt.count("-"),
            "complexity_rating": (
                "high"
                if len(prompt) > 5000
                else "medium" if len(prompt) > 2000 else "low"
            ),
        }

    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall quality score from analysis metrics"""
        try:
            # Section completeness (40%)
            sections = analysis.get("section_completeness", {})
            section_score = sum(sections.values()) / len(sections) * 0.4

            # Station integration (30%)
            integration = analysis.get("station_integration", {})
            integration_score = integration.get("integration_score", 0) * 0.3

            # Protocol coverage (20%)
            protocol = analysis.get("protocol_coverage", {})
            protocol_score = protocol.get("coverage_score", 0) * 0.2

            # Length appropriateness (10%)
            length = analysis.get("prompt_length", 0)
            length_score = min(length / 5000, 1.0) * 0.1  # Optimal around 5000 chars

            total_score = (
                section_score + integration_score + protocol_score + length_score
            )
            return round(total_score * 100, 2)  # Convert to percentage

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0


# Example usage and testing functions
async def example_compilation():
    """Example of how to use the MonolithCompilerService"""
    compiler = MonolithCompilerService()

    try:
        # Compile a monolithic challenger
        result = await compiler.compile_perfect_monolith(
            golden_case_id="test_case_001", custom_name="Test Perfect Monolith"
        )

        print("Compiled Monolith:")
        print(f"Name: {result['prompt_name']}")
        print(f"Length: {len(result['prompt_text'])} characters")
        print(
            f"Complexity Score: {result['compilation_metadata']['compilation_stats']['complexity_score']}"
        )

        # Analyze quality
        quality_analysis = await compiler.analyze_compilation_quality(
            result["prompt_text"]
        )
        print(f"Quality Score: {quality_analysis['quality_score']}%")

        return result

    except Exception as e:
        print(f"Compilation failed: {e}")
        return None


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_compilation())
