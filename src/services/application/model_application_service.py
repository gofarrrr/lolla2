"""
METIS Model Application Service
Part of Application Services Cluster - Focused on executing model-specific application strategies

Extracted from model_manager.py during Phase 5.3 decomposition.
Single Responsibility: Execute various application strategies using selected models.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.services.contracts.application_contracts import (
    IModelApplicationService,
    ModelApplicationContract,
    ApplicationStrategy,
    ModelApplicationStatus,
)
from src.integrations.llm.unified_client import UnifiedLLMClient


class ModelApplicationService(IModelApplicationService):
    """
    Focused service for executing model application strategies
    Clean extraction from model_manager.py application strategy methods
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm_client = UnifiedLLMClient()

        # Application strategy configurations
        self.strategy_configs = {
            ApplicationStrategy.SYSTEMS_THINKING: {
                "prompt_template": self._get_systems_thinking_template(),
                "max_tokens": 2500,
                "temperature": 0.7,
                "requires_context_enhancement": True,
                "expected_sections": [
                    "system_components",
                    "interconnections",
                    "feedback_loops",
                ],
            },
            ApplicationStrategy.CRITICAL_THINKING: {
                "prompt_template": self._get_critical_thinking_template(),
                "max_tokens": 2000,
                "temperature": 0.6,
                "requires_context_enhancement": False,
                "expected_sections": ["assumptions", "evidence", "counterarguments"],
            },
            ApplicationStrategy.MECE_FRAMEWORK: {
                "prompt_template": self._get_mece_framework_template(),
                "max_tokens": 2200,
                "temperature": 0.5,
                "requires_context_enhancement": False,
                "expected_sections": [
                    "categories",
                    "completeness_check",
                    "mutual_exclusivity",
                ],
            },
            ApplicationStrategy.HYPOTHESIS_TESTING: {
                "prompt_template": self._get_hypothesis_testing_template(),
                "max_tokens": 2300,
                "temperature": 0.6,
                "requires_context_enhancement": True,
                "expected_sections": [
                    "hypothesis",
                    "test_design",
                    "validation_criteria",
                ],
            },
            ApplicationStrategy.DECISION_FRAMEWORK: {
                "prompt_template": self._get_decision_framework_template(),
                "max_tokens": 2400,
                "temperature": 0.5,
                "requires_context_enhancement": True,
                "expected_sections": ["options", "criteria", "evaluation_matrix"],
            },
            ApplicationStrategy.GENERIC_APPLICATION: {
                "prompt_template": self._get_generic_application_template(),
                "max_tokens": 1800,
                "temperature": 0.7,
                "requires_context_enhancement": False,
                "expected_sections": ["analysis", "insights", "recommendations"],
            },
        }

        # Model-strategy compatibility matrix
        self.model_strategy_compatibility = {
            "deepseek_chat": {
                ApplicationStrategy.SYSTEMS_THINKING: 0.95,
                ApplicationStrategy.CRITICAL_THINKING: 0.88,
                ApplicationStrategy.HYPOTHESIS_TESTING: 0.92,
                ApplicationStrategy.MECE_FRAMEWORK: 0.85,
                ApplicationStrategy.DECISION_FRAMEWORK: 0.90,
                ApplicationStrategy.GENERIC_APPLICATION: 0.80,
            },
            "claude_sonnet": {
                ApplicationStrategy.CRITICAL_THINKING: 0.93,
                ApplicationStrategy.SYSTEMS_THINKING: 0.87,
                ApplicationStrategy.DECISION_FRAMEWORK: 0.88,
                ApplicationStrategy.MECE_FRAMEWORK: 0.82,
                ApplicationStrategy.HYPOTHESIS_TESTING: 0.85,
                ApplicationStrategy.GENERIC_APPLICATION: 0.85,
            },
            "generic_llm": {
                ApplicationStrategy.GENERIC_APPLICATION: 0.90,
                ApplicationStrategy.CRITICAL_THINKING: 0.75,
                ApplicationStrategy.SYSTEMS_THINKING: 0.70,
                ApplicationStrategy.MECE_FRAMEWORK: 0.72,
                ApplicationStrategy.HYPOTHESIS_TESTING: 0.68,
                ApplicationStrategy.DECISION_FRAMEWORK: 0.70,
            },
        }

        # Quality assessment weights
        self.quality_weights = {
            "coherence": 0.25,
            "completeness": 0.25,
            "accuracy": 0.20,
            "relevance": 0.15,
            "structure": 0.15,
        }

        # Application metrics
        self.application_metrics = {
            "total_applications": 0,
            "applications_by_strategy": {
                strategy.value: 0 for strategy in ApplicationStrategy
            },
            "success_rate_by_strategy": {
                strategy.value: 0.0 for strategy in ApplicationStrategy
            },
            "average_processing_time_ms": 0.0,
        }

        self.logger.info("🎯 ModelApplicationService initialized")

    async def apply_model_strategy(
        self,
        model_id: str,
        strategy: ApplicationStrategy,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ModelApplicationContract:
        """
        Core service method: Apply specific strategy using selected model
        Comprehensive strategy execution with quality assessment and context enhancement
        """
        try:
            start_time = datetime.utcnow()
            application_id = f"{model_id}_{strategy.value}_{start_time.timestamp()}"

            # Validate model-strategy compatibility
            compatibility_score = await self._assess_compatibility(model_id, strategy)

            if compatibility_score < 0.6:
                return self._create_failed_application_contract(
                    application_id,
                    model_id,
                    strategy,
                    input_data,
                    f"Low compatibility score: {compatibility_score:.2f}",
                    0.0,
                )

            # Prepare application context
            enhanced_context = await self._prepare_application_context(
                strategy, input_data, context, compatibility_score
            )

            # Execute strategy-specific application
            application_result = await self._execute_strategy_application(
                model_id, strategy, enhanced_context
            )

            if not application_result["success"]:
                processing_time = (
                    datetime.utcnow() - start_time
                ).total_seconds() * 1000
                return self._create_failed_application_contract(
                    application_id,
                    model_id,
                    strategy,
                    input_data,
                    application_result["error"],
                    processing_time,
                )

            # Assess output quality
            quality_metrics = await self._assess_output_quality(
                application_result["output"], strategy
            )

            # Calculate confidence score
            confidence_score = await self._calculate_application_confidence(
                compatibility_score, quality_metrics, application_result
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Create successful application contract
            application_contract = ModelApplicationContract(
                application_id=application_id,
                model_id=model_id,
                strategy_used=strategy,
                application_status=ModelApplicationStatus.COMPLETED,
                input_data=input_data,
                output_data=application_result["output"],
                processing_metadata={
                    "compatibility_score": compatibility_score,
                    "context_enhancement_applied": enhanced_context.get(
                        "enhanced", False
                    ),
                    "llm_response_time_ms": application_result.get(
                        "response_time_ms", 0
                    ),
                    "tokens_used": application_result.get("tokens_used", 0),
                },
                confidence_score=confidence_score,
                quality_metrics=quality_metrics,
                application_timestamp=datetime.utcnow(),
                processing_time_ms=processing_time,
                service_version="v5_modular",
            )

            # Update metrics
            await self._update_application_metrics(strategy, True, processing_time)

            self.logger.info(
                f"🎯 Strategy applied successfully: {model_id} - {strategy.value} ({processing_time:.0f}ms)"
            )
            return application_contract

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            error_contract = self._create_failed_application_contract(
                f"error_{start_time.timestamp()}",
                model_id,
                strategy,
                input_data,
                str(e),
                processing_time,
            )

            await self._update_application_metrics(strategy, False, processing_time)

            self.logger.error(
                f"❌ Strategy application failed: {model_id} - {strategy.value}: {e}"
            )
            return error_contract

    async def get_supported_strategies(
        self, model_id: str
    ) -> List[ApplicationStrategy]:
        """
        Core service method: Get list of strategies supported by model
        Strategy compatibility assessment based on performance matrix
        """
        try:
            compatibility_scores = self.model_strategy_compatibility.get(model_id, {})

            if not compatibility_scores:
                # Fallback for unknown models - assume basic support
                return [ApplicationStrategy.GENERIC_APPLICATION]

            # Return strategies with compatibility >= 0.7
            supported_strategies = [
                strategy
                for strategy, score in compatibility_scores.items()
                if score >= 0.7
            ]

            # Sort by compatibility score
            supported_strategies.sort(
                key=lambda s: compatibility_scores.get(s, 0.0), reverse=True
            )

            self.logger.debug(
                f"🎯 Supported strategies for {model_id}: {len(supported_strategies)}"
            )
            return supported_strategies

        except Exception as e:
            self.logger.error(
                f"❌ Strategy support assessment failed for {model_id}: {e}"
            )
            return [ApplicationStrategy.GENERIC_APPLICATION]

    async def get_strategy_performance_summary(
        self, strategy: ApplicationStrategy
    ) -> Dict[str, Any]:
        """Get performance summary for a specific strategy"""
        try:
            strategy_value = strategy.value

            performance_data = {
                "strategy": strategy_value,
                "total_applications": self.application_metrics[
                    "applications_by_strategy"
                ][strategy_value],
                "success_rate": self.application_metrics["success_rate_by_strategy"][
                    strategy_value
                ],
                "average_processing_time_ms": self.application_metrics.get(
                    f"avg_time_{strategy_value}", 0.0
                ),
                "compatible_models": [],
                "strategy_configuration": self.strategy_configs.get(strategy, {}),
            }

            # Find compatible models
            for model_id, compatibilities in self.model_strategy_compatibility.items():
                if strategy in compatibilities and compatibilities[strategy] >= 0.7:
                    performance_data["compatible_models"].append(
                        {
                            "model_id": model_id,
                            "compatibility_score": compatibilities[strategy],
                        }
                    )

            # Sort by compatibility
            performance_data["compatible_models"].sort(
                key=lambda x: x["compatibility_score"], reverse=True
            )

            return performance_data

        except Exception as e:
            self.logger.error(
                f"❌ Strategy performance summary failed for {strategy}: {e}"
            )
            return {"error": str(e)}

    async def compare_strategy_performance(
        self, strategies: List[ApplicationStrategy]
    ) -> Dict[str, Any]:
        """Compare performance across multiple strategies"""
        try:
            comparison_data = {
                "strategies_compared": len(strategies),
                "comparison_results": [],
                "performance_ranking": [],
                "comparison_timestamp": datetime.utcnow().isoformat(),
            }

            strategy_performances = []

            for strategy in strategies:
                performance = await self.get_strategy_performance_summary(strategy)

                strategy_performances.append(
                    {
                        "strategy": strategy.value,
                        "success_rate": performance.get("success_rate", 0.0),
                        "total_applications": performance.get("total_applications", 0),
                        "avg_processing_time": performance.get(
                            "average_processing_time_ms", 0.0
                        ),
                        "compatible_models_count": len(
                            performance.get("compatible_models", [])
                        ),
                    }
                )

            # Sort by success rate
            strategy_performances.sort(key=lambda x: x["success_rate"], reverse=True)

            comparison_data["comparison_results"] = strategy_performances
            comparison_data["performance_ranking"] = [
                {
                    "rank": i + 1,
                    "strategy": perf["strategy"],
                    "success_rate": perf["success_rate"],
                }
                for i, perf in enumerate(strategy_performances)
            ]

            return comparison_data

        except Exception as e:
            self.logger.error(f"❌ Strategy performance comparison failed: {e}")
            return {"error": str(e)}

    async def _assess_compatibility(
        self, model_id: str, strategy: ApplicationStrategy
    ) -> float:
        """Assess compatibility between model and strategy"""
        try:
            model_compatibilities = self.model_strategy_compatibility.get(model_id, {})
            return model_compatibilities.get(
                strategy, 0.5
            )  # Default moderate compatibility

        except Exception as e:
            self.logger.error(f"❌ Compatibility assessment failed: {e}")
            return 0.5

    async def _prepare_application_context(
        self,
        strategy: ApplicationStrategy,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        compatibility_score: float,
    ) -> Dict[str, Any]:
        """Prepare enhanced context for strategy application"""
        try:
            strategy_config = self.strategy_configs.get(strategy, {})
            enhanced_context = {
                **context,
                "strategy_name": strategy.value,
                "compatibility_score": compatibility_score,
                "enhanced": False,
            }

            # Apply context enhancement if required
            if strategy_config.get("requires_context_enhancement", False):
                enhancement_result = await self._enhance_context_for_strategy(
                    strategy, input_data, context
                )
                enhanced_context.update(enhancement_result)
                enhanced_context["enhanced"] = True

            return enhanced_context

        except Exception as e:
            self.logger.error(f"❌ Context preparation failed: {e}")
            return context

    async def _execute_strategy_application(
        self, model_id: str, strategy: ApplicationStrategy, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the actual strategy application using LLM"""
        try:
            strategy_config = self.strategy_configs[strategy]

            # Build prompt from template
            prompt = await self._build_strategy_prompt(strategy, context)

            # Generate response using LLM
            start_time = datetime.utcnow()

            response = await self.llm_client.generate_response(
                prompt=prompt,
                max_tokens=strategy_config.get("max_tokens", 2000),
                temperature=strategy_config.get("temperature", 0.7),
            )

            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            if not response:
                return {"success": False, "error": "No response from LLM"}

            # Parse and structure the output
            structured_output = await self._structure_strategy_output(
                strategy, response
            )

            return {
                "success": True,
                "output": structured_output,
                "raw_response": response,
                "response_time_ms": response_time,
                "tokens_used": len(response.split()),  # Simplified token count
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _assess_output_quality(
        self, output_data: Dict[str, Any], strategy: ApplicationStrategy
    ) -> Dict[str, float]:
        """Assess quality of strategy application output"""
        try:
            quality_scores = {}

            # Coherence assessment
            quality_scores["coherence"] = await self._assess_coherence(output_data)

            # Completeness assessment based on expected sections
            quality_scores["completeness"] = await self._assess_completeness(
                output_data, strategy
            )

            # Structure assessment
            quality_scores["structure"] = await self._assess_structure(
                output_data, strategy
            )

            # Relevance assessment
            quality_scores["relevance"] = await self._assess_relevance(output_data)

            # Accuracy assessment (simplified)
            quality_scores["accuracy"] = await self._assess_accuracy(output_data)

            return quality_scores

        except Exception as e:
            self.logger.error(f"❌ Quality assessment failed: {e}")
            return {
                "coherence": 0.7,
                "completeness": 0.7,
                "structure": 0.7,
                "relevance": 0.7,
                "accuracy": 0.7,
            }

    async def _calculate_application_confidence(
        self,
        compatibility_score: float,
        quality_metrics: Dict[str, float],
        application_result: Dict[str, Any],
    ) -> float:
        """Calculate overall confidence in application result"""
        try:
            # Weighted combination of factors
            quality_average = (
                sum(quality_metrics.values()) / len(quality_metrics)
                if quality_metrics
                else 0.5
            )

            confidence_factors = [
                compatibility_score * 0.3,  # Model-strategy fit
                quality_average * 0.5,  # Output quality
                min(application_result.get("response_time_ms", 5000) / 10000, 1.0)
                * 0.1,  # Response time (inverted)
                0.1,  # Base confidence
            ]

            final_confidence = sum(confidence_factors)
            return max(0.1, min(final_confidence, 0.95))  # Bound between 0.1 and 0.95

        except Exception as e:
            self.logger.error(f"❌ Confidence calculation failed: {e}")
            return 0.5

    async def _enhance_context_for_strategy(
        self,
        strategy: ApplicationStrategy,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply strategy-specific context enhancements"""
        enhancement = {}

        try:
            if strategy == ApplicationStrategy.SYSTEMS_THINKING:
                enhancement.update(
                    {
                        "system_perspective_cues": [
                            "Consider interconnections and relationships",
                            "Identify feedback loops and system dynamics",
                            "Look for emergent properties and patterns",
                        ],
                        "systems_analysis_framework": "holistic_approach",
                    }
                )

            elif strategy == ApplicationStrategy.HYPOTHESIS_TESTING:
                enhancement.update(
                    {
                        "hypothesis_generation_guidelines": [
                            "Formulate testable hypotheses",
                            "Define clear validation criteria",
                            "Design appropriate test methods",
                        ],
                        "scientific_rigor_level": "high",
                    }
                )

            elif strategy == ApplicationStrategy.DECISION_FRAMEWORK:
                enhancement.update(
                    {
                        "decision_structure_cues": [
                            "Identify decision criteria and weights",
                            "Evaluate alternatives systematically",
                            "Consider risks and trade-offs",
                        ],
                        "decision_methodology": "structured_analysis",
                    }
                )

            return enhancement

        except Exception as e:
            self.logger.error(f"❌ Context enhancement failed: {e}")
            return {}

    async def _build_strategy_prompt(
        self, strategy: ApplicationStrategy, context: Dict[str, Any]
    ) -> str:
        """Build prompt for specific strategy"""
        try:
            strategy_config = self.strategy_configs[strategy]
            template = strategy_config["prompt_template"]

            # Replace template variables with context values
            formatted_prompt = template.format(
                problem_statement=context.get("problem_statement", ""),
                business_context=context.get("business_context", {}),
                input_data=context.get("input_data", {}),
                enhancement_cues=context.get("system_perspective_cues", [])
                or context.get("hypothesis_generation_guidelines", [])
                or context.get("decision_structure_cues", []),
                compatibility_score=context.get("compatibility_score", 0.8),
            )

            return formatted_prompt

        except Exception as e:
            self.logger.error(f"❌ Prompt building failed: {e}")
            return f"Analyze the following using {strategy.value} approach: {context}"

    async def _structure_strategy_output(
        self, strategy: ApplicationStrategy, raw_response: str
    ) -> Dict[str, Any]:
        """Structure raw LLM response according to strategy requirements"""
        try:
            strategy_config = self.strategy_configs.get(strategy, {})
            expected_sections = strategy_config.get("expected_sections", [])

            # Simple section extraction (would be more sophisticated in production)
            structured = {
                "raw_analysis": raw_response,
                "structured_sections": {},
                "strategy_applied": strategy.value,
                "extraction_confidence": 0.8,
            }

            # Extract sections based on strategy
            for section in expected_sections:
                section_content = self._extract_section_content(raw_response, section)
                structured["structured_sections"][section] = section_content

            return structured

        except Exception as e:
            self.logger.error(f"❌ Output structuring failed: {e}")
            return {
                "raw_analysis": raw_response,
                "error": str(e),
                "strategy_applied": strategy.value,
            }

    def _extract_section_content(self, text: str, section_name: str) -> str:
        """Extract content for specific section from text"""
        try:
            # Simple extraction based on section keywords
            section_keywords = {
                "system_components": ["component", "element", "part"],
                "interconnections": ["connection", "relationship", "link"],
                "feedback_loops": ["feedback", "loop", "cycle"],
                "assumptions": ["assume", "assumption", "premise"],
                "evidence": ["evidence", "data", "proof"],
                "counterarguments": ["counter", "alternative", "opposing"],
                "categories": ["category", "group", "type"],
                "hypothesis": ["hypothesis", "theory", "proposition"],
                "options": ["option", "alternative", "choice"],
                "criteria": ["criteria", "criterion", "factor"],
            }

            keywords = section_keywords.get(section_name, [section_name])

            # Find paragraphs containing keywords (simplified)
            paragraphs = text.split("\n\n")
            relevant_paragraphs = []

            for paragraph in paragraphs:
                if any(keyword.lower() in paragraph.lower() for keyword in keywords):
                    relevant_paragraphs.append(paragraph.strip())

            return "\n\n".join(relevant_paragraphs) if relevant_paragraphs else ""

        except Exception as e:
            return f"Section extraction failed: {str(e)}"

    async def _assess_coherence(self, output_data: Dict[str, Any]) -> float:
        """Assess coherence of output"""
        try:
            raw_text = output_data.get("raw_analysis", "")

            # Simple heuristics for coherence
            coherence_score = 0.7  # Base score

            # Check for logical flow indicators
            flow_indicators = [
                "therefore",
                "thus",
                "consequently",
                "however",
                "furthermore",
            ]
            flow_count = sum(
                1 for indicator in flow_indicators if indicator in raw_text.lower()
            )

            coherence_score += min(flow_count * 0.05, 0.15)

            # Check for structure
            if len(raw_text.split("\n\n")) >= 3:  # Multiple paragraphs
                coherence_score += 0.1

            return min(coherence_score, 1.0)

        except Exception:
            return 0.7

    async def _assess_completeness(
        self, output_data: Dict[str, Any], strategy: ApplicationStrategy
    ) -> float:
        """Assess completeness based on expected sections"""
        try:
            expected_sections = self.strategy_configs.get(strategy, {}).get(
                "expected_sections", []
            )

            if not expected_sections:
                return 0.8  # Default for strategies without section requirements

            structured_sections = output_data.get("structured_sections", {})

            # Check how many expected sections have content
            sections_with_content = sum(
                1
                for section in expected_sections
                if structured_sections.get(section, "").strip()
            )

            completeness_score = sections_with_content / len(expected_sections)
            return completeness_score

        except Exception:
            return 0.7

    async def _assess_structure(
        self, output_data: Dict[str, Any], strategy: ApplicationStrategy
    ) -> float:
        """Assess structural quality of output"""
        try:
            raw_text = output_data.get("raw_analysis", "")

            # Check for clear structure indicators
            structure_indicators = [
                "1.",
                "2.",
                "3.",
                "•",
                "-",
                "First",
                "Second",
                "Next",
                "Finally",
            ]
            structure_score = 0.6  # Base score

            structure_count = sum(
                1 for indicator in structure_indicators if indicator in raw_text
            )
            structure_score += min(structure_count * 0.05, 0.25)

            # Check paragraph organization
            paragraphs = raw_text.split("\n\n")
            if len(paragraphs) >= 3:
                structure_score += 0.15

            return min(structure_score, 1.0)

        except Exception:
            return 0.7

    async def _assess_relevance(self, output_data: Dict[str, Any]) -> float:
        """Assess relevance of output to input"""
        # Simplified relevance assessment
        return 0.8

    async def _assess_accuracy(self, output_data: Dict[str, Any]) -> float:
        """Assess accuracy of output (simplified)"""
        # Simplified accuracy assessment
        return 0.75

    def _create_failed_application_contract(
        self,
        application_id: str,
        model_id: str,
        strategy: ApplicationStrategy,
        input_data: Dict[str, Any],
        error_message: str,
        processing_time_ms: float,
    ) -> ModelApplicationContract:
        """Create contract for failed application"""
        return ModelApplicationContract(
            application_id=application_id,
            model_id=model_id,
            strategy_used=strategy,
            application_status=ModelApplicationStatus.FAILED,
            input_data=input_data,
            output_data={"error": error_message},
            processing_metadata={"failure_reason": error_message},
            confidence_score=0.0,
            quality_metrics={"error": 1.0},
            application_timestamp=datetime.utcnow(),
            processing_time_ms=processing_time_ms,
            service_version="v5_modular_error",
        )

    async def _update_application_metrics(
        self, strategy: ApplicationStrategy, success: bool, processing_time_ms: float
    ):
        """Update application performance metrics"""
        try:
            self.application_metrics["total_applications"] += 1
            self.application_metrics["applications_by_strategy"][strategy.value] += 1

            # Update success rate
            strategy_applications = self.application_metrics[
                "applications_by_strategy"
            ][strategy.value]
            if strategy_applications > 0:
                current_success_rate = self.application_metrics[
                    "success_rate_by_strategy"
                ][strategy.value]
                new_success_rate = (
                    (current_success_rate * (strategy_applications - 1))
                    + (1.0 if success else 0.0)
                ) / strategy_applications
                self.application_metrics["success_rate_by_strategy"][
                    strategy.value
                ] = new_success_rate

            # Update average processing time
            total_applications = self.application_metrics["total_applications"]
            current_avg = self.application_metrics["average_processing_time_ms"]
            new_avg = (
                (current_avg * (total_applications - 1)) + processing_time_ms
            ) / total_applications
            self.application_metrics["average_processing_time_ms"] = new_avg

        except Exception as e:
            self.logger.error(f"❌ Metrics update failed: {e}")

    # Strategy prompt templates
    def _get_systems_thinking_template(self) -> str:
        return """
        Apply systems thinking analysis to the following problem:
        
        Problem: {problem_statement}
        Business Context: {business_context}
        Input Data: {input_data}
        
        Systems Analysis Guidelines:
        {enhancement_cues}
        
        Please provide a comprehensive systems analysis including:
        1. System Components: Identify key elements and stakeholders
        2. Interconnections: Map relationships and dependencies  
        3. Feedback Loops: Identify reinforcing and balancing loops
        4. System Dynamics: Analyze how the system behaves over time
        5. Leverage Points: Identify where small changes can have big impact
        
        Focus on holistic understanding and emergent properties.
        """

    def _get_critical_thinking_template(self) -> str:
        return """
        Apply critical thinking analysis to the following problem:
        
        Problem: {problem_statement}
        Business Context: {business_context}
        Input Data: {input_data}
        
        Critical Analysis Guidelines:
        {enhancement_cues}
        
        Please provide a thorough critical analysis including:
        1. Assumptions: Identify and examine underlying assumptions
        2. Evidence: Evaluate the quality and relevance of available evidence
        3. Logic: Assess the logical structure of arguments
        4. Counterarguments: Consider alternative perspectives and objections
        5. Biases: Identify potential cognitive biases affecting judgment
        6. Conclusion: Draw well-reasoned conclusions based on analysis
        
        Focus on objective evaluation and logical reasoning.
        """

    def _get_mece_framework_template(self) -> str:
        return """
        Apply MECE (Mutually Exclusive, Collectively Exhaustive) framework to analyze:
        
        Problem: {problem_statement}
        Business Context: {business_context}
        Input Data: {input_data}
        
        MECE Analysis Requirements:
        1. Categories: Create mutually exclusive categories that cover all aspects
        2. Completeness Check: Ensure all important elements are included
        3. Mutual Exclusivity: Verify no overlap between categories
        4. Hierarchical Structure: Organize into clear levels if needed
        5. Validation: Check that categorization serves the analysis purpose
        
        Ensure your analysis is both comprehensive and organized.
        """

    def _get_hypothesis_testing_template(self) -> str:
        return """
        Apply hypothesis testing methodology to:
        
        Problem: {problem_statement}
        Business Context: {business_context}
        Input Data: {input_data}
        
        Hypothesis Testing Guidelines:
        {enhancement_cues}
        
        Please structure your analysis as follows:
        1. Hypothesis: Formulate clear, testable hypotheses
        2. Test Design: Define how to test each hypothesis
        3. Validation Criteria: Specify what evidence would support/refute hypotheses
        4. Data Analysis: Examine available data for hypothesis testing
        5. Results: Present findings and statistical significance if applicable
        6. Conclusions: Accept, reject, or modify hypotheses based on evidence
        
        Focus on scientific rigor and evidence-based conclusions.
        """

    def _get_decision_framework_template(self) -> str:
        return """
        Apply structured decision framework to:
        
        Problem: {problem_statement}
        Business Context: {business_context}
        Input Data: {input_data}
        
        Decision Framework Guidelines:
        {enhancement_cues}
        
        Structure your decision analysis as follows:
        1. Options: Identify and clearly define all viable alternatives
        2. Criteria: Establish decision criteria and relative weights
        3. Evaluation Matrix: Score each option against each criterion
        4. Risk Assessment: Analyze risks and uncertainties for each option
        5. Trade-offs: Explicitly identify trade-offs between options
        6. Recommendation: Provide clear recommendation with rationale
        
        Focus on systematic evaluation and transparent reasoning.
        """

    def _get_generic_application_template(self) -> str:
        return """
        Analyze the following business situation:
        
        Problem: {problem_statement}
        Business Context: {business_context}
        Input Data: {input_data}
        
        Please provide a comprehensive analysis including:
        1. Analysis: Break down the key aspects of the situation
        2. Insights: Identify important patterns, trends, and relationships
        3. Recommendations: Suggest actionable next steps
        4. Considerations: Note important factors to keep in mind
        
        Provide practical, actionable insights that address the core issue.
        """

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "ModelApplicationService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "strategy_application",
                "model_compatibility_assessment",
                "output_quality_evaluation",
                "context_enhancement",
                "performance_tracking",
            ],
            "supported_strategies": [
                strategy.value for strategy in ApplicationStrategy
            ],
            "model_compatibility_matrix": len(self.model_strategy_compatibility),
            "application_statistics": {
                "total_applications": self.application_metrics["total_applications"],
                "average_processing_time_ms": round(
                    self.application_metrics["average_processing_time_ms"], 2
                ),
                "applications_by_strategy": self.application_metrics[
                    "applications_by_strategy"
                ],
            },
            "quality_assessment_weights": self.quality_weights,
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_model_application_service: Optional[ModelApplicationService] = None


def get_model_application_service() -> ModelApplicationService:
    """Get or create global model application service instance"""
    global _model_application_service

    if _model_application_service is None:
        _model_application_service = ModelApplicationService()

    return _model_application_service
