"""
METIS Mental Model Extractor
Extract individual mental models from database for direct agent access
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelCategory(Enum):
    """Categories of mental models"""

    ANALYTICAL = "analytical"
    STRATEGIC = "strategic"
    CREATIVE = "creative"
    SYSTEMS = "systems"
    DECISION = "decision"
    BEHAVIORAL = "behavioral"
    COMMUNICATION = "communication"
    LEADERSHIP = "leadership"


@dataclass
class ExtractedMentalModel:
    """Represents an extracted mental model ready for agent use"""

    model_id: str
    name: str
    category: ModelCategory
    description: str
    application_context: List[str]
    key_concepts: List[str]
    implementation_steps: List[str]
    bias_vulnerabilities: List[str]
    synergy_models: List[str]
    conflict_models: List[str]
    ethical_considerations: List[str]
    prompt_integration_guide: str
    effectiveness_metrics: Dict[str, float]
    source_file: str
    extraction_timestamp: datetime

    def to_agent_prompt(self) -> str:
        """Convert mental model to agent-ready prompt format"""
        prompt = f"""
## Mental Model: {self.name}

**Category**: {self.category.value}
**Description**: {self.description}

**Key Concepts**:
{chr(10).join(f"• {concept}" for concept in self.key_concepts)}

**Implementation Steps**:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(self.implementation_steps))}

**Application Context**: {', '.join(self.application_context)}

**Bias Vulnerabilities**: {', '.join(self.bias_vulnerabilities)}

**Works Well With**: {', '.join(self.synergy_models)}

**Conflicts With**: {', '.join(self.conflict_models)}

**Prompt Integration Guide**: {self.prompt_integration_guide}

**Ethical Considerations**: {', '.join(self.ethical_considerations)}
"""
        return prompt.strip()

    def to_json(self) -> str:
        """Convert to JSON for storage"""
        data = asdict(self)
        data["category"] = self.category.value
        data["extraction_timestamp"] = self.extraction_timestamp.isoformat()
        return json.dumps(data, indent=2)


class MentalModelExtractor:
    """Extracts mental models from database files for agent access"""

    def __init__(
        self,
        source_dir: str = "db",
        output_dir: str = "src/intelligence/extracted_models",
        agent_prompts_dir: str = "src/agents/mental_model_prompts",
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.agent_prompts_dir = Path(agent_prompts_dir)

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_prompts_dir.mkdir(parents=True, exist_ok=True)

        self.extracted_models: Dict[str, ExtractedMentalModel] = {}
        self.extraction_statistics = {
            "total_files_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "models_by_category": {cat.value: 0 for cat in ModelCategory},
            "total_concepts_extracted": 0,
            "total_implementation_steps": 0,
        }

        self.logger = logging.getLogger(__name__)

    async def extract_all_models(self) -> int:
        """Extract all mental models from source directory"""
        self.logger.info(f"🧠 Starting mental model extraction from {self.source_dir}")

        if not self.source_dir.exists():
            self.logger.error(f"❌ Source directory {self.source_dir} does not exist")
            return 0

        # Find all JSON/MD files
        model_files = list(self.source_dir.glob("*.json")) + list(
            self.source_dir.glob("*.md")
        )

        if not model_files:
            self.logger.warning(f"⚠️ No model files found in {self.source_dir}")
            return 0

        self.logger.info(f"📁 Found {len(model_files)} model files to process")

        # Process each file
        for file_path in model_files:
            try:
                await self.extract_model_from_file(file_path)
                self.extraction_statistics["successful_extractions"] += 1

            except Exception as e:
                self.logger.error(f"❌ Failed to extract from {file_path}: {e}")
                self.extraction_statistics["failed_extractions"] += 1

            self.extraction_statistics["total_files_processed"] += 1

        # Generate agent-ready files
        await self.generate_agent_files()

        # Generate extraction report
        await self.generate_extraction_report()

        success_count = self.extraction_statistics["successful_extractions"]
        self.logger.info(
            f"✅ Mental model extraction completed: {success_count} models extracted"
        )

        return success_count

    async def extract_model_from_file(
        self, file_path: Path
    ) -> Optional[ExtractedMentalModel]:
        """Extract mental model from a single file"""
        try:
            # Read and parse file
            content = file_path.read_text(encoding="utf-8")

            if file_path.suffix == ".json":
                data = json.loads(content)
            else:
                # For .md files, try to extract JSON from content
                if content.strip().startswith("{"):
                    data = json.loads(content)
                else:
                    # Handle markdown format - convert to structured data
                    data = await self._parse_markdown_model(content, file_path.name)

            # Extract mental model information
            model = await self._extract_model_data(data, file_path.name)

            if model:
                self.extracted_models[model.model_id] = model
                self.extraction_statistics["models_by_category"][
                    model.category.value
                ] += 1
                self.extraction_statistics["total_concepts_extracted"] += len(
                    model.key_concepts
                )
                self.extraction_statistics["total_implementation_steps"] += len(
                    model.implementation_steps
                )

                self.logger.info(
                    f"📚 Extracted model: {model.name} ({model.category.value})"
                )
                return model

        except Exception as e:
            self.logger.error(f"❌ Error extracting from {file_path}: {e}")
            raise

        return None

    async def _extract_model_data(
        self, data: Dict[str, Any], source_file: str
    ) -> Optional[ExtractedMentalModel]:
        """Extract structured mental model data from parsed content"""
        try:
            # Navigate nested structure to find knowledge elements
            knowledge_elements = data.get("knowledge_elements", [])
            if not knowledge_elements:
                # Try alternative paths
                if "phase5_transformation" in data:
                    knowledge_elements = data.get("knowledge_elements", [])
                else:
                    # Treat entire data as single knowledge element
                    knowledge_elements = [data]

            if not knowledge_elements:
                self.logger.warning(f"⚠️ No knowledge elements found in {source_file}")
                return None

            # Extract primary model (take first knowledge element as primary)
            primary_element = knowledge_elements[0]

            # Extract model information
            model_id = primary_element.get(
                "id", f"model_{source_file.replace('.', '_')}"
            )
            name = primary_element.get(
                "name", primary_element.get("title", f"Model from {source_file}")
            )
            description = primary_element.get(
                "description",
                primary_element.get("overview", "Mental model extracted from database"),
            )

            # Determine category
            category = self._determine_category(primary_element, name, description)

            # Extract detailed information
            application_context = self._extract_application_context(primary_element)
            key_concepts = self._extract_key_concepts(primary_element)
            implementation_steps = self._extract_implementation_steps(primary_element)
            bias_vulnerabilities = self._extract_bias_vulnerabilities(primary_element)
            synergy_models = self._extract_synergy_models(primary_element)
            conflict_models = self._extract_conflict_models(primary_element)
            ethical_considerations = self._extract_ethical_considerations(
                primary_element
            )
            prompt_integration_guide = self._extract_prompt_integration_guide(
                primary_element
            )
            effectiveness_metrics = self._extract_effectiveness_metrics(primary_element)

            model = ExtractedMentalModel(
                model_id=model_id,
                name=name,
                category=category,
                description=description,
                application_context=application_context,
                key_concepts=key_concepts,
                implementation_steps=implementation_steps,
                bias_vulnerabilities=bias_vulnerabilities,
                synergy_models=synergy_models,
                conflict_models=conflict_models,
                ethical_considerations=ethical_considerations,
                prompt_integration_guide=prompt_integration_guide,
                effectiveness_metrics=effectiveness_metrics,
                source_file=source_file,
                extraction_timestamp=datetime.now(),
            )

            return model

        except Exception as e:
            self.logger.error(f"❌ Error extracting model data: {e}")
            raise

    async def _parse_markdown_model(
        self, content: str, filename: str
    ) -> Dict[str, Any]:
        """Parse markdown format mental model"""
        # Simple markdown parser for mental models
        lines = content.split("\n")
        data = {
            "id": f"md_{filename.replace('.md', '')}",
            "name": f"Model from {filename}",
            "description": "Mental model from markdown file",
            "content_lines": lines[:50],  # First 50 lines
        }

        # Extract title if available
        for line in lines[:10]:
            if line.startswith("# "):
                data["name"] = line[2:].strip()
                break

        return data

    def _determine_category(
        self, element: Dict[str, Any], name: str, description: str
    ) -> ModelCategory:
        """Determine mental model category from content"""
        content_text = (name + " " + description + " " + str(element)).lower()

        # Category keywords
        category_keywords = {
            ModelCategory.STRATEGIC: [
                "strategy",
                "strategic",
                "planning",
                "competitive",
                "business model",
                "market",
            ],
            ModelCategory.ANALYTICAL: [
                "analysis",
                "analytical",
                "data",
                "logic",
                "reasoning",
                "critical thinking",
            ],
            ModelCategory.SYSTEMS: [
                "system",
                "systems",
                "holistic",
                "ecosystem",
                "network",
                "interconnection",
            ],
            ModelCategory.DECISION: [
                "decision",
                "choice",
                "option",
                "alternative",
                "criteria",
                "evaluation",
            ],
            ModelCategory.CREATIVE: [
                "creative",
                "innovation",
                "brainstorm",
                "ideation",
                "design thinking",
            ],
            ModelCategory.BEHAVIORAL: [
                "behavior",
                "psychology",
                "cognitive",
                "bias",
                "human",
                "motivation",
            ],
            ModelCategory.COMMUNICATION: [
                "communication",
                "message",
                "narrative",
                "storytelling",
                "presentation",
            ],
            ModelCategory.LEADERSHIP: [
                "leadership",
                "management",
                "team",
                "influence",
                "authority",
                "guidance",
            ],
        }

        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_text)
            category_scores[category] = score

        # Return highest scoring category, default to ANALYTICAL
        best_category = max(category_scores.items(), key=lambda x: x[1])
        return best_category[0] if best_category[1] > 0 else ModelCategory.ANALYTICAL

    def _extract_application_context(self, element: Dict[str, Any]) -> List[str]:
        """Extract application context from element"""
        contexts = []

        # Look for various context indicators
        context_fields = [
            "application_context",
            "use_cases",
            "contexts",
            "applications",
            "domains",
        ]
        for field in context_fields:
            value = element.get(field, [])
            if isinstance(value, list):
                contexts.extend(value)
            elif isinstance(value, str) and value:
                contexts.append(value)

        # Extract from description if no explicit context
        if not contexts:
            description = element.get("description", "")
            if "strategy" in description.lower():
                contexts.append("Strategic Planning")
            if "analysis" in description.lower():
                contexts.append("Analysis")
            if "decision" in description.lower():
                contexts.append("Decision Making")

        return contexts[:5] if contexts else ["General Problem Solving"]

    def _extract_key_concepts(self, element: Dict[str, Any]) -> List[str]:
        """Extract key concepts from element"""
        concepts = []

        # Look for concepts in various fields
        concept_fields = [
            "key_concepts",
            "concepts",
            "principles",
            "core_ideas",
            "fundamentals",
        ]
        for field in concept_fields:
            value = element.get(field, [])
            if isinstance(value, list):
                concepts.extend(str(v) for v in value)
            elif isinstance(value, str) and value:
                concepts.append(value)

        # Extract from nested structures
        if "knowledge_representation" in element:
            kr = element["knowledge_representation"]
            if "concepts" in kr:
                concepts.extend(kr["concepts"])

        # Default concepts if none found
        if not concepts:
            concepts = ["Core principle", "Key insight", "Application method"]

        return concepts[:8]  # Limit to most important concepts

    def _extract_implementation_steps(self, element: Dict[str, Any]) -> List[str]:
        """Extract implementation steps from element"""
        steps = []

        # Look for steps in various fields
        step_fields = [
            "implementation_steps",
            "steps",
            "process",
            "methodology",
            "approach",
        ]
        for field in step_fields:
            value = element.get(field, [])
            if isinstance(value, list):
                steps.extend(str(v) for v in value)
            elif isinstance(value, str) and value:
                steps.extend(value.split("."))  # Split by periods

        # Generate default steps if none found
        if not steps:
            steps = [
                "Identify the problem context",
                "Apply the mental model framework",
                "Analyze using key concepts",
                "Generate insights and conclusions",
                "Validate results and iterate",
            ]

        return [step.strip() for step in steps[:6] if step.strip()]  # Limit to 6 steps

    def _extract_bias_vulnerabilities(self, element: Dict[str, Any]) -> List[str]:
        """Extract bias vulnerabilities from element"""
        biases = []

        # Look for bias information
        bias_fields = [
            "bias_vulnerabilities",
            "biases",
            "cognitive_biases",
            "limitations",
            "pitfalls",
        ]
        for field in bias_fields:
            value = element.get(field, [])
            if isinstance(value, list):
                biases.extend(str(v) for v in value)
            elif isinstance(value, str) and value:
                biases.append(value)

        # Default biases if none found
        if not biases:
            biases = ["Confirmation bias", "Anchoring bias", "Availability heuristic"]

        return biases[:5]

    def _extract_synergy_models(self, element: Dict[str, Any]) -> List[str]:
        """Extract synergistic models from element"""
        synergies = []

        # Look for synergy information
        synergy_fields = [
            "synergy_models",
            "complements",
            "works_with",
            "enhances",
            "combines_with",
        ]
        for field in synergy_fields:
            value = element.get(field, [])
            if isinstance(value, list):
                synergies.extend(str(v) for v in value)
            elif isinstance(value, str) and value:
                synergies.append(value)

        return synergies[:4]

    def _extract_conflict_models(self, element: Dict[str, Any]) -> List[str]:
        """Extract conflicting models from element"""
        conflicts = []

        # Look for conflict information
        conflict_fields = ["conflict_models", "conflicts", "contradicts", "opposes"]
        for field in conflict_fields:
            value = element.get(field, [])
            if isinstance(value, list):
                conflicts.extend(str(v) for v in value)
            elif isinstance(value, str) and value:
                conflicts.append(value)

        return conflicts[:3]

    def _extract_ethical_considerations(self, element: Dict[str, Any]) -> List[str]:
        """Extract ethical considerations from element"""
        ethics = []

        # Look for ethics information
        ethics_fields = [
            "ethical_considerations",
            "ethics",
            "moral_implications",
            "values",
        ]
        for field in ethics_fields:
            value = element.get(field, [])
            if isinstance(value, list):
                ethics.extend(str(v) for v in value)
            elif isinstance(value, str) and value:
                ethics.append(value)

        # Default ethics if none found
        if not ethics:
            ethics = [
                "Consider stakeholder impact",
                "Maintain intellectual honesty",
                "Respect diverse perspectives",
            ]

        return ethics[:4]

    def _extract_prompt_integration_guide(self, element: Dict[str, Any]) -> str:
        """Extract prompt integration guide from element"""
        # Look for prompt integration info
        guide_fields = [
            "prompt_integration_guide",
            "prompt_guide",
            "integration_instructions",
            "usage_guide",
        ]
        for field in guide_fields:
            value = element.get(field, "")
            if value:
                return str(value)

        # Generate default guide
        name = element.get("name", "this mental model")
        return f"To integrate {name}, first establish the context, then apply the key concepts systematically, and validate insights against the framework principles."

    def _extract_effectiveness_metrics(
        self, element: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract effectiveness metrics from element"""
        metrics = {}

        # Look for metrics
        metrics_fields = ["effectiveness_metrics", "metrics", "performance", "scores"]
        for field in metrics_fields:
            value = element.get(field, {})
            if isinstance(value, dict):
                for k, v in value.items():
                    try:
                        metrics[k] = float(v)
                    except (ValueError, TypeError):
                        pass

        # Default metrics if none found
        if not metrics:
            metrics = {"clarity": 0.7, "applicability": 0.8, "reliability": 0.75}

        return metrics

    async def generate_agent_files(self):
        """Generate agent-ready prompt files"""
        self.logger.info(
            f"📝 Generating {len(self.extracted_models)} agent-ready prompt files..."
        )

        for model_id, model in self.extracted_models.items():
            # Create individual prompt file
            prompt_file = self.agent_prompts_dir / f"{model_id}.txt"
            prompt_file.write_text(model.to_agent_prompt(), encoding="utf-8")

            # Create structured JSON file
            json_file = self.output_dir / f"{model_id}.json"
            json_file.write_text(model.to_json(), encoding="utf-8")

        # Create master index file
        await self.generate_master_index()

        self.logger.info(f"✅ Generated {len(self.extracted_models)} agent-ready files")

    async def generate_master_index(self):
        """Generate master index of all extracted models"""
        index = {
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(self.extracted_models),
                "statistics": self.extraction_statistics,
            },
            "models_by_category": {},
            "models_by_id": {},
        }

        # Group by category
        for model in self.extracted_models.values():
            category = model.category.value
            if category not in index["models_by_category"]:
                index["models_by_category"][category] = []

            index["models_by_category"][category].append(
                {
                    "model_id": model.model_id,
                    "name": model.name,
                    "description": model.description,
                    "key_concepts_count": len(model.key_concepts),
                    "implementation_steps_count": len(model.implementation_steps),
                }
            )

            # Add to ID index
            index["models_by_id"][model.model_id] = {
                "name": model.name,
                "category": category,
                "prompt_file": f"src/agents/mental_model_prompts/{model.model_id}.txt",
                "json_file": f"src/intelligence/extracted_models/{model.model_id}.json",
            }

        # Write master index
        index_file = self.output_dir / "mental_models_index.json"
        index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")

        self.logger.info(
            f"📊 Generated master index with {len(self.extracted_models)} models"
        )

    async def generate_extraction_report(self):
        """Generate comprehensive extraction report"""
        report = {
            "extraction_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_files_processed": self.extraction_statistics[
                    "total_files_processed"
                ],
                "successful_extractions": self.extraction_statistics[
                    "successful_extractions"
                ],
                "failed_extractions": self.extraction_statistics["failed_extractions"],
                "success_rate": (
                    self.extraction_statistics["successful_extractions"]
                    / max(self.extraction_statistics["total_files_processed"], 1)
                    * 100
                ),
            },
            "content_analysis": {
                "models_by_category": dict(
                    self.extraction_statistics["models_by_category"]
                ),
                "total_concepts_extracted": self.extraction_statistics[
                    "total_concepts_extracted"
                ],
                "total_implementation_steps": self.extraction_statistics[
                    "total_implementation_steps"
                ],
                "average_concepts_per_model": (
                    self.extraction_statistics["total_concepts_extracted"]
                    / max(len(self.extracted_models), 1)
                ),
                "average_steps_per_model": (
                    self.extraction_statistics["total_implementation_steps"]
                    / max(len(self.extracted_models), 1)
                ),
            },
            "output_files": {
                "agent_prompts_directory": str(self.agent_prompts_dir),
                "structured_models_directory": str(self.output_dir),
                "master_index_file": str(self.output_dir / "mental_models_index.json"),
            },
            "models_extracted": list(self.extracted_models.keys()),
        }

        # Write extraction report
        report_file = self.output_dir / "extraction_report.json"
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

        self.logger.info(f"📋 Generated extraction report: {report_file}")

        return report


# Global extractor instance
_mental_model_extractor: Optional[MentalModelExtractor] = None


async def get_mental_model_extractor(source_dir: str = "db") -> MentalModelExtractor:
    """Get global mental model extractor instance"""
    global _mental_model_extractor
    if _mental_model_extractor is None:
        _mental_model_extractor = MentalModelExtractor(source_dir=source_dir)
    return _mental_model_extractor
