#!/usr/bin/env python3
"""
YAML Persona Loader for Method Acting Consultant Analysis
Loads and processes consultant personas from cognitive architecture YAML files
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from src.config.architecture_loader import load_full_architecture


class PersonaLoader:
    """Loads and processes consultant personas from YAML files"""

    def __init__(self, architecture_path: str = None):
        if architecture_path is None:
            architecture_path = os.path.join(
                os.path.dirname(__file__), "../../cognitive_architecture"
            )
        self.architecture_path = Path(architecture_path)
        self._persona_cache = {}

    def load_all_personas(self) -> Dict[str, Dict[str, Any]]:
        """Load all consultant personas from YAML files"""
        if self._persona_cache:
            return self._persona_cache

        personas: Dict[str, Dict[str, Any]] = {}
        try:
            master_path = self.architecture_path / "nway_cognitive_architecture.yaml"
            clusters = load_full_architecture(master_path)
            for cluster in clusters.values():
                for nway in cluster.nways:
                    for persona_name, persona_data in nway.consultant_personas.items():
                        if persona_name not in personas:
                            personas[persona_name] = dict(persona_data)
                            personas[persona_name]["source_nway"] = nway.id
                            personas[persona_name]["source_file"] = cluster.name
            self._persona_cache = personas
            return personas
        except Exception:
            # Fallback to legacy file scanning if master load failed
            clusters_path = self.architecture_path / "clusters"
            if clusters_path.exists():
                for yaml_file in clusters_path.glob("*.yaml"):
                    cluster_personas = self._load_personas_from_file(yaml_file)
                    personas.update(cluster_personas)
            
            # OPERATION UNIFICATION: Also load from consultant_personas.yaml
            consultant_personas_file = self.architecture_path / "consultants" / "consultant_personas.yaml"
            if consultant_personas_file.exists():
                direct_personas = self._load_direct_personas_file(consultant_personas_file)
                personas.update(direct_personas)
                
            self._persona_cache = personas
            return personas

    def _load_personas_from_file(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load personas from a single YAML file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            personas = {}

            # Extract personas from NWAY blocks
            for key, value in data.items():
                if key.startswith("NWAY_") and isinstance(value, dict):
                    consultant_personas = value.get("consultant_personas", {})
                    for persona_name, persona_data in consultant_personas.items():
                        if persona_name not in personas:
                            personas[persona_name] = persona_data
                            personas[persona_name]["source_nway"] = key
                            personas[persona_name]["source_file"] = file_path.name

            return personas

        except Exception as e:
            print(f"Warning: Could not load personas from {file_path}: {e}")
            return {}
    
    def _load_direct_personas_file(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load personas directly from consultant_personas.yaml file format"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            personas = {}
            
            # Extract from consultant_personas section
            consultant_personas = data.get("consultant_personas", {})
            for persona_name, persona_data in consultant_personas.items():
                # Skip non-consultant sections like 'complementary_pairs', etc.
                if isinstance(persona_data, dict) and "core_identity" in persona_data:
                    personas[persona_name] = dict(persona_data)
                    personas[persona_name]["source_nway"] = "CONSULTANT_PERSONAS_DIRECT"
                    personas[persona_name]["source_file"] = file_path.name
                    # Map fields for compatibility
                    if "core_identity" in persona_data:
                        personas[persona_name]["identity"] = persona_data["core_identity"]
            
            print(f"✅ Loaded {len(personas)} personas from {file_path.name}")
            return personas
            
        except Exception as e:
            print(f"Warning: Could not load direct personas from {file_path}: {e}")
            return {}

    def get_persona(self, consultant_type: str) -> Optional[Dict[str, Any]]:
        """Get a specific persona by consultant type"""
        personas = self.load_all_personas()
        return personas.get(consultant_type)

    def get_available_personas(self) -> List[str]:
        """Get list of available persona names"""
        personas = self.load_all_personas()
        return list(personas.keys())

    def create_method_acting_prompt(
        self,
        consultant_type: str,
        query: str,
        framework: Dict[str, Any],
        assigned_nways: List[str] = None,
    ) -> str:
        """Create a method acting prompt with dramatic stakes and job-specific excellence"""
        persona = self.get_persona(consultant_type)
        if not persona:
            return self._create_generic_prompt(consultant_type, query, framework)

        # Extract persona elements
        identity = persona.get("identity", f"A {consultant_type} consultant")
        mental_model_affinities = persona.get("mental_model_affinities", {})
        cognitive_signature = persona.get("cognitive_signature", "Professional analyst")
        blind_spots = persona.get("blind_spots", [])

        # Create dramatic stakes specific to each consultant type
        dramatic_stakes = self._get_dramatic_stakes(consultant_type, query)
        performance_excellence = self._get_performance_excellence_traits(
            consultant_type, persona
        )

        # Build enhanced method acting prompt
        method_acting_prompt = f"""🎭 EMERGENCY CONSULTANT ACTIVATION PROTOCOL 🎭

CRITICAL SITUATION: {dramatic_stakes}

IDENTITY ACTIVATION:
You ARE {identity}. This is your moment to prove why you're the absolute best in your field. Your reputation, career, and the success of this critical decision all depend on your expertise right now.

{performance_excellence}

COGNITIVE SIGNATURE: {cognitive_signature}

YOUR MENTAL MODEL MASTERY (these are your superpowers):"""

        for model, affinity in mental_model_affinities.items():
            method_acting_prompt += f"\n🎯 {model.upper()}: {affinity}"

        if assigned_nways:
            method_acting_prompt += (
                f"\n\n🧠 YOUR ASSIGNED NWAY FRAMEWORKS: {', '.join(assigned_nways)}"
            )
            method_acting_prompt += (
                "\nThese are your primary analytical weapons - use them with precision."
            )

        method_acting_prompt += f"""

🚨 CRITICAL QUERY: {query}

📊 FRAMEWORK CONTEXT: {framework.get('framework_type', 'Strategic Analysis')}

🔬 PERFORMANCE DECOMPOSITION - Execute in sequence:

STAGE 1 - RAPID PATTERN RECOGNITION (30 seconds of your best thinking):
- What patterns does your exceptional {consultant_type} experience immediately recognize?
- Which red flags are screaming at you that others would miss?
- What opportunities are hidden in plain sight?

STAGE 2 - MENTAL MODEL DEPLOYMENT (Your signature analysis):
- Deploy your mental model mastery: {', '.join(mental_model_affinities.keys())}
- What unique insights emerge when you apply your cognitive signature?
- How do your NWAY frameworks reveal critical factors others miss?

STAGE 3 - EXPERT SYNTHESIS (Your reputation-defining moment):
- What is your high-confidence recommendation?
- What are the specific risks only you would identify?
- What implementation factors will make or break this decision?

⚠️ VULNERABILITY CHECK: 
Your known blind spots: {', '.join(blind_spots) if blind_spots else 'Consider your typical biases'}
What might you be missing? (Self-awareness is your strength)

🎯 PERFORMANCE STANDARD:
This analysis must demonstrate why you're the leading {consultant_type} in your field. Show the depth of expertise that separates true masters from generic advisors.

Remember: You ARE {identity}. Lives, money, and reputations depend on getting this exactly right."""

        return method_acting_prompt

    def _get_dramatic_stakes(self, consultant_type: str, query: str) -> str:
        """Generate dramatic stakes specific to consultant type and query"""
        stakes_map = {
            "financial_analyst": "A $500M investment decision hangs in the balance. Your financial analysis will determine whether the company makes a fortune or faces bankruptcy. The CEO is counting on your expertise to save the firm.",
            "strategic_analyst": "The company's entire future strategy depends on your analysis. Wrong strategic advice could cost thousands of jobs and destroy a century-old business. The board needs your strategic insight to navigate this critical inflection point.",
            "risk_assessor": "A major catastrophic risk could destroy everything if not properly identified. Your risk assessment could prevent a disaster that would make headlines worldwide. Lives and livelihoods depend on your expertise.",
            "market_researcher": "A product launch representing 3 years of development and $100M investment needs your market insight. Get the market analysis wrong and the company loses everything. Get it right and you create the next billion-dollar product.",
            "implementation_specialist": "A critical transformation project affecting 10,000 employees is failing. Your implementation expertise is the last hope to save the project and prevent massive layoffs. The CEO has called you in as the final expert.",
            "technology_advisor": "A technology decision will determine whether the company leads the industry or becomes obsolete. Your technical analysis could either position the firm as an industry leader or condemn it to irrelevance.",
        }

        return stakes_map.get(
            consultant_type,
            f"A critical business decision depends on your {consultant_type} expertise. Failure is not an option.",
        )

    def _get_performance_excellence_traits(
        self, consultant_type: str, persona: Dict[str, Any]
    ) -> str:
        """Generate performance excellence traits specific to each consultant type"""
        excellence_map = {
            "financial_analyst": """FINANCIAL ANALYSIS MASTERY:
• You've analyzed deals worth $50B+ and never missed a hidden cost
• Your opportunity cost calculations have saved companies hundreds of millions
• Warren Buffett's margin of safety principle is embedded in your DNA
• You can smell financial BS from a mile away and always follow the money""",
            "strategic_analyst": """STRATEGIC ANALYSIS EXCELLENCE:
• You've guided Fortune 500 companies through their most complex strategic decisions
• Your first-principles thinking cuts through complexity like a laser
• You see strategic patterns that others miss and predict industry shifts
• Your systematic approach has never failed to find the root cause""",
            "risk_assessor": """RISK ASSESSMENT MASTERY:
• You've prevented catastrophic failures by identifying risks others missed
• Your engineering background gives you insights into system vulnerabilities
• You think in probabilities and cascading failures
• Your risk models have withstood every stress test imaginable""",
            "market_researcher": """MARKET RESEARCH EXPERTISE:
• You understand consumer psychology better than consumers understand themselves
• Your behavioral pattern recognition has predicted market shifts with 95% accuracy
• You see through marketing hype to real consumer needs
• Your market insights have launched billion-dollar products""",
            "implementation_specialist": """IMPLEMENTATION EXCELLENCE:
• You've successfully led 200+ organizational transformations
• Your people-first approach gets buy-in from the most resistant stakeholders
• You understand the psychology of change and human motivation
• Your execution track record is flawless - projects succeed when you lead them""",
            "technology_advisor": """TECHNOLOGY STRATEGY MASTERY:
• You've architected systems handling billions of transactions
• Your simplification approach has saved companies millions in technical debt
• You see technology patterns and abstractions that others miss
• Your technology decisions have defined industry standards""",
        }

        return excellence_map.get(
            consultant_type,
            f"You are recognized as the leading {consultant_type} expert in your field.",
        )

    def _create_generic_prompt(
        self, consultant_type: str, query: str, framework: Dict[str, Any]
    ) -> str:
        """Fallback generic prompt when persona is not found"""
        return f"""You are a {consultant_type} consultant analyzing: {query}

Framework: {framework.get('framework_type', 'Strategic Analysis')}

Please provide your analysis focusing on:
1. Key insights using your specialized knowledge
2. Risk assessment and considerations  
3. Strategic recommendations
4. Success factors and implementation considerations

Be specific and leverage your expertise in {consultant_type}."""

    def get_consultant_tools(self, consultant_type: str) -> List[str]:
        """Get recommended tools for a consultant type based on their persona"""
        persona = self.get_persona(consultant_type)
        if not persona:
            return ["general_research", "market_analysis"]

        # Map persona characteristics to tool recommendations
        tool_mapping = {
            "financial_analyst": [
                "financial_modeling",
                "market_research",
                "risk_analysis",
                "valuation_tools",
            ],
            "strategic_analyst": [
                "strategic_planning",
                "competitive_analysis",
                "scenario_planning",
                "swot_analysis",
            ],
            "risk_assessor": [
                "risk_modeling",
                "probability_analysis",
                "stress_testing",
                "monte_carlo",
            ],
            "market_researcher": [
                "market_intelligence",
                "consumer_research",
                "trend_analysis",
                "surveys",
            ],
            "technical_analyst": [
                "technical_research",
                "feasibility_studies",
                "architecture_review",
                "code_analysis",
            ],
        }

        return tool_mapping.get(consultant_type, ["general_research", "analysis_tools"])


# Singleton instance for easy access
persona_loader = PersonaLoader()
