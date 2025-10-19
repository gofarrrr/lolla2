"""
Mock Dispatch Orchestrator for Unblocking Pipeline Testing
==========================================================

A lightweight mock that implements the DispatchOrchestrator interface
to enable end-to-end pipeline testing without full database dependencies.
"""

import time
from typing import List, Dict, Any
from datetime import datetime, timezone

from src.orchestration.contracts import (
    StructuredAnalyticalFramework, 
    DispatchPackage,
    ConsultantBlueprint,
    NWayConfiguration,
    FrameworkType
)


class MockDispatchOrchestrator:
    """
    Mock implementation of DispatchOrchestrator interface.
    
    Provides realistic consultant selection responses to unblock pipeline testing.
    This is a temporary tactical solution for BUILD ORDER T-05.
    """
    
    def __init__(self):
        self.consultant_database = self._get_mock_consultants()
    
    async def run_dispatch(self, framework: StructuredAnalyticalFramework) -> DispatchPackage:
        """
        Mock implementation of run_dispatch that returns realistic consultant selection.
        
        Args:
            framework: StructuredAnalyticalFramework from problem structuring
            
        Returns:
            DispatchPackage with selected consultants and configuration
        """
        start_time = time.time()
        
        # Smart consultant selection based on framework type
        selected_consultants = self._select_consultants_for_framework(framework)
        
        # Create N-Way configuration
        nway_config = NWayConfiguration(
            pattern_name="strategic_trio" if len(selected_consultants) == 3 else "balanced_panel",
            consultant_cluster=selected_consultants,
            interaction_strategy="collaborative_analysis"
        )
        
        # Generate rationale
        consultant_types = [c.consultant_type for c in selected_consultants]
        rationale = f"Selected {len(selected_consultants)} consultants ({', '.join(consultant_types)}) based on {framework.framework_type.value} framework requirements"
        
        processing_time = time.time() - start_time
        
        return DispatchPackage(
            selected_consultants=selected_consultants,
            nway_configuration=nway_config,
            dispatch_rationale=rationale,
            confidence_score=0.85,  # High confidence for mock
            processing_time_seconds=processing_time,
            s2_tier="S2_TIER_2",
            s2_rationale="Mock strategic analysis tier selection",
            timestamp=datetime.now(timezone.utc)
        )
    
    def _select_consultants_for_framework(self, framework: StructuredAnalyticalFramework) -> List[ConsultantBlueprint]:
        """Select appropriate consultants based on framework type."""
        
        # Framework-specific consultant selection logic
        if framework.framework_type == FrameworkType.STRATEGIC_ANALYSIS:
            return [
                self._create_consultant("mckinsey_strategist", "Strategic planning specialist", 0.92, ["competitive_analysis", "market_positioning"]),
                self._create_consultant("market_researcher", "Market intelligence expert", 0.88, ["market_analysis", "customer_insights"]),
                self._create_consultant("financial_analyst", "Financial modeling specialist", 0.85, ["financial_projections", "investment_analysis"])
            ]
        elif framework.framework_type == FrameworkType.INNOVATION_DISCOVERY:
            return [
                self._create_consultant("innovation_specialist", "Innovation strategy expert", 0.90, ["innovation_frameworks", "technology_assessment"]),
                self._create_consultant("technical_architect", "Technical systems expert", 0.87, ["technical_architecture", "systems_design"]),
                self._create_consultant("design_thinker", "Human-centered design expert", 0.84, ["user_experience", "design_strategy"])
            ]
        elif framework.framework_type == FrameworkType.OPERATIONAL_OPTIMIZATION:
            return [
                self._create_consultant("operations_expert", "Operations optimization specialist", 0.89, ["process_optimization", "efficiency_analysis"]),
                self._create_consultant("supply_chain_analyst", "Supply chain expert", 0.86, ["logistics", "supply_optimization"]),
                self._create_consultant("quality_engineer", "Quality systems specialist", 0.83, ["quality_management", "process_control"])
            ]
        elif framework.framework_type == FrameworkType.CRISIS_MANAGEMENT:
            return [
                self._create_consultant("crisis_manager", "Crisis response specialist", 0.91, ["crisis_response", "risk_mitigation"]),
                self._create_consultant("risk_assessor", "Risk analysis expert", 0.88, ["risk_assessment", "scenario_planning"]),
                self._create_consultant("communications_strategist", "Crisis communications expert", 0.85, ["stakeholder_communications", "reputation_management"])
            ]
        else:
            # Default balanced selection
            return [
                self._create_consultant("mckinsey_strategist", "Strategic planning specialist", 0.90, ["strategy", "analysis"]),
                self._create_consultant("technical_architect", "Technical systems expert", 0.87, ["technology", "systems"]),
                self._create_consultant("operations_expert", "Operations specialist", 0.84, ["operations", "efficiency"])
            ]
    
    def _create_consultant(self, consultant_type: str, specialization: str, effectiveness: float, dimensions: List[str]) -> ConsultantBlueprint:
        """Create a ConsultantBlueprint with realistic data."""
        return ConsultantBlueprint(
            consultant_id=f"{consultant_type}_001",
            consultant_type=consultant_type,
            specialization=specialization,
            predicted_effectiveness=effectiveness,
            assigned_dimensions=dimensions
        )
    
    def _get_mock_consultants(self) -> Dict[str, Dict[str, Any]]:
        """Return mock consultant database for compatibility."""
        return {
            "mckinsey_strategist": {
                "type": "mckinsey_strategist",
                "specialization": "Strategic planning, competitive analysis, market positioning",
                "expertise_areas": ["strategy", "competition", "markets", "positioning"]
            },
            "technical_architect": {
                "type": "technical_architect", 
                "specialization": "System architecture, technical design, engineering excellence",
                "expertise_areas": ["technology", "architecture", "systems", "engineering"]
            },
            "market_researcher": {
                "type": "market_researcher",
                "specialization": "Market intelligence, customer insights, competitive research", 
                "expertise_areas": ["market_analysis", "customer_research", "competitive_intelligence"]
            },
            "financial_analyst": {
                "type": "financial_analyst",
                "specialization": "Financial modeling, investment analysis, valuation",
                "expertise_areas": ["financial_modeling", "investment_analysis", "valuation", "projections"]
            },
            "operations_expert": {
                "type": "operations_expert",
                "specialization": "Operations optimization, process improvement, efficiency",
                "expertise_areas": ["operations", "process_optimization", "efficiency", "lean"]
            }
        }