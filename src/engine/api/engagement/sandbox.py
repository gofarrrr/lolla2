"""
What-If Sandbox functionality for METIS Engagement API
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4

from fastapi import HTTPException
from cachetools import TTLCache

from .models import ReevaluationRequest

try:
    from src.engine.models.data_contracts import (
        MetisDataContract,
        EngagementPhase as ContractEngagementPhase,
    )

    CONTRACTS_AVAILABLE = True
except ImportError:
    CONTRACTS_AVAILABLE = False
    MetisDataContract = None
    ContractEngagementPhase = None


class WhatIfSandbox:
    """What-If analysis sandbox for re-evaluating engagements with changed assumptions"""

    # Operation Crystal Day 1: Phase-specific average costs (USD)
    PHASE_AVG_COSTS = {
        "problem_structuring": 0.02,  # Light LLM usage for structuring
        "hypothesis_generation": 0.03,  # Medium LLM usage for hypothesis creation
        "analysis_execution": 0.04,  # Heavy LLM usage + research
        "synthesis_delivery": 0.03,  # Medium LLM usage for synthesis
        "perplexity_research": 0.01,  # Per-phase research API cost
    }

    # Operation Crystal Day 2: Explicit dependency map (replaces keyword heuristics)
    ASSUMPTION_PHASE_DEPENDENCIES = {
        # Financial assumptions
        "budget": ["analysis_execution", "synthesis_delivery"],
        "cost": ["analysis_execution", "synthesis_delivery"],
        "revenue": [
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "price": ["analysis_execution", "synthesis_delivery"],
        "roi": ["analysis_execution", "synthesis_delivery"],
        "margin": ["analysis_execution", "synthesis_delivery"],
        "funding": ["problem_structuring", "analysis_execution", "synthesis_delivery"],
        # Market assumptions
        "market": ["hypothesis_generation", "analysis_execution", "synthesis_delivery"],
        "market_size": [
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "market_growth": [
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "market_growth_rate": [
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "competition": [
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "competitive_landscape": [
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "customer": [
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "customer_base": [
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "demand": ["hypothesis_generation", "analysis_execution", "synthesis_delivery"],
        # Strategic assumptions
        "strategy": [
            "problem_structuring",
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "strategic_direction": [
            "problem_structuring",
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "timeline": [
            "problem_structuring",
            "hypothesis_generation",
            "synthesis_delivery",
        ],
        "schedule": [
            "problem_structuring",
            "hypothesis_generation",
            "synthesis_delivery",
        ],
        "deadline": ["problem_structuring", "synthesis_delivery"],
        "resources": ["problem_structuring", "analysis_execution"],
        "resource_allocation": ["problem_structuring", "analysis_execution"],
        "capability": ["problem_structuring", "analysis_execution"],
        "capabilities": ["problem_structuring", "analysis_execution"],
        "team_size": ["problem_structuring", "analysis_execution"],
        "headcount": ["problem_structuring", "analysis_execution"],
        # Scope assumptions
        "scope": [
            "problem_structuring",
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "project_scope": [
            "problem_structuring",
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ],
        "boundaries": ["problem_structuring", "hypothesis_generation"],
        "requirements": ["problem_structuring", "hypothesis_generation"],
        "constraints": ["problem_structuring", "analysis_execution"],
        "limitations": ["problem_structuring", "analysis_execution"],
        # Operational assumptions
        "process": ["analysis_execution", "synthesis_delivery"],
        "operations": ["analysis_execution", "synthesis_delivery"],
        "technology": ["analysis_execution", "synthesis_delivery"],
        "infrastructure": ["analysis_execution", "synthesis_delivery"],
        "implementation": ["analysis_execution", "synthesis_delivery"],
        "execution": ["analysis_execution", "synthesis_delivery"],
    }

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)

        # Operation Crystal Day 1: TTL cache for temporary What-If forks
        # 24-hour TTL (86400 seconds), max 1000 forks in memory
        self.temp_forks = TTLCache(maxsize=1000, ttl=86400)
        self._fork_metrics = {"created": 0, "expired": 0, "active": 0}

        # Operation Crystal Day 2: Re-run limits per engagement
        self.max_reruns_per_engagement = 10  # Hard limit per parent engagement

        self.logger.info(
            "🚀 WhatIfSandbox initialized with TTL cache (24h expiry, 1000 max)"
        )
        self.logger.info(
            f"💰 Cost estimation enabled - avg costs: {self.PHASE_AVG_COSTS}"
        )
        self.logger.info(
            f"🚧 Re-run limits enabled: {self.max_reruns_per_engagement} forks per engagement"
        )

    async def reevaluate_engagement(
        self, engagement_id: UUID, changes: Dict[str, Any]
    ) -> MetisDataContract:
        """
        Re-evaluate engagement with changed assumptions.

        This method:
        1. Loads the completed engagement contract
        2. Modifies the specified assumptions
        3. Intelligently determines which phases need re-execution
        4. Re-runs only the affected downstream phases
        5. Returns updated contract with revised analysis
        """
        if engagement_id not in self.orchestrator.contracts:
            raise HTTPException(status_code=404, detail="Engagement not found")

        original_contract = self.orchestrator.contracts[engagement_id]

        # Verify engagement is completed
        if len(original_contract.workflow_state.phase_results) < 4:
            raise HTTPException(
                status_code=400,
                detail="Cannot re-evaluate incomplete engagement. Complete all phases first.",
            )

        # Operation Crystal Day 2: Check re-run limits
        current_rerun_count = self._get_rerun_count(engagement_id)
        if current_rerun_count >= self.max_reruns_per_engagement:
            raise HTTPException(
                status_code=429,  # Too Many Requests
                detail={
                    "error": "Re-run limit exceeded",
                    "max_reruns": self.max_reruns_per_engagement,
                    "current_count": current_rerun_count,
                    "message": f"Maximum {self.max_reruns_per_engagement} What-If scenarios allowed per engagement",
                },
            )

        self.logger.info(f"🔄 Starting re-evaluation for engagement {engagement_id}")
        self.logger.info(f"📝 Changes requested: {changes}")

        # Create modified contract
        modified_contract = self._create_modified_contract(original_contract, changes)

        # Determine which phases need re-execution
        phases_to_rerun = self._determine_affected_phases(changes)
        self.logger.info(f"🎯 Phases to re-run: {phases_to_rerun}")

        # Operation Crystal Day 1: Generate cost estimate before re-execution
        cost_estimate = self._estimate_rerun_cost(phases_to_rerun)
        self.logger.info(
            f"💰 Re-run cost estimate: ${cost_estimate['total_cost_usd']:.2f}"
        )

        # Re-execute affected phases
        if self.orchestrator.workflow_orchestrator and CONTRACTS_AVAILABLE:
            # Use selective re-execution
            updated_contract = (
                await self.orchestrator.workflow_orchestrator.rerun_phases(
                    modified_contract, phases_to_rerun
                )
            )
        else:
            # Simulation mode
            updated_contract = await self._simulate_reevaluation(
                modified_contract, phases_to_rerun
            )

        # Store the updated contract in TTL cache (24-hour expiry)
        temp_id = UUID(
            str(engagement_id).replace("-", "0", 1)
        )  # Simple temp ID generation

        # Operation Crystal Day 1: Store forks in TTL cache instead of orchestrator contracts
        client_name = self.orchestrator.client_names.get(engagement_id, "Unknown")
        self.temp_forks[temp_id] = {
            "contract": updated_contract,
            "client_name": client_name,
            "parent_id": engagement_id,
            "created_at": datetime.utcnow().isoformat(),
            "changes_applied": changes,
            "cost_estimate": cost_estimate,  # Include cost estimate in fork metadata
        }

        # Operation Crystal Day 2: Increment rerun counter for the parent engagement
        self._increment_rerun_count(engagement_id)

        # Update metrics
        self._fork_metrics["created"] += 1
        self._fork_metrics["active"] = len(self.temp_forks)
        new_rerun_count = self._get_rerun_count(engagement_id)

        self.logger.info(
            f"✅ Re-evaluation completed. Temporary fork {temp_id} stored in TTL cache"
        )
        self.logger.info(
            f"📊 Fork metrics - Active: {self._fork_metrics['active']}, Created: {self._fork_metrics['created']}"
        )
        self.logger.info(
            f"🔢 Parent engagement {engagement_id} now has {new_rerun_count}/{self.max_reruns_per_engagement} forks"
        )

        return updated_contract

    async def process_reevaluation_request(
        self, engagement_id: UUID, request: ReevaluationRequest
    ) -> Dict[str, Any]:
        """Process a structured re-evaluation request"""
        changes = {
            request.assumption_id: {
                "new_value": request.new_value,
                "context": request.assumption_context,
            }
        }

        updated_contract = await self.reevaluate_engagement(engagement_id, changes)

        # Get the cost estimate from the stored fork
        temp_id = UUID(str(engagement_id).replace("-", "0", 1))
        cost_estimate = self.temp_forks.get(temp_id, {}).get("cost_estimate", {})

        return {
            "original_engagement_id": str(engagement_id),
            "reevaluated_contract": (
                updated_contract.to_cloudevents_dict()
                if hasattr(updated_contract, "to_cloudevents_dict")
                else {}
            ),
            "changes_applied": changes,
            "timestamp": datetime.utcnow().isoformat(),
            "cost_estimate": cost_estimate,  # Include cost estimate in response
        }

    def _create_modified_contract(
        self, original_contract: MetisDataContract, changes: Dict[str, Any]
    ) -> MetisDataContract:
        """Create a modified contract with the requested changes"""
        if not CONTRACTS_AVAILABLE:
            # Return mock contract in simulation mode
            return original_contract

        # Deep copy the original contract
        import copy

        modified_contract = copy.deepcopy(original_contract)

        # Apply changes to business context
        for assumption_id, change_data in changes.items():
            if isinstance(change_data, dict):
                new_value = change_data.get("new_value")
                context = change_data.get("context", "")
            else:
                new_value = change_data
                context = ""

            # Store the change in business context
            if (
                "assumptions"
                not in modified_contract.engagement_context.business_context
            ):
                modified_contract.engagement_context.business_context["assumptions"] = (
                    {}
                )

            modified_contract.engagement_context.business_context["assumptions"][
                assumption_id
            ] = {
                "value": new_value,
                "context": context,
                "changed_at": datetime.utcnow().isoformat(),
            }

        # Mark as modified
        modified_contract.processing_metadata["reevaluated"] = True
        modified_contract.processing_metadata["reevaluation_timestamp"] = (
            datetime.utcnow().isoformat()
        )

        return modified_contract

    def _determine_affected_phases(self, changes: Dict[str, Any]) -> List[str]:
        """
        Determine which phases need re-execution based on explicit dependency map.

        Operation Crystal Day 2: Replaced fragile keyword heuristics with deterministic
        dependency mapping for reliable and predictable phase determination.
        """
        change_keys = list(changes.keys())
        affected_phases = set()

        for change_key in change_keys:
            change_key_lower = change_key.lower().strip()

            # First try exact match
            if change_key_lower in self.ASSUMPTION_PHASE_DEPENDENCIES:
                phases = self.ASSUMPTION_PHASE_DEPENDENCIES[change_key_lower]
                affected_phases.update(phases)
                self.logger.info(
                    f"🎯 Exact match for '{change_key}' affects phases: {phases}"
                )
                continue

            # Try partial matches for compound assumption names
            matched = False
            for assumption_key, phases in self.ASSUMPTION_PHASE_DEPENDENCIES.items():
                if (
                    assumption_key in change_key_lower
                    or change_key_lower in assumption_key
                ):
                    affected_phases.update(phases)
                    self.logger.info(
                        f"🎯 Partial match '{assumption_key}' for '{change_key}' affects phases: {phases}"
                    )
                    matched = True
                    break

            # Fallback for unknown assumptions (conservative approach)
            if not matched:
                fallback_phases = ["analysis_execution", "synthesis_delivery"]
                affected_phases.update(fallback_phases)
                self.logger.warning(
                    f"⚠️ Unknown assumption '{change_key}', using fallback phases: {fallback_phases}"
                )

        final_phases = list(affected_phases)
        self.logger.info(f"📋 Total affected phases: {final_phases}")
        return final_phases

    async def _simulate_reevaluation(
        self, contract: MetisDataContract, phases_to_rerun: List[str]
    ) -> MetisDataContract:
        """Simulate re-evaluation when workflow orchestrator is not available"""
        for phase_name in phases_to_rerun:
            # Update the phase result to indicate re-evaluation
            if phase_name in contract.workflow_state.phase_results:
                contract.workflow_state.phase_results[phase_name]["reevaluated"] = True
                contract.workflow_state.phase_results[phase_name][
                    "reevaluation_timestamp"
                ] = datetime.utcnow().isoformat()
                contract.workflow_state.phase_results[phase_name]["result"][
                    "simulated_reevaluation"
                ] = True

        return contract

    def get_temp_fork(self, temp_id: UUID) -> Dict[str, Any]:
        """Retrieve a temporary What-If fork from TTL cache"""
        fork_data = self.temp_forks.get(temp_id)
        if not fork_data:
            raise HTTPException(
                status_code=404, detail=f"Temporary fork {temp_id} not found or expired"
            )

        return fork_data

    def get_temp_fork_contract(self, temp_id: UUID) -> MetisDataContract:
        """Get the contract from a temporary fork"""
        fork_data = self.get_temp_fork(temp_id)
        return fork_data["contract"]

    def list_active_forks(
        self, parent_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        """List all active temporary forks, optionally filtered by parent engagement"""
        active_forks = []

        for fork_id, fork_data in self.temp_forks.items():
            if parent_id is None or fork_data["parent_id"] == parent_id:
                active_forks.append(
                    {
                        "temp_id": fork_id,
                        "parent_id": fork_data["parent_id"],
                        "client_name": fork_data["client_name"],
                        "created_at": fork_data["created_at"],
                        "changes_summary": list(fork_data["changes_applied"].keys()),
                    }
                )

        return active_forks

    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get TTL cache metrics for monitoring"""
        # Update active count (some may have expired)
        self._fork_metrics["active"] = len(self.temp_forks)

        return {
            "in_memory_fork_count": self._fork_metrics["active"],
            "total_forks_created": self._fork_metrics["created"],
            "cache_maxsize": self.temp_forks.maxsize,
            "cache_ttl_hours": self.temp_forks.ttl / 3600,
            "cache_utilization": len(self.temp_forks) / self.temp_forks.maxsize,
        }

    def cleanup_expired_forks(self) -> int:
        """Force cleanup of expired forks and return count cleaned"""
        initial_count = len(self.temp_forks)

        # TTLCache automatically removes expired items, but we can trigger cleanup
        # by accessing cache info or iterating (which triggers cleanup)
        list(self.temp_forks.keys())  # This triggers internal cleanup

        final_count = len(self.temp_forks)
        cleaned_count = initial_count - final_count

        if cleaned_count > 0:
            self._fork_metrics["expired"] += cleaned_count
            self._fork_metrics["active"] = final_count
            self.logger.info(f"🧹 Cleaned up {cleaned_count} expired What-If forks")

        return cleaned_count

    def _estimate_rerun_cost(self, phases_to_rerun: List[str]) -> Dict[str, Any]:
        """
        Estimate the cost of re-running specified phases.

        Args:
            phases_to_rerun: List of phase names that need re-execution

        Returns:
            Cost breakdown dictionary with phase costs and total
        """
        cost_breakdown = {}
        total_cost = 0.0

        for phase in phases_to_rerun:
            # Base phase cost
            phase_cost = self.PHASE_AVG_COSTS.get(phase, 0.025)  # Default fallback

            # Add research cost for phases that typically use external research
            if phase in ["hypothesis_generation", "analysis_execution"]:
                phase_cost += self.PHASE_AVG_COSTS["perplexity_research"]

            cost_breakdown[phase] = round(phase_cost, 3)
            total_cost += phase_cost

        # Round total to 2 decimal places for currency display
        total_cost = round(total_cost, 2)

        result = {
            "phase_costs": cost_breakdown,
            "total_cost_usd": total_cost,
            "phases_count": len(phases_to_rerun),
            "estimated_at": datetime.utcnow().isoformat(),
            "currency": "USD",
            "note": "Estimates based on average LLM and research API usage per phase",
        }

        self.logger.info(
            f"💰 Cost estimate: ${total_cost:.2f} for {len(phases_to_rerun)} phases"
        )
        return result

    async def get_rerun_cost_estimate(
        self, engagement_id: UUID, changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get cost estimate for re-running engagement with changes WITHOUT actually executing.

        Args:
            engagement_id: Original engagement ID
            changes: Dictionary of changes to be applied

        Returns:
            Cost estimation details
        """
        if engagement_id not in self.orchestrator.contracts:
            raise HTTPException(status_code=404, detail="Engagement not found")

        original_contract = self.orchestrator.contracts[engagement_id]

        # Verify engagement is completed
        if len(original_contract.workflow_state.phase_results) < 4:
            raise HTTPException(
                status_code=400,
                detail="Cannot estimate cost for incomplete engagement. Complete all phases first.",
            )

        # Determine which phases would be affected (same logic as actual rerun)
        phases_to_rerun = self._determine_affected_phases(changes)

        # Generate cost estimate
        cost_estimate = self._estimate_rerun_cost(phases_to_rerun)

        # Add context about the changes
        cost_estimate["changes_summary"] = {
            "assumptions_modified": list(changes.keys()),
            "affected_phases": phases_to_rerun,
        }

        self.logger.info(
            f"💰 Generated cost estimate for engagement {engagement_id}: ${cost_estimate['total_cost_usd']:.2f}"
        )

        return cost_estimate

    def get_dependency_info(self) -> Dict[str, Any]:
        """
        Get information about assumption-phase dependencies for debugging/monitoring.

        Returns summary statistics about the dependency map.
        """
        dependencies = self.ASSUMPTION_PHASE_DEPENDENCIES

        # Calculate statistics
        total_assumptions = len(dependencies)

        phase_frequency = {}
        for phases in dependencies.values():
            for phase in phases:
                phase_frequency[phase] = phase_frequency.get(phase, 0) + 1

        # Find most commonly affected phases
        most_affected_phases = sorted(
            phase_frequency.items(), key=lambda x: x[1], reverse=True
        )

        # Group by category
        categories = {
            "financial": [
                "budget",
                "cost",
                "revenue",
                "price",
                "roi",
                "margin",
                "funding",
            ],
            "market": ["market", "competition", "customer", "demand"],
            "strategic": ["strategy", "timeline", "resources", "capability"],
            "scope": ["scope", "boundaries", "requirements", "constraints"],
            "operational": ["process", "operations", "technology", "infrastructure"],
        }

        category_counts = {}
        for category, keywords in categories.items():
            count = sum(
                1
                for assumption in dependencies.keys()
                if any(keyword in assumption for keyword in keywords)
            )
            category_counts[category] = count

        return {
            "dependency_map_stats": {
                "total_assumptions_mapped": total_assumptions,
                "category_breakdown": category_counts,
                "phase_frequency": dict(most_affected_phases),
                "most_affected_phase": (
                    most_affected_phases[0][0] if most_affected_phases else None
                ),
            },
            "sample_mappings": {
                "financial_example": dependencies.get("budget", []),
                "market_example": dependencies.get("market_growth_rate", []),
                "strategic_example": dependencies.get("timeline", []),
            },
            "mapping_deterministic": True,
            "last_updated": "Operation Crystal Day 2",
        }

    def _get_rerun_count(self, engagement_id: UUID) -> int:
        """
        Get the current rerun count for an engagement.

        Operation Crystal Day 2: Tracks rerun count in engagement metadata.
        """
        if engagement_id not in self.orchestrator.contracts:
            return 0

        contract = self.orchestrator.contracts[engagement_id]
        return contract.processing_metadata.get("rerun_count", 0)

    def _increment_rerun_count(self, engagement_id: UUID):
        """
        Increment the rerun count for an engagement.

        Operation Crystal Day 2: Hard limit enforcement per parent engagement.
        """
        if engagement_id not in self.orchestrator.contracts:
            self.logger.warning(
                f"⚠️ Cannot increment rerun count - engagement {engagement_id} not found"
            )
            return

        contract = self.orchestrator.contracts[engagement_id]
        current_count = contract.processing_metadata.get("rerun_count", 0)
        new_count = current_count + 1

        contract.processing_metadata["rerun_count"] = new_count
        contract.processing_metadata["last_rerun_at"] = datetime.utcnow().isoformat()

        if new_count >= self.max_reruns_per_engagement * 0.8:  # 80% warning threshold
            self.logger.warning(
                f"⚠️ Engagement {engagement_id} approaching rerun limit: {new_count}/{self.max_reruns_per_engagement}"
            )

        self.logger.info(
            f"🔢 Incremented rerun count for {engagement_id}: {current_count} → {new_count}"
        )

    def get_rerun_status(self, engagement_id: UUID) -> Dict[str, Any]:
        """
        Get rerun status and limits for an engagement.

        Useful for frontend UI to show limits and warnings.
        """
        if engagement_id not in self.orchestrator.contracts:
            return {
                "error": "Engagement not found",
                "current_count": 0,
                "max_reruns": self.max_reruns_per_engagement,
                "remaining": self.max_reruns_per_engagement,
            }

        current_count = self._get_rerun_count(engagement_id)
        remaining = max(0, self.max_reruns_per_engagement - current_count)

        contract = self.orchestrator.contracts[engagement_id]
        last_rerun = contract.processing_metadata.get("last_rerun_at")

        # Determine status
        if current_count >= self.max_reruns_per_engagement:
            status = "limit_exceeded"
        elif current_count >= self.max_reruns_per_engagement * 0.8:
            status = "approaching_limit"
        elif current_count >= self.max_reruns_per_engagement * 0.5:
            status = "moderate_usage"
        else:
            status = "low_usage"

        return {
            "engagement_id": str(engagement_id),
            "current_count": current_count,
            "max_reruns": self.max_reruns_per_engagement,
            "remaining": remaining,
            "status": status,
            "utilization_percent": round(
                (current_count / self.max_reruns_per_engagement) * 100, 1
            ),
            "last_rerun_at": last_rerun,
            "warnings": self._generate_rerun_warnings(current_count),
        }

    def _generate_rerun_warnings(self, current_count: int) -> List[str]:
        """Generate user-friendly warnings about rerun limits"""
        warnings = []

        if current_count >= self.max_reruns_per_engagement:
            warnings.append("Maximum What-If scenarios reached for this engagement")
        elif current_count >= self.max_reruns_per_engagement * 0.9:
            warnings.append(
                f"Only {self.max_reruns_per_engagement - current_count} What-If scenarios remaining"
            )
        elif current_count >= self.max_reruns_per_engagement * 0.8:
            warnings.append("Approaching What-If scenario limit")

        return warnings

    def reset_rerun_count(
        self, engagement_id: UUID, admin_override: bool = False
    ) -> Dict[str, Any]:
        """
        Reset rerun count for an engagement (admin function).

        Should be used sparingly and logged for audit purposes.
        """
        if not admin_override:
            raise HTTPException(
                status_code=403, detail="Rerun count reset requires admin privileges"
            )

        if engagement_id not in self.orchestrator.contracts:
            raise HTTPException(status_code=404, detail="Engagement not found")

        contract = self.orchestrator.contracts[engagement_id]
        old_count = contract.processing_metadata.get("rerun_count", 0)

        contract.processing_metadata["rerun_count"] = 0
        contract.processing_metadata["rerun_count_reset_at"] = (
            datetime.utcnow().isoformat()
        )
        contract.processing_metadata["rerun_count_reset_reason"] = "admin_override"

        self.logger.warning(
            f"🔄 ADMIN RESET: Rerun count for {engagement_id} reset from {old_count} to 0"
        )

        return {
            "engagement_id": str(engagement_id),
            "old_count": old_count,
            "new_count": 0,
            "reset_at": contract.processing_metadata["rerun_count_reset_at"],
            "reset_by": "admin_override",
        }

    async def promote_scenario(
        self, temp_id: UUID, scenario_name: str
    ) -> Dict[str, Any]:
        """
        Promote a temporary What-If fork to a permanent engagement.

        Operation Crystal Day 3: Allows users to save valuable scenarios permanently.

        Args:
            temp_id: UUID of the temporary fork to promote
            scenario_name: User-provided name for the scenario

        Returns:
            Details of the newly promoted permanent engagement
        """
        # Validate temp fork exists
        fork_data = self.get_temp_fork(temp_id)

        # Extract fork information
        temp_contract = fork_data["contract"]
        parent_id = fork_data["parent_id"]
        client_name = fork_data["client_name"]
        changes_applied = fork_data["changes_applied"]
        cost_estimate = fork_data.get("cost_estimate", {})

        # Generate new permanent engagement ID
        permanent_id = uuid4()

        self.logger.info(
            f"🎯 Promoting What-If scenario {temp_id} to permanent engagement {permanent_id}"
        )
        self.logger.info(f"📝 Scenario name: '{scenario_name}'")

        # Create permanent contract with promotion metadata
        promoted_contract = self._create_promoted_contract(
            temp_contract, permanent_id, parent_id, scenario_name, changes_applied
        )

        # Store in orchestrator as permanent engagement
        self.orchestrator.contracts[permanent_id] = promoted_contract
        self.orchestrator.client_names[permanent_id] = (
            f"{client_name} - {scenario_name}"
        )

        # Store in state manager if available
        if self.orchestrator.state_manager:
            try:
                await self.orchestrator.state_manager.set_state(
                    f"engagement_{permanent_id}",
                    promoted_contract.to_cloudevents_dict(),
                    # Assuming StateType.ENGAGEMENT exists
                    "ENGAGEMENT",
                )
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Failed to store promoted scenario in state manager: {e}"
                )

        # Optional: Remove the temporary fork (user's choice)
        # We'll keep it for now to maintain audit trail

        # Prepare response
        promotion_details = {
            "promoted_engagement_id": str(permanent_id),
            "original_temp_id": str(temp_id),
            "parent_engagement_id": str(parent_id),
            "scenario_name": scenario_name,
            "client_name": f"{client_name} - {scenario_name}",
            "promoted_at": datetime.utcnow().isoformat(),
            "changes_applied": changes_applied,
            "cost_estimate": cost_estimate,
            "contract_preview": {
                "problem_statement": promoted_contract.engagement_context.problem_statement,
                "phase_count": len(promoted_contract.workflow_state.phase_results),
                "overall_confidence": getattr(
                    promoted_contract, "overall_confidence", 0.0
                ),
            },
        }

        self.logger.info(
            f"✅ Successfully promoted scenario '{scenario_name}' to permanent engagement {permanent_id}"
        )

        return promotion_details

    def _create_promoted_contract(
        self,
        temp_contract,
        permanent_id: UUID,
        parent_id: UUID,
        scenario_name: str,
        changes_applied: Dict[str, Any],
    ):
        """Create a new contract for the promoted scenario"""
        import copy

        # Deep copy the temporary contract
        promoted_contract = copy.deepcopy(temp_contract)

        # Update with permanent information
        promoted_contract.engagement_context.engagement_id = permanent_id

        # Add promotion metadata
        promoted_contract.processing_metadata.update(
            {
                "promoted_from_temp": True,
                "promoted_at": datetime.utcnow().isoformat(),
                "parent_engagement_id": str(parent_id),
                "scenario_name": scenario_name,
                "original_temp_id": str(temp_contract.engagement_context.engagement_id),
                "promotion_changes": changes_applied,
            }
        )

        # Update problem statement to reflect scenario
        original_problem = promoted_contract.engagement_context.problem_statement
        promoted_contract.engagement_context.problem_statement = (
            f"{original_problem} [{scenario_name}]"
        )

        # Add scenario context to business context
        if hasattr(promoted_contract.engagement_context, "business_context"):
            promoted_contract.engagement_context.business_context.update(
                {
                    "scenario_type": "what_if_promotion",
                    "scenario_name": scenario_name,
                    "parent_engagement": str(parent_id),
                    "scenario_changes": changes_applied,
                }
            )

        return promoted_contract

    def list_promoted_scenarios(
        self, parent_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        """
        List all promoted scenarios, optionally filtered by parent engagement.

        Args:
            parent_id: Optional parent engagement to filter by

        Returns:
            List of promoted scenario details
        """
        promoted_scenarios = []

        for engagement_id, contract in self.orchestrator.contracts.items():
            # Check if this is a promoted scenario
            if contract.processing_metadata.get("promoted_from_temp", False):
                scenario_parent_id = contract.processing_metadata.get(
                    "parent_engagement_id"
                )

                # Filter by parent if specified
                if parent_id is None or scenario_parent_id == str(parent_id):
                    scenario_info = {
                        "engagement_id": str(engagement_id),
                        "scenario_name": contract.processing_metadata.get(
                            "scenario_name"
                        ),
                        "parent_engagement_id": scenario_parent_id,
                        "promoted_at": contract.processing_metadata.get("promoted_at"),
                        "client_name": self.orchestrator.client_names.get(
                            engagement_id
                        ),
                        "problem_statement": contract.engagement_context.problem_statement,
                        "changes_applied": contract.processing_metadata.get(
                            "promotion_changes", {}
                        ),
                    }
                    promoted_scenarios.append(scenario_info)

        # Sort by promotion date (newest first)
        promoted_scenarios.sort(key=lambda x: x.get("promoted_at", ""), reverse=True)

        self.logger.info(
            f"📋 Found {len(promoted_scenarios)} promoted scenarios"
            + (f" for parent {parent_id}" if parent_id else "")
        )

        return promoted_scenarios
