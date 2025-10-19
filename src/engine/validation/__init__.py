#!/usr/bin/env python3
"""
METIS Validation Framework - Operation Crucible
Complete validation benchmarking system for measuring architectural superiority

This package provides the complete validation pipeline for objectively measuring
METIS multi-phase cognitive architecture against sophisticated baseline competitors.

Key Components:
- BaselineChallenger: Creates strongest possible single-prompt competitors
- BlindJudge: Provides objective, unbiased evaluation with rigorous blinding
- ValidationOrchestrator: Coordinates complete end-to-end validation runs

Usage:
    from src.validation import run_aperture_labs_validation

    # Execute single validation run
    result = await run_aperture_labs_validation()
    print(f"Winner: {result.evaluation.winner}")
    print(f"Architectural Alpha: {result.evaluation.margin:.2f}")
"""

from .challenger import (
    BaselineChallenger,
    get_baseline_challenger,
    generate_baseline_mega_prompt,
    execute_baseline_challenge,
    MegaPromptMetadata,
)

from .judge import (
    BlindJudge,
    get_blind_judge,
    evaluate_metis_vs_baseline,
    EvaluationResult,
    EvaluationCriterion,
    BlindedOutput,
)

from .validation_orchestrator import (
    ValidationOrchestrator,
    get_validation_orchestrator,
    run_single_validation,
    run_aperture_labs_validation,
    ValidationScenario,
    ValidationRun,
    ValidationReport,
)

# Version information
__version__ = "1.0.0"
__author__ = "METIS AI Platform Team"
__description__ = (
    "Complete validation benchmarking framework for cognitive architecture evaluation"
)

# Package metadata
VALIDATION_FRAMEWORK_INFO = {
    "name": "METIS Validation Framework",
    "version": __version__,
    "components": {
        "BaselineChallenger": "Sophisticated single-prompt competitor generation",
        "BlindJudge": "Objective evaluation with rigorous blinding methodology",
        "ValidationOrchestrator": "End-to-end validation pipeline coordination",
    },
    "key_features": [
        "DeepSeek V3 integration with advanced reasoning",
        "Anthropic best practices implementation",
        "Multi-criteria weighted evaluation",
        "Statistical significance testing",
        "Comprehensive audit trails",
        "Batch validation scenario execution",
    ],
    "evaluation_criteria": {
        "logical_structure": 0.20,
        "evidence_grounding": 0.25,
        "depth_of_analysis": 0.18,
        "actionability": 0.20,
        "consulting_quality": 0.12,
        "factual_accuracy": 0.05,
    },
}

# Convenience functions for quick access
__all__ = [
    # Core classes
    "BaselineChallenger",
    "BlindJudge",
    "ValidationOrchestrator",
    # Factory functions
    "get_baseline_challenger",
    "get_blind_judge",
    "get_validation_orchestrator",
    # Execution functions
    "generate_baseline_mega_prompt",
    "execute_baseline_challenge",
    "evaluate_metis_vs_baseline",
    "run_single_validation",
    "run_aperture_labs_validation",
    # Data classes
    "ValidationScenario",
    "ValidationRun",
    "ValidationReport",
    "EvaluationResult",
    "EvaluationCriterion",
    "BlindedOutput",
    "MegaPromptMetadata",
    # Package info
    "VALIDATION_FRAMEWORK_INFO",
    "__version__",
]


def get_framework_status() -> dict:
    """Get current validation framework status and capabilities"""

    try:
        challenger = get_baseline_challenger()
        judge = get_blind_judge()
        orchestrator = get_validation_orchestrator()

        return {
            "status": "operational",
            "version": __version__,
            "components_initialized": {
                "challenger": challenger.llm_client is not None,
                "judge": judge.llm_client is not None,
                "orchestrator": True,
            },
            "preferred_model": orchestrator.preferred_model,
            "reasoning_enabled": orchestrator.enable_reasoning,
            "evaluation_criteria": len(judge.evaluation_criteria),
            "ready_for_validation": True,
        }

    except Exception as e:
        return {
            "status": "error",
            "version": __version__,
            "error": str(e),
            "ready_for_validation": False,
        }


def validate_installation() -> bool:
    """Validate that all validation components are properly installed and configured"""

    try:
        # Test component initialization
        challenger = get_baseline_challenger()
        judge = get_blind_judge()
        orchestrator = get_validation_orchestrator()

        # Test basic functionality
        scenario = orchestrator.create_aperture_labs_scenario()

        # Check LLM client availability
        if not challenger.llm_client:
            print("❌ BaselineChallenger: No LLM client available")
            return False

        if not judge.llm_client:
            print("❌ BlindJudge: No LLM client available")
            return False

        print("✅ METIS Validation Framework: All components operational")
        print(f"   🧠 Model: {orchestrator.preferred_model}")
        print(
            f"   🔬 Reasoning: {'Enabled' if orchestrator.enable_reasoning else 'Disabled'}"
        )
        print(f"   ⚖️ Evaluation Criteria: {len(judge.evaluation_criteria)}")
        print("   📊 Ready for validation runs")

        return True

    except Exception as e:
        print(f"❌ Validation Framework Error: {e}")
        return False


# Module-level initialization message
if __name__ != "__main__":
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"🔥 METIS Validation Framework v{__version__} initialized - Operation Crucible ready"
    )
