"""
METIS Centralized Bias System Activation
Consolidates initialization of bias detection and validation systems
"""

import logging
import os
from typing import Any


def activate_bias_systems(engine: Any, enable: bool = True) -> None:
    """
    Centralized activation of bias detection and validation systems

    Args:
        engine: Cognitive engine instance
        enable: Whether to enable bias systems (default True)
    """
    logger = engine.logger if hasattr(engine, "logger") else logging.getLogger(__name__)

    if not enable:
        engine.bias_detection_enabled = False
        engine.mandatory_validation_enabled = False
        engine.ai_augmentation = None
        engine.validation_gate_engine = None
        if hasattr(engine, "cognitive_auditor"):
            engine.cognitive_auditor = None
        if hasattr(engine, "munger_overlay"):
            engine.munger_overlay = None
            engine.rigor_level_tracking_enabled = False
        logger.info("✅ Bias systems disabled by configuration")
        return

    # Initialize AI Augmentation Engine for bias detection
    try:
        import sys

        # Add scripts/demo to path for AIAugmentationEngine import
        scripts_demo_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts", "demo"
        )
        if scripts_demo_path not in sys.path:
            sys.path.append(scripts_demo_path)

        from ai_augmentation_poc import AIAugmentationEngine

        engine.ai_augmentation = AIAugmentationEngine()
        engine.bias_detection_enabled = True
        logger.info("✅ AI Augmentation Engine activated - Bias detection enabled")

    except Exception as e:
        engine.ai_augmentation = None
        engine.bias_detection_enabled = False
        logger.warning(f"⚠️ AI Augmentation Engine unavailable: {e}")

    # Initialize LLM Validation Gates for mandatory validation
    try:
        from src.core.llm_validation_gates import get_validation_gate_engine

        engine.validation_gate_engine = get_validation_gate_engine()
        engine.mandatory_validation_enabled = True
        logger.info("✅ LLM Validation Gates activated - Mandatory validation enabled")

    except Exception as e:
        engine.validation_gate_engine = None
        engine.mandatory_validation_enabled = False
        logger.warning(f"⚠️ LLM Validation Gates unavailable: {e}")

    # Initialize Cognitive Auditor (if available)
    try:
        from src.engine.engines.models.cognitive_auditor import CognitiveAuditor

        engine.cognitive_auditor = CognitiveAuditor()
        logger.info("✅ Cognitive Auditor activated")
    except Exception as e:
        if hasattr(engine, "cognitive_auditor"):
            engine.cognitive_auditor = None
        logger.warning(f"⚠️ Cognitive Auditor unavailable: {e}")

    # Initialize Munger Overlay (if available)
    try:
        from src.core.munger_overlay import MungerOverlay

        engine.munger_overlay = MungerOverlay()
        engine.rigor_level_tracking_enabled = True
        logger.info("✅ Munger Overlay activated - Rigor level tracking enabled")
    except Exception as e:
        if hasattr(engine, "munger_overlay"):
            engine.munger_overlay = None
            engine.rigor_level_tracking_enabled = False
        logger.warning(f"⚠️ Munger Overlay unavailable: {e}")
