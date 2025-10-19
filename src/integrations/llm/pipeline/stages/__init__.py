"""
Pipeline Stages Package

Contains all specialized pipeline stages for LLM call processing.

Implemented Stages:
- ✅ Task 3.0 - Security & Privacy:
  - InjectionFirewallStage (10/13 tests)
  - PIIRedactionStage (9/9 tests)
  - SensitivityRoutingStage (9/9 tests)
- ✅ Task 4.0 - Validation & Context:
  - OutputContractStage (10/10 tests)
  - RAGContextInjectionStage (11/11 tests)

Pending Stages (Tasks 5-6):
- ProviderAdapterStage (Task 5.1)
- ReasoningModeStage (Task 6.1)
- StyleGateStage (Task 6.3)
- ConfidenceEscalationStage (Task 6.5)

Each stage:
- Inherits from PipelineStage
- Implements execute() method
- Handles one specific concern (Single Responsibility)
- Returns new context immutably
- Has comprehensive unit tests (≥85% coverage)
- Has CC < 5 (Grade A)

Version: 1.0.0
Date: 2025-10-17
"""

# === Implemented Stages ===
from .injection_firewall import InjectionFirewallStage
from .pii_redaction import PIIRedactionStage
from .sensitivity_routing import SensitivityRoutingStage
from .output_contract import OutputContractStage
from .rag_context_injection import RAGContextInjectionStage
from .provider_adapter import ProviderAdapterStage
from .reasoning_mode import ReasoningModeStage
from .style_gate import StyleGateStage
from .confidence_escalation import ConfidenceEscalationStage

# === Public API ===
__all__ = [
    # Task 3.0 - Security & Privacy Stages
    "InjectionFirewallStage",
    "PIIRedactionStage",
    "SensitivityRoutingStage",
    # Task 4.0 - Validation & Context Stages
    "OutputContractStage",
    "RAGContextInjectionStage",
    # Task 5.0 - Provider Adaptation
    "ProviderAdapterStage",
    # Task 6.0 - Advanced Features
    "ReasoningModeStage",
    "StyleGateStage",
    "ConfidenceEscalationStage",
]

# === Version Info ===
__version__ = "1.0.0"
__author__ = "METIS V5.3 Team"
__date__ = "2025-10-17"
