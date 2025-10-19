"""
Reconstruction Stage - Abstract Base Class
==========================================

Abstract base class for reconstruction stages in the State Reconstruction Pattern.

Each stage:
1. Receives immutable ReconstructionState
2. Performs one specific extraction/transformation
3. Returns new ReconstructionState with updates

Design Principles:
- Single responsibility per stage
- Immutable state transformations
- Low cyclomatic complexity (CC<10)
- Easy unit testing
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.services.report_reconstruction.reconstruction_state import ReconstructionState


class ReconstructionStage(ABC):
    """
    Abstract base class for reconstruction stages.

    Similar to PipelineStage but for report reconstruction.
    Each stage performs one specific extraction or transformation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name for logging and debugging."""
        pass

    @property
    def description(self) -> Optional[str]:
        """Optional stage description for debugging."""
        return None

    @abstractmethod
    def process(self, state: ReconstructionState) -> ReconstructionState:
        """
        Process reconstruction state through this stage.

        This method must:
        1. Extract/transform data from the current state
        2. Return a NEW state with updates (immutable pattern)
        3. Keep cyclomatic complexity low (CC<10)

        Args:
            state: Current reconstruction state (immutable)

        Returns:
            Updated reconstruction state with this stage's contributions

        Raises:
            ReconstructionError: If stage processing fails
        """
        pass

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class ReconstructionError(Exception):
    """Exception raised during reconstruction stage processing."""

    def __init__(self, stage_name: str, message: str, cause: Optional[Exception] = None):
        self.stage_name = stage_name
        self.message = message
        self.cause = cause
        super().__init__(f"[{stage_name}] {message}")
