"""
Workflow Engine Interfaces for Dependency Injection
"""

from typing import Protocol
from src.engine.models.data_contracts import MetisDataContract


class IWorkflowEngine(Protocol):
    """Interface for workflow engine operations"""

    async def execute_engagement(
        self, contract: MetisDataContract
    ) -> MetisDataContract:
        """Execute complete engagement workflow"""
        ...

    async def execute_phase(
        self, contract: MetisDataContract, phase: str
    ) -> MetisDataContract:
        """Execute specific workflow phase"""
        ...
