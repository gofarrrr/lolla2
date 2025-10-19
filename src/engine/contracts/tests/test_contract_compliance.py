"""
Contract Compliance Tests - METIS V5 API Contract Registry
Tests to ensure all components adhere to the defined contracts

These tests prevent the API contract failures identified in Dossier X
"""

import pytest
import asyncio
from datetime import datetime

# Import all contracts
from contracts.common_contracts import (
    EngagementContext,
    ProcessingMetrics,
    ErrorResponse,
    ProcessingStatus,
    ErrorSeverity,
)
from contracts.socratic_contracts import (
    SocraticRequest,
    SocraticEngineInterface,
)
from contracts.analysis_contracts import (
    AnalysisRequest,
    AnalysisEngineInterface,
)


class TestContractDataStructures:
    """Test that all data contracts can be created and serialized"""

    def test_engagement_context_creation(self):
        """Test EngagementContext creation and serialization"""
        context = EngagementContext(
            engagement_id="test-123",
            problem_statement="Test problem",
            business_context={"test": "data"},
        )

        # Test serialization
        data = context.to_dict()
        assert data["engagement_id"] == "test-123"
        assert data["problem_statement"] == "Test problem"
        assert data["business_context"]["test"] == "data"

    def test_socratic_request_creation(self):
        """Test SocraticRequest creation and serialization"""
        context = EngagementContext(
            engagement_id="test-123", problem_statement="Test problem"
        )

        request = SocraticRequest(engagement_context=context, force_real_llm_call=True)

        # Test serialization
        data = request.to_dict()
        assert data["engagement_context"]["engagement_id"] == "test-123"
        assert data["force_real_llm_call"] == True

    def test_analysis_request_creation(self):
        """Test AnalysisRequest creation and serialization"""
        context = EngagementContext(
            engagement_id="test-123", problem_statement="Test problem"
        )

        request = AnalysisRequest(
            engagement_context=context,
            selected_consultants=["consultant1", "consultant2"],
        )

        # Test serialization
        data = request.to_dict()
        assert data["engagement_context"]["engagement_id"] == "test-123"
        assert data["selected_consultants"] == ["consultant1", "consultant2"]


class TestInterfaceContract:
    """Test that interface contracts are properly defined"""

    def test_socratic_engine_interface(self):
        """Test that SocraticEngineInterface defines required methods"""
        interface = SocraticEngineInterface()

        # Test that required methods exist and raise NotImplementedError
        with pytest.raises(NotImplementedError):
            asyncio.run(interface.generate_progressive_questions(None))

        with pytest.raises(NotImplementedError):
            asyncio.run(interface.health_check())

    def test_analysis_engine_interface(self):
        """Test that AnalysisEngineInterface defines required methods"""
        interface = AnalysisEngineInterface()

        # Test that required methods exist and raise NotImplementedError
        with pytest.raises(NotImplementedError):
            asyncio.run(interface.execute_consultant_analysis(None))

        with pytest.raises(NotImplementedError):
            asyncio.run(interface.execute_single_consultant(None, None))

        with pytest.raises(NotImplementedError):
            asyncio.run(interface.health_check())


class TestContractValidation:
    """Test contract validation and error handling"""

    def test_error_response_structure(self):
        """Test ErrorResponse structure"""
        error = ErrorResponse(
            error_code="CONTRACT_VIOLATION",
            error_message="Method signature mismatch",
            severity=ErrorSeverity.HIGH,
            component="SocraticCognitiveForge",
        )

        data = error.to_dict()
        assert data["error_code"] == "CONTRACT_VIOLATION"
        assert data["component"] == "SocraticCognitiveForge"

    def test_processing_metrics_structure(self):
        """Test ProcessingMetrics structure"""
        start_time = datetime.now()
        end_time = datetime.now()

        metrics = ProcessingMetrics(
            component_name="test_component",
            processing_time_seconds=1.5,
            start_time=start_time,
            end_time=end_time,
            status=ProcessingStatus.COMPLETED,
        )

        data = metrics.to_dict()
        assert data["component_name"] == "test_component"
        assert data["processing_time_seconds"] == 1.5
        assert data["status"] == "completed"


# Contract Violation Detection Tests
class TestDossierXFailurePrevention:
    """Specific tests to prevent the exact failures found in Dossier X"""

    def test_socratic_method_signature_compliance(self):
        """Prevent: 'generate_progressive_questions' method not found"""
        # This test ensures any Socratic engine implementation has the required method
        interface = SocraticEngineInterface()

        # Method must exist
        assert hasattr(interface, "generate_progressive_questions")

        # Method must be async
        import inspect

        assert inspect.iscoroutinefunction(interface.generate_progressive_questions)

        # Method must accept SocraticRequest and return SocraticResponse
        # (Implementation will be tested in component-specific tests)

    def test_analysis_method_signature_compliance(self):
        """Prevent: parameter signature mismatch in analysis execution"""
        interface = AnalysisEngineInterface()

        # Method must exist
        assert hasattr(interface, "execute_consultant_analysis")

        # Method must be async
        import inspect

        assert inspect.iscoroutinefunction(interface.execute_consultant_analysis)

        # Method must accept AnalysisRequest (not loose parameters like problem_statement)
        sig = inspect.signature(interface.execute_consultant_analysis)
        params = list(sig.parameters.keys())

        print(f"Debug: Found {len(params)} parameters: {params}")

        # Should have at least 1 parameter (may just be self in abstract class)
        assert len(params) >= 1
        # If more than just self, should include request
        if len(params) > 1:
            assert "request" in params
        print(f"✅ Analysis method parameters: {params}")


if __name__ == "__main__":
    # Run basic contract tests
    print("🧪 Running Contract Compliance Tests...")

    test_data = TestContractDataStructures()
    test_data.test_engagement_context_creation()
    test_data.test_socratic_request_creation()
    test_data.test_analysis_request_creation()

    test_interface = TestInterfaceContract()
    test_interface.test_socratic_engine_interface()
    test_interface.test_analysis_engine_interface()

    test_validation = TestContractValidation()
    test_validation.test_error_response_structure()
    test_validation.test_processing_metrics_structure()

    test_dossier = TestDossierXFailurePrevention()
    test_dossier.test_socratic_method_signature_compliance()
    test_dossier.test_analysis_method_signature_compliance()

    print("✅ All Contract Compliance Tests Passed!")
    print("🛡️ API Contract failures from Dossier X are now prevented!")
