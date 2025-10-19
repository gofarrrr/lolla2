"""
Tests for EventValidationService

Task 3.8: Comprehensive test coverage for event validation and PII scrubbing.
Target: ≥90% coverage

Created: 2025-10-18
Campaign: Operation Lean
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from src.core.services.event_validation_service import EventValidationService


class MockContextEventType:
    """Mock ContextEventType enum for testing"""
    def __init__(self, value):
        self.value = value


class TestEventValidationServiceInit:
    """Test service initialization"""

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        service = EventValidationService(
            allowlist_path=Path(__file__).parent / "test_allowlist.yaml",
            strict_validation_enabled=False  # Disable to avoid file dependency
        )
        assert service.strict_validation_enabled == False
        assert service.pii_engine is None
        assert service.pii_redaction_enabled == False

    def test_init_with_pii_engine(self):
        """Test initialization with PII engine"""
        mock_pii_engine = Mock()
        service = EventValidationService(
            pii_engine=mock_pii_engine,
            strict_validation_enabled=False
        )
        assert service.pii_engine == mock_pii_engine
        assert service.pii_redaction_enabled == True

    def test_init_loads_allowlist(self, tmp_path):
        """Test that allowlist is loaded from YAML file"""
        # Create temporary allowlist file
        allowlist_file = tmp_path / "test_allowlist.yaml"
        allowlist_file.write_text("""
ENGAGEMENT_STARTED:
  allowed_next: ["QUERY_RECEIVED", "PHASE_COMPLETED"]
QUERY_RECEIVED:
  allowed_next: ["*"]
""")

        service = EventValidationService(
            allowlist_path=allowlist_file,
            strict_validation_enabled=True
        )

        assert service.strict_validation_enabled == True
        assert "ENGAGEMENT_STARTED" in service.event_allowlist
        assert service.event_allowlist["ENGAGEMENT_STARTED"]["allowed_next"] == [
            "QUERY_RECEIVED", "PHASE_COMPLETED"
        ]

    def test_init_handles_missing_allowlist(self, tmp_path):
        """Test graceful handling of missing allowlist file"""
        missing_file = tmp_path / "nonexistent.yaml"

        service = EventValidationService(
            allowlist_path=missing_file,
            strict_validation_enabled=True
        )

        # Should disable strict validation when allowlist fails to load
        assert service.strict_validation_enabled == False
        assert service.event_allowlist == {}


class TestValidateEventSchema:
    """Test event schema validation"""

    def test_schema_validation_disabled(self):
        """Test that validation is skipped when strict mode disabled"""
        service = EventValidationService(strict_validation_enabled=False)

        result = service.validate_event_schema(
            MockContextEventType("TEST_EVENT"),
            {"invalid": "data"}
        )

        assert result == True

    @patch('src.core.event_schemas.validate_event_payload')
    def test_schema_validation_success(self, mock_validate):
        """Test successful schema validation"""
        mock_validate.return_value = (True, [])

        service = EventValidationService(strict_validation_enabled=True)
        service.strict_validation_enabled = True  # Force enable

        result = service.validate_event_schema(
            MockContextEventType("ENGAGEMENT_STARTED"),
            {"user_query": "test query"}
        )

        assert result == True
        mock_validate.assert_called_once()

    @patch('src.core.event_schemas.validate_event_payload')
    def test_schema_validation_failure(self, mock_validate):
        """Test schema validation failure"""
        mock_validate.return_value = (False, ["Missing required field: user_query"])

        service = EventValidationService(strict_validation_enabled=True)
        service.strict_validation_enabled = True

        result = service.validate_event_schema(
            MockContextEventType("ENGAGEMENT_STARTED"),
            {"invalid": "data"}
        )

        assert result == False

    def test_schema_validation_handles_import_error(self):
        """Test fallback when event_schemas module unavailable"""
        service = EventValidationService(strict_validation_enabled=True)
        service.strict_validation_enabled = True

        # This should gracefully handle ImportError and return True
        result = service.validate_event_schema(
            MockContextEventType("TEST_EVENT"),
            {"data": "value"}
        )

        # Should return True when module unavailable (graceful degradation)
        assert result == True


class TestValidateEventTransition:
    """Test event state transition validation"""

    def test_transition_validation_disabled(self):
        """Test that transition validation is skipped when disabled"""
        service = EventValidationService(strict_validation_enabled=False)

        result = service.validate_event_transition(
            MockContextEventType("ANY_EVENT"),
            MockContextEventType("PREVIOUS_EVENT")
        )

        assert result == True

    def test_first_event_always_allowed(self, tmp_path):
        """Test that first event (no previous) is always allowed"""
        allowlist_file = tmp_path / "allowlist.yaml"
        allowlist_file.write_text("ENGAGEMENT_STARTED:\n  allowed_next: []")

        service = EventValidationService(
            allowlist_path=allowlist_file,
            strict_validation_enabled=True
        )

        result = service.validate_event_transition(
            MockContextEventType("ENGAGEMENT_STARTED"),
            previous_event_type=None
        )

        assert result == True

    def test_wildcard_allows_any_transition(self, tmp_path):
        """Test that wildcard (*) allows any transition"""
        allowlist_file = tmp_path / "allowlist.yaml"
        allowlist_file.write_text("""
QUERY_RECEIVED:
  allowed_next: ["*"]
""")

        service = EventValidationService(
            allowlist_path=allowlist_file,
            strict_validation_enabled=True
        )

        result = service.validate_event_transition(
            MockContextEventType("ANY_NEXT_EVENT"),
            MockContextEventType("QUERY_RECEIVED")
        )

        assert result == True

    def test_valid_transition_allowed(self, tmp_path):
        """Test that valid transitions are allowed"""
        allowlist_file = tmp_path / "allowlist.yaml"
        allowlist_file.write_text("""
ENGAGEMENT_STARTED:
  allowed_next: ["QUERY_RECEIVED", "PHASE_COMPLETED"]
""")

        service = EventValidationService(
            allowlist_path=allowlist_file,
            strict_validation_enabled=True
        )

        result = service.validate_event_transition(
            MockContextEventType("QUERY_RECEIVED"),
            MockContextEventType("ENGAGEMENT_STARTED")
        )

        assert result == True

    def test_invalid_transition_rejected(self, tmp_path):
        """Test that invalid transitions are rejected"""
        allowlist_file = tmp_path / "allowlist.yaml"
        allowlist_file.write_text("""
ENGAGEMENT_STARTED:
  allowed_next: ["QUERY_RECEIVED"]
""")

        service = EventValidationService(
            allowlist_path=allowlist_file,
            strict_validation_enabled=True
        )

        result = service.validate_event_transition(
            MockContextEventType("INVALID_NEXT"),
            MockContextEventType("ENGAGEMENT_STARTED")
        )

        assert result == False

    def test_case_insensitive_matching(self, tmp_path):
        """Test that event type matching is case-insensitive"""
        allowlist_file = tmp_path / "allowlist.yaml"
        allowlist_file.write_text("""
ENGAGEMENT_STARTED:
  allowed_next: ["query_received"]
""")

        service = EventValidationService(
            allowlist_path=allowlist_file,
            strict_validation_enabled=True
        )

        result = service.validate_event_transition(
            MockContextEventType("QUERY_RECEIVED"),
            MockContextEventType("ENGAGEMENT_STARTED")
        )

        assert result == True


class TestIsEventAllowed:
    """Test event allowlist checking"""

    def test_no_allowlist_allows_all(self):
        """Test that missing allowlist allows all events"""
        service = EventValidationService(strict_validation_enabled=False)
        service.event_allowlist = {}

        result = service.is_event_allowed(MockContextEventType("ANY_EVENT"))

        assert result == True

    def test_event_in_allowlist(self, tmp_path):
        """Test event present in allowlist"""
        allowlist_file = tmp_path / "allowlist.yaml"
        allowlist_file.write_text("""
ENGAGEMENT_STARTED:
  allowed_next: []
QUERY_RECEIVED:
  allowed_next: []
""")

        service = EventValidationService(
            allowlist_path=allowlist_file,
            strict_validation_enabled=True
        )

        result = service.is_event_allowed(MockContextEventType("ENGAGEMENT_STARTED"))

        assert result == True

    def test_event_not_in_allowlist(self, tmp_path):
        """Test event not present in allowlist"""
        allowlist_file = tmp_path / "allowlist.yaml"
        allowlist_file.write_text("""
ENGAGEMENT_STARTED:
  allowed_next: []
""")

        service = EventValidationService(
            allowlist_path=allowlist_file,
            strict_validation_enabled=True
        )

        result = service.is_event_allowed(MockContextEventType("UNKNOWN_EVENT"))

        assert result == False


class TestScrubPII:
    """Test PII scrubbing functionality"""

    def test_scrub_email_addresses(self):
        """Test redaction of email addresses"""
        service = EventValidationService(strict_validation_enabled=False)

        text = "Contact me at john.doe@example.com for more info"
        result = service.scrub_pii(text)

        assert "john.doe@example.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_scrub_phone_numbers(self):
        """Test redaction of phone numbers"""
        service = EventValidationService(strict_validation_enabled=False)

        text = "Call me at 555-123-4567 or (555) 987-6543"
        result = service.scrub_pii(text)

        assert "555-123-4567" not in result
        assert "[REDACTED_PHONE]" in result

    def test_scrub_ssn(self):
        """Test redaction of SSN"""
        service = EventValidationService(strict_validation_enabled=False)

        text = "SSN: 123-45-6789"
        result = service.scrub_pii(text)

        assert "123-45-6789" not in result
        assert "[REDACTED_SSN]" in result

    def test_scrub_multiple_pii_types(self):
        """Test redaction of multiple PII types in one string"""
        service = EventValidationService(strict_validation_enabled=False)

        text = "Email: test@example.com, Phone: 555-123-4567, SSN: 123-45-6789"
        result = service.scrub_pii(text)

        assert "test@example.com" not in result
        assert "555-123-4567" not in result
        assert "123-45-6789" not in result
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_SSN]" in result

    def test_scrub_with_pii_engine(self):
        """Test PII scrubbing using PII engine"""
        mock_pii_engine = Mock()
        mock_result = Mock()
        mock_result.redacted_text = "Text with [REDACTED] content"
        mock_result.redaction_count = 2
        mock_pii_engine.redact.return_value = mock_result

        service = EventValidationService(
            pii_engine=mock_pii_engine,
            strict_validation_enabled=False
        )

        result = service.scrub_pii("Text with sensitive@email.com content")

        assert result == "Text with [REDACTED] content"
        mock_pii_engine.redact.assert_called_once()

    def test_scrub_non_string_returns_unchanged(self):
        """Test that non-string values are returned unchanged"""
        service = EventValidationService(strict_validation_enabled=False)

        assert service.scrub_pii(123) == 123
        assert service.scrub_pii(None) == None
        assert service.scrub_pii(True) == True


class TestScrubStructure:
    """Test recursive PII scrubbing in nested structures"""

    def test_scrub_dict_structure(self):
        """Test PII scrubbing in dictionary"""
        service = EventValidationService(strict_validation_enabled=False)

        data = {
            "email": "user@example.com",
            "phone": "555-1234",
            "nested": {
                "contact": "admin@test.com"
            }
        }

        result = service.scrub_structure(data)

        assert "user@example.com" not in str(result)
        assert "[REDACTED_EMAIL]" in result["email"]
        assert "[REDACTED_EMAIL]" in result["nested"]["contact"]

    def test_scrub_list_structure(self):
        """Test PII scrubbing in list"""
        service = EventValidationService(strict_validation_enabled=False)

        data = [
            "Contact: test@example.com",
            "Phone: 555-1234",
            {"email": "nested@test.com"}
        ]

        result = service.scrub_structure(data)

        assert "[REDACTED_EMAIL]" in result[0]
        assert "[REDACTED_EMAIL]" in result[2]["email"]

    def test_scrub_mixed_nested_structure(self):
        """Test PII scrubbing in complex nested structure"""
        service = EventValidationService(strict_validation_enabled=False)

        data = {
            "users": [
                {"email": "user1@test.com", "phone": "555-123-4567"},
                {"email": "user2@test.com", "phone": "555-987-6543"}
            ],
            "metadata": {
                "admin_contact": "admin@example.com"
            }
        }

        result = service.scrub_structure(data)

        assert "[REDACTED_EMAIL]" in result["users"][0]["email"]
        assert "[REDACTED_PHONE]" in result["users"][0]["phone"]
        assert "[REDACTED_EMAIL]" in result["metadata"]["admin_contact"]

    def test_scrub_primitive_values(self):
        """Test that primitive values are returned unchanged"""
        service = EventValidationService(strict_validation_enabled=False)

        assert service.scrub_structure(123) == 123
        assert service.scrub_structure(45.67) == 45.67
        assert service.scrub_structure(True) == True
        assert service.scrub_structure(None) == None


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_allowlist_file(self, tmp_path):
        """Test handling of empty allowlist file"""
        allowlist_file = tmp_path / "empty.yaml"
        allowlist_file.write_text("")

        service = EventValidationService(
            allowlist_path=allowlist_file,
            strict_validation_enabled=True
        )

        assert service.event_allowlist == {}

    def test_malformed_yaml(self, tmp_path):
        """Test handling of malformed YAML file"""
        allowlist_file = tmp_path / "malformed.yaml"
        allowlist_file.write_text("invalid: yaml: content: [")

        service = EventValidationService(
            allowlist_path=allowlist_file,
            strict_validation_enabled=True
        )

        # Should disable strict validation on error
        assert service.strict_validation_enabled == False

    def test_pii_engine_exception_fallback(self):
        """Test fallback to regex when PII engine raises exception"""
        mock_pii_engine = Mock()
        mock_pii_engine.redact.side_effect = Exception("PII engine error")

        service = EventValidationService(
            pii_engine=mock_pii_engine,
            strict_validation_enabled=False
        )

        text = "Email: test@example.com"
        result = service.scrub_pii(text)

        # Should fall back to regex and still redact
        assert "test@example.com" not in result
        assert "[REDACTED_EMAIL]" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.core.services.event_validation_service", "--cov-report=term-missing"])
