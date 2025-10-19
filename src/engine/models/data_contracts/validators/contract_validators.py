"""
Contract Validators

Validation functions for data contract compliance.
"""

from typing import Dict, Any
from ..models.analysis_models import MetisDataContract


def validate_data_contract_compliance(event_data: Dict[str, Any]) -> bool:
    """Validate CloudEvents compliance and METIS schema adherence"""
    try:
        contract = MetisDataContract.from_cloudevents_dict(event_data)
        return True
    except Exception:
        return False




