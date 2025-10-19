from typing import Dict, Any, List


class DiagnosticError(Exception):
    def __init__(self, error: str, missing: List[str], fix: str):
        self.payload = {"error": error, "missing": missing, "fix": fix}
        super().__init__(error)


def require_columns(dataset: List[Dict[str, Any]], required: List[str]):
    if dataset is None:
        raise DiagnosticError(
            "Dataset is None", required, "Provide a non-empty dataset list of rows"
        )
    if not isinstance(dataset, list) or (
        len(dataset) > 0 and not isinstance(dataset[0], dict)
    ):
        raise DiagnosticError(
            "Dataset must be a list of dict rows",
            required,
            "Ensure CSV/JSON parsed to list[dict]",
        )
    if not dataset:
        raise DiagnosticError("Empty dataset", required, "Provide at least one row")
    missing = [col for col in required if col not in dataset[0].keys()]
    if missing:
        raise DiagnosticError(
            error=f"Missing required columns: {missing}",
            missing=missing,
            fix="Add missing columns or update mapping to a unique field",
        )


def normalise_numbers(row: Dict[str, Any], numeric_fields: List[str]):
    for f in numeric_fields:
        if f in row and row[f] is not None:
            try:
                row[f] = float(row[f])
            except (TypeError, ValueError):
                # Leave as-is; enforcement can handle later
                pass
    return row
