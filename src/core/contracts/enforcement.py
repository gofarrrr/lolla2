from typing import Dict


def apply_rounding_and_signs(report_json: Dict, decimals: int = 1):
    """Round numeric fields and ensure signed deltas in per_criterion.

    Mutates the provided report_json in-place.
    """
    items = report_json.get("per_criterion", [])
    for item in items:
        if "a" in item:
            try:
                item["a"] = float(f"{float(item['a']):.{decimals}f}")
            except (TypeError, ValueError):
                pass
        if "b" in item:
            try:
                item["b"] = float(f"{float(item['b']):.{decimals}f}")
            except (TypeError, ValueError):
                pass
        if "delta" in item:
            try:
                val = float(item["delta"])
                # Keep numeric but ensure sign by storing a string with sign for display contexts if needed
                # Here we keep numeric delta rounded; sign can be inferred downstream
                item["delta"] = float(f"{val:.{decimals}f}")
            except (TypeError, ValueError):
                pass
    return report_json
