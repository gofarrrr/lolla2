"""Calibration utilities: Brier score and Expected Calibration Error (ECE)."""
from typing import List, Tuple, Dict


def brier_score(pairs: List[Tuple[float, int]]) -> float:
    if not pairs:
        return 0.0
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def expected_calibration_error(pairs: List[Tuple[float, int]], n_bins: int = 10) -> Dict[str, float]:
    if not pairs:
        return {"ece": 0.0, "bins": 0, "max_gap": 0.0}
    # Clamp
    clamped = [(max(0.0, min(1.0, p)), 1 if y else 0) for p, y in pairs]
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in clamped:
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, y))
    ece = 0.0
    max_gap = 0.0
    total = len(clamped)
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        conf_avg = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        gap = abs(conf_avg - acc)
        max_gap = max(max_gap, gap)
        ece += (len(bucket) / total) * gap
    return {"ece": ece, "bins": n_bins, "max_gap": max_gap}