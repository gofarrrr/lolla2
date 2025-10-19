#!/usr/bin/env python3
"""
Calibration Service - Minimal in-memory calibration loop wiring

Bridges pipeline stages to the realtime calibration engine by:
- Recording predictions when a ConfidenceAssessment is reported
- Recording outcomes via API and updating calibration
- Providing a lightweight status/report access

Designed to work offline (no DB required). Supabase features are not used.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from uuid import uuid4

try:
    # Prefer the engine version if available
    from src.intelligence.realtime_confidence_calibration import (
        RealtimeConfidenceCalibrator,
    )
except Exception:  # pragma: no cover - fallback (should not happen)
    RealtimeConfidenceCalibrator = None  # type: ignore


@dataclass
class PendingPrediction:
    trace_id: str
    persona_id: str
    predicted_confidence: float
    timestamp: datetime
    context: Dict[str, Any]
    prediction_id: Optional[str] = None


class CalibrationService:
    """Singleton service for managing calibration loop"""

    def __init__(self, database_service: Optional[Any] = None) -> None:
        self._db = database_service
        self._calibrator = (
            RealtimeConfidenceCalibrator() if RealtimeConfidenceCalibrator else None
        )
        # Pending predictions keyed by trace_id (allow multiple)
        self._pending: Dict[str, List[PendingPrediction]] = {}
        # Kick off hydration if DB is provided
        if self._db is not None:
            try:
                # Fire-and-forget hydration; tests can call hydrate() directly
                asyncio.get_running_loop()
                asyncio.create_task(self.hydrate())
            except RuntimeError:
                # No running loop (e.g., during import) — hydration will be explicit
                pass

    def report_prediction(
        self,
        *,
        trace_id: str,
        persona_id: str,
        predicted_confidence: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store a pending prediction to be closed later with an outcome.

        Persists the initial prediction if a database is available. Returns prediction_id when persisted.
        """
        ctx = dict(context or {})
        item = PendingPrediction(
            trace_id=trace_id,
            persona_id=persona_id,
            predicted_confidence=float(predicted_confidence),
            timestamp=datetime.utcnow(),
            context=ctx,
        )
        # Persist initial prediction if DB available
        if self._db is not None:
            try:
                pid = str(uuid4())
                payload = {
                    "id": pid,
                    "trace_id": trace_id,
                    "persona_id": persona_id,
                    "predicted_probability": float(predicted_confidence),
                    "is_early": False,
                    "notes": None,
                    "created_at": datetime.utcnow().isoformat(),
                }
                # store and record id
                awaitable = self._db.store_initial_prediction(payload)
                # handle sync vs async
                try:
                    # might be async def
                    res = asyncio.create_task(awaitable)  # type: ignore[arg-type]
                except TypeError:
                    # fallback
                    pass
                item.prediction_id = pid
            except Exception:
                # Non-fatal in offline mode
                pass
        self._pending.setdefault(trace_id, []).append(item)
        return item.prediction_id

    async def report_outcome(
        self,
        *,
        trace_id: str,
        outcome: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Attach an outcome to all pending predictions for a trace and record observations.

        Returns number of observations recorded.
        """
        if self._calibrator is None:
            return 0

        items = self._pending.pop(trace_id, [])
        count = 0
        for item in items:
            await self._calibrator.record_observation(
                model_id=item.persona_id,
                predicted_confidence=item.predicted_confidence,
                actual_outcome=float(outcome),
                engagement_id=trace_id,
                context={**item.context, **(context or {})},
            )
            # Persist outcome if DB available
            try:
                if self._db is not None:
                    await self._db.store_calibration_outcome(
                        id=item.prediction_id or str(uuid4()),
                        actual_outcome=float(outcome),
                        notes=(context or {}).get("notes") if context else None,
                    )
            except Exception:
                # Non-fatal in offline mode
                pass
            count += 1
        return count

    async def generate_report(self) -> Dict[str, Any]:
        """Return a serializable calibration report."""
        if self._calibrator is None:
            return {"error": "Calibrator unavailable"}
        report = await self._calibrator.generate_calibration_report()
        # Convert dataclasses to dict-like structures for API safety
        out: Dict[str, Any] = {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "system_calibration_score": report.system_calibration_score,
            "average_brier_score": report.average_brier_score,
            "total_models": report.total_models,
            "total_observations": report.total_observations,
            "models_flagged": report.models_flagged,
            "calibration_adjustments": report.calibration_adjustments,
            "improvement_suggestions": report.improvement_suggestions,
        }
        return out

    async def hydrate(self) -> int:
        """Load historical calibration rows from the database and feed the calibrator.

        Returns number of observations ingested.
        """
        if self._db is None or self._calibrator is None:
            return 0
        try:
            rows = await self._db.fetch_all_calibration_data()
        except Exception:
            return 0
        count = 0
        for r in rows or []:
            try:
                pred = float(r.get("predicted_probability") or 0.0)
                actual = r.get("actual_outcome")
                if actual is None:
                    continue  # skip open predictions for calibration metrics
                await self._calibrator.record_observation(
                    model_id=str(r.get("persona_id") or "unknown"),
                    predicted_confidence=pred,
                    actual_outcome=float(actual),
                    engagement_id=str(r.get("trace_id") or ""),
                    context=r.get("context") or {},
                )
                count += 1
            except Exception:
                continue
        return count
    async def calibrate_value(self, model_id: str, value: float) -> Tuple[float, Dict[str, Any]]:
        if self._calibrator is None:
            return value, {"calibration_applied": False, "reason": "Calibrator unavailable"}
        return await self._calibrator.calibrate_confidence(model_id, value)


_instance: Optional[CalibrationService] = None


def get_calibration_service(database_service: Optional[Any] = None) -> CalibrationService:
    global _instance
    if _instance is None:
        _instance = CalibrationService(database_service=database_service)
    else:
        # Late-binding DB if singleton exists without DB
        if getattr(_instance, "_db", None) is None and database_service is not None:
            _instance._db = database_service
            try:
                asyncio.get_running_loop()
                asyncio.create_task(_instance.hydrate())
            except RuntimeError:
                pass
    return _instance

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
