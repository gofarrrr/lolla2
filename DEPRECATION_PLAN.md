# API Deprecation Plan
**Created**: 2025-10-20  
**Owner**: CTO (Codex)  
**Target completion**: 2026-02-28

---

## Overview

The legacy API stack (`src/engine/api/`) must be consolidated into the Lean router stack (`src/api/routes/`). This plan tracks migration progress, communicates timelines, and documents operational safeguards.

---

## Deprecation Strategy

1. **Phase 1 – Awareness (Nov 2025)**  
   Add `X-Api-Version: legacy` and `X-Api-Deprecation` headers to engine routes. Publish deprecation notice in the API changelog.

2. **Phase 2 – Migration (Dec 2025)**  
   Move high-traffic routes into `src/api/routes/`, add parity integration tests, and cut over clients.

3. **Phase 3 – Sunset (Jan 2026)**  
   Legacy routes return HTTP 410 with migration guidance once traffic drops below 5%.

4. **Phase 4 – Removal (Feb 2026)**  
   Delete `src/engine/api/` directory after verifying zero traffic for 30 consecutive days.

---

## Route Inventory

| Legacy Route | 30-day Traffic % | Replacement Route | Owner | Target Date | Status |
|--------------|------------------|-------------------|-------|-------------|--------|
| `/api/enhanced_foundation` | 16.7% (sample) | `src/api/routes/foundation_routes.py` | @lean-foundation | 2025-11-15 | 🟢 In Progress |
| `/api/progressive_questions` | 22.4% (sample) | `src/api/routes/questions_routes.py` | @questions-pod | 2025-11-30 | 🟠 Ready for cutover |
| `/api/analysis_execution` | 45.1% (sample) | `src/api/routes/analyze_routes.py` | @analysis-core | 2025-10-19 | ✅ Complete |
| `/api/streaming` | 7.3% (sample) | `src/api/routes/stateful_analysis_routes.py` | @realtime-squad | 2025-11-22 | 🟠 Scoping |

> Populate the traffic column with `scripts/measure_route_traffic.py` using production logs.

---

## Migration Checklist

- [ ] Measure current traffic (30-day window)
- [ ] Identify calling clients (internal/external)
- [ ] Implement Lean replacement route with feature parity
- [ ] Add integration test comparing legacy vs replacement responses
- [ ] Deploy new route and migrate callers
- [ ] Monitor errors for two weeks
- [ ] Add deprecation header and warning log to legacy route
- [ ] Maintain warning period for 30 days
- [ ] Return HTTP 410 from legacy route
- [ ] Remove route from production and delete file

---

## Instrumentation

- **Middleware**: ✅ `DeprecationHeaderMiddleware` now inserts deprecation headers on legacy routes.
- **Traffic measurement**: `LOG_FILE=/path/to/access.log scripts/measure_route_traffic.py` (sample run using `scripts/sample_access.log`).
- **Alerting**: configure dashboard to trigger if legacy traffic spikes after migration.
- **Guardrails**: `pytest -m architecture -k legacy_api_freeze` blocks new files in `src/engine/api/`; update `LEGACY_API_ALLOWED` when routes are deleted.

---

## Rollback Checklist

- Re-enable legacy router immediately (remove depredation headers if necessary).
- Compare legacy vs replacement responses to isolate divergence.
- Fix parity issues before attempting migration again.

---

## Communication

- **Internal engineering**: announce phase changes in Slack and sprint planning.
- **API customers**: send email + changelog updates at least 30 days before sunset.
- **Documentation**: update API reference with new endpoints and deprecation timeline.

### Sample Update (Slack / Changelog)

```
Heads-up team! 🎯 We are cutting 10% of traffic from the legacy `/api/progressive_questions`
route next Monday (2025-11-03). Replacement lives at `/api/v53/questions/*`.
Actions:
- @questions-pod owns migration + monitoring
- @reliability-oncall will watch latency/errors for 7 days
- Rollback: re-enable legacy route via `main.py` toggle + remove Lean router include
```

Link the deprecation plan in onboarding docs (`README.md` → "API Overview") so new engineers land here first.
