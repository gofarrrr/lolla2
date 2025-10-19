# Operation Lean – Architecture Audit Report
**Date**: 2025-10-20  
**Prepared by**: CTO (Codex)  
**Scope**: `/Users/marcin/lolly_v7` (METIS V5.3 platform)

---

## Executive Summary

- Operation Lean delivered meaningful progress (route extraction, service facades, infrastructure hardening), but the live architecture still carries large legacy seams from earlier phases.
- `UnifiedContextStream`, `main.py`, and dual API stacks represent the biggest structural risks; they continue to grow in size and violate intended layering rules.
- Cross-layer imports (especially `src/engine` → `src/core`) remain pervasive and should be addressed with shared contracts and dependency injection improvements.
- Several pilot and backup implementations co-exist with production code; they need cataloguing to prevent divergence.
- Test coverage does not yet protect the refactored modules, leaving high-risk areas without automated safeguards.

---

## Recent Successes Worth Preserving

- `data_contracts.py` refactor (LEAN_ROADMAP.md:415-422) reduced total LOC by 44%, removed circular dependencies via `TYPE_CHECKING`, and validated 84 import sites.
- `main.py` shrink (LEAN_ROADMAP.md:402) took the entry-point from 1,384 LOC to 804 LOC while keeping cyclomatic complexity at 6.
- Method actor / devils advocate refactors shipped with new service seams and matching tests (LEAN_ROADMAP.md:430-447); these should be the template for future extractions.

---

## Operational Risks to Track

- **DeepSeek dependency**: Primary LLM provider subject to 503/429 bursts and high latency (CLAUDE.md:9-19). Circuit breakers and retry limits must be documented and tested.
- **Python bytecode cache**: `__pycache__/` can hide code changes during development (CLAUDE.md:26-51). Teams must run `make dev-restart` before debugging and CI should clear caches prior to test runs.

---

## High-Risk Findings (ordered by severity)

1. **UnifiedContextStream remains a 1.3k LOC god-object**  
   - `src/core/unified_context_stream.py:256` is still responsible for event management, metrics, persistence, and evidence extraction despite service extraction (`src/core/services/`).  
   - The file now imports its own helper services, but the helpers still reach back into the stream (`context_persistence_service.py:102`, `context_persistence_service.py:288`).  
   - Recommendation: complete the façade refactor by isolating state mutations, persistence, and metrics behind interfaces; ensure services do not import the stream type directly. Add regression tests before further extractions.

2. **Analysis orchestration logic is duplicated across API and service layers**  
   - `src/api/routes/analyze_routes.py:270` implements consultant selection and memory-aware generation logic that is almost identical to `src/services/application/analysis_orchestration_service.py:28`.  
   - Divergence risks inconsistent behaviour; the API route bypasses the Lean service container.  
   - Recommendation: make the route depend on `AnalysisOrchestrationService` (inject via FastAPI dependency or service container) and delete duplicated helpers once parity is confirmed by tests.

3. **Layering violations between infrastructure and core layers**  
   - Example: `src/engine/api/unified_analysis_api.py:38` imports `src/core.enhanced_devils_advocate_system` and re-imports `UnifiedContextStream` at runtime (`src/engine/api/unified_analysis_api.py:124`).  
   - Numerous other `src/engine` modules import `src/core` (search produced 261 hits).  
   - Recommendation: introduce slim interfaces in `src/interfaces/` for core concepts (context stream, pipeline orchestrator) and depend on those from `src/engine`. Update the architecture decision record once the dependency direction is enforced.  
   - Progress: Baseline captured in `docs/ENGINE_CORE_DEPENDENCY_BASELINE.md`, guardrail now reads the snapshot, and initial protocols (`context_stream_interface.py`, `pipeline_orchestrator_interface.py`, `llm_manager_interface.py`) are available for migrations.

4. **FastAPI entrypoint still should be decomposed**  
   - `src/main.py:1` is 804 LOC even after Operation Lean. Startup, router registration, and health diagnostics live alongside analytics route wiring.  
   - Recommendation: break the file into `app_factory.py`, `router_registry.py`, and `startup_events.py` modules. Keep `main.py` as a thin bootstrap. Add smoke tests for the app factory.

5. **Dual API stacks increase maintenance surface**  
   - Lean routers live in `src/api/routes/`, while legacy routers persist under `src/engine/api/`. Many features were split between both (e.g., progressive questions under engine, confidence routes under new stack).  
   - Progress: `/api/v53/analysis/*` and `/api/progressive-questions/*` now served from Lean routes; engine modules re-export for backward compatibility. Update documentation once migration reaches 80% of traffic and remove the legacy shims.

6. **Pilot and backup variants drift from production**  
   - Cleanup performed on 2025-10-20 removed dormant pilot/backups (`src/main_backup_pre_refactor.py`, legacy stage executors, unused devils-advocate scaffolding).  
   - Remaining experimental entry points (`src/engine/main.py`, `src/experiments/`) should be feature-flaged or retired with the same workflow.

---

## Additional Opportunities

- **Service cluster accuracy**: `src/services/__init__.py:1` still advertises 17 services across three clusters, yet the current count is 25+ once analytics and orchestration helpers are included. Update the service registry to reflect reality or prune unused modules.
- **Dependency contracts**: Shared data models live in `src/interfaces/` but remain underused. New protocols for context streams, pipeline orchestrators, and LLM managers have landed; migrate infrastructure modules to these contracts to remove direct `src.core` imports.
- **UnifiedLLMClient hard-codes engine dependencies**: `src/integrations/llm/unified_client.py:28` imports PII redaction, injection firewall, and grounding contracts directly from `src/engine`. Consider injecting these capabilities or introducing a security façade to simplify testing.
- **LLM resiliency**: Initial retry/circuit-breaker scaffolding is now in place (`src/integrations/llm/resiliency.py`) with env configuration (`METIS_LLM_*`), but security regression tests and detailed metrics remain outstanding.
- **Supabase/Zep interactions lack tests**: Persistence services call external systems (`src/storage/supabase_store.py:1`, `src/storage/zep_memory.py:1`) without integration coverage. Mock-based tests should verify serialization formats and error handling.
- **Telemetry stores in FastAPI app state**: `src/api/routes/confidence_routes.py:80` relies on `app.state.confidence_store`. This implicit dependency should be moved into a repository abstraction or at least initialised during startup.

---

## Items Requiring Further Investigation

- `src/core/services/context_metrics_service.py` and `context_persistence_service.py` depend on circular imports; confirm whether they are still necessary or can be moved to `src/services/orchestration_infra/`.
- `src/services/selection/` contains multiple analytics helpers (e.g., `pattern_optimizer.py`, `system2_enhanced_chemistry_engine.py`) whose production usage is unclear. Trace orchestrator calls to validate necessity.
- `src/core/method_actor_devils_advocate_backup.py` appears unused. Validate with stakeholders before removal.
- `src/engine/api/enhanced_foundation.py` duplicates functionality from newer routes; check whether clients still rely on it.

---

## Testing & Quality Observations

- Unit tests exist for portions of the context services (`tests/core/services/test_context_persistence_service.py`, `tests/core/services/test_context_formatting_service.py`) but do **not** cover the latest refactors (analysis routes, System-2 services).  
- There is no regression suite for the Lean routers under `tests/api/routes`. Add end-to-end tests for `/api/v53/analyze` and `/api/v53/confidence/*`. A starting point exists with `tests/api/test_deprecation_middleware.py` covering the deprecation headers.  
- No automated checks guard the unified LLM client’s security features; consider adding contract tests that stub engine dependencies.
- Architecture guardrails (`make test-architecture`) now block new `src/engine → src/core` imports; extend them with milestone targets as migrations progress.

---

## Suggested Next Steps (Q4 2025)

1. Finish UnifiedContextStream decomposition (extract persistence/metrics/evidence via interfaces), add regression tests, and update Lean roadmap status.
2. Consolidate analysis execution through `AnalysisOrchestrationService`; deprecate direct logic in `analyze_routes`.
3. Draft ADR describing the target dependency flow and plan interface creation to break `src/engine` → `src/core` imports.
4. Continue API migration: identify top 10 legacy routers by traffic and schedule their relocation into `src/api/routes`.
5. Catalogue any newly discovered pilot/backup modules and run `scripts/cleanup_backups.sh` after ensuring references are removed.
6. Expand automated test coverage for the new service modules and routers; integrate into `Makefile` so developers can run targeted suites.

---

### Appendix: Command Notes

- File counts gathered with `find src -name '*.py' | wc -l`.
- Layer import violations identified using `rg "from src\\.core" src/engine`.
- Line numbers captured via targeted `rg -n` searches during the audit.

---

## Execution Framework (P0 → P3)

- **P0 (Week 1)**: ✅ delete/re-home backup files, land dependency-direction architecture test with baseline, reconcile documented service counts.
- **P1 (Weeks 2-3)**: finish UnifiedContextStream decomposition, consolidate `/api/v53/analyze` through the service layer, publish `DEPRECATION_PLAN.md`, add first LLM security regression.
- **P2 (Month 2)**: roll out shared interfaces, expand route/regression suites, formalize DeepSeek fallbacks and cache warnings in docs.
- **P3 (Monitor)**: keep `main.py` under 850 LOC and CC≤8; refactor only if budget exceeded.

---

## Success Metrics

- **P0 completion**: zero backup files outside `src/experiments/`, architecture tests pass in CI, service documentation matches `find src/services -maxdepth 1 -name '*_service.py'` counts.
- **P1 completion**: `UnifiedContextStream` trimmed below 800 LOC, route/service parity test green, `DEPRECATION_PLAN.md` populated with traffic data, unified LLM security paths covered by >70% tests.
- **P2 completion**: `src/engine → src/core` import violations reduced from 146 to ≤120, new router coverage >80%, DeepSeek circuit-breaker test suite running, docs updated with cache & provider guidance.

---

## Rollback Procedures

- **Dependency-direction test fails**: temporarily skip via `pytest -m "not architecture"`; investigate offending imports, either revert or update baseline count before re-enabling.
- **API migration regressions**: re-enable legacy route, remove deprecation headers, diff request/response payloads to isolate regression, fix parity, retry rollout next sprint.
- **Context-stream refactor fallout**: revert offending commit, restore passing tests, document gap, add missing unit/integration coverage before attempting another extraction.

---

## Communication Plan

- **Engineering team**: announce architecture test and cleanup changes in Slack and sprint kickoff; highlight `make dev-restart` requirement.
- **API consumers**: publish `DEPRECATION_PLAN.md` and circulate via API changelog/email before adding headers.
- **Documentation**: link `ARCHITECTURE_GUIDE.md` from README, add dev workflow section, update onboarding with cache reset instructions.
