# METIS V5.3 Architecture Guide (Operation Lean Update)
**The canonical source of truth**

**Last Updated**: 2025-10-20 (Operation Lean architecture audit)  
**Status**: Production architecture with active refactors  
**Audience**: METIS engineering teams (core, integrations, services)

---

## Purpose

This guide explains how the METIS platform is structured **today** so teams can add features, debug issues, and plan refactors without guessing. It reflects the refactoring work completed during Operation Lean and highlights the seams that still rely on legacy wiring.

---

## Critical Development Workflow

- **Always reset Python caches before debugging**: `make dev-restart` stops the backend, clears `__pycache__/`, and restarts services. Skipping this step can leave stale bytecode in place and hide code changes.
- **When to run it**: after pulling latest changes, before investigating “impossible” behaviour, and anytime orchestrator/service code is modified.
- **Optional safeguard**: copy `scripts/pre-commit.example` to `.git/hooks/pre-commit` to get a reminder whenever Python files are staged.

---

## Architecture Snapshot

- 1016 Python modules under `src/` (FastAPI application, cognitive pipeline, infrastructure, tooling).
- Primary entry point remains `src/main.py` (804 LOC) with dependency injection, router registration, and runtime wiring.
- API surface is split across `src/api` (new Lean routes) and `src/engine/api` (legacy & long-tail endpoints) and both mount on the same FastAPI app.
- Cognitive execution relies on `src/orchestration/dispatch_orchestrator.py`, `src/core/stateful_pipeline_orchestrator.py`, and `src/core/unified_context_stream.py`.
- Business capabilities live in `src/services/` (6 reliability services, 9 selection services, 7 application services, 3 integration services plus supporting modules).
- Infrastructure base is the `src/engine/` package (≈500 Python files) providing providers, engines, monitoring, calibration, and persistence.
- Supporting subsystems include `src/integrations/`, `src/storage/`, `src/rag/`, `src/utils/`, `src/models/`, and `src/config/`.

---

## Layered Architecture Overview

The platform still follows a five-domain mental model, but Operation Lean introduced additional seams and service facades. Treat the layers below as **responsibility zones** rather than rigid dependency-only strata; several legacy upward imports remain.

### Interface Layer – API & Entry Points

- `src/main.py` hosts the FastAPI app, initializes the Lean service container, and wires both legacy and new routers. Startup/DI work happens inside `V53ServiceContainer`.
- `src/api/routes/` now contains the refactored analysis, transparency, confidence, documents, engagements, and ideaflow endpoints. These modules expose clean APIRouter instances and are the default home for **new** customer-facing routes.
- `src/engine/api/` still holds 70+ routers for historical features (progressive questions, devils advocate, benchmarking, engagement orchestration). Many of these routers import directly from `src/core` and `src/services`.
- UI assets and templated responses live under `src/ui/` and `src/engine/ui/`.

### Orchestration Layer – System-2 Brain

- All dispatch and pipeline planning logic is in `src/orchestration/`.
- Core modules: `dispatch_orchestrator.py`, `pipeline_manager.py`, `stage_plan_builder.py`, `context_orchestrator.py`, and `contracts.py`.
- Orchestrators depend on service container wiring plus infrastructure helpers (feature flags, Supabase, quality raters). `DispatchOrchestrator` still reaches into `src.engine.core.contextual_lollapalooza_engine` and `src.services.selection` directly.
- Pilot variants live alongside production classes (`*_pilot_b.py`). Treat them as staging code paths.

### Cognitive Core Layer – Pipeline & Context

- Eight-stage pipeline lives in `src/core/stateful_pipeline_orchestrator.py`, `src/core/pipeline_runner.py`, and `src/core/stage_executors/`.
- `src/core/unified_context_stream.py` remains the system’s backbone (1359 LOC). Operation Lean extracted helper services to `src/core/services/` (validation, evidence extraction, formatting, persistence, metrics), but the stream still owns significant orchestration logic and shares state via dependency injection.
- Additional cognitive subsystems: `enhanced_devils_advocate_system.py`, `cognitive_pipeline_chain.py`, `cognitive_consultant_router.py`, `enhanced_parallel_cognitive_forges.py`, and domain-specific helpers under `src/core/critique`, `src/core/context_engineering`, and `src/core/inquiry_complexes`.
- Memory, event bus, audit, caching, and telemetry primitives also live here (`incremental_context_manager.py`, `event_bus.py`, `structured_logging.py`, `distributed_redis_cache.py`).

### Service Layer – Business Modules

- `src/services/` delivers business logic. Service clusters align with Operation Lean goals but the counts differ from pre-Lean claims:
  - **Reliability cluster** (`src/services/reliability/`): 6 services covering failure detection, exploration strategies, validation, feedback, pattern governance, and coordination.
  - **Selection cluster** (`src/services/selection/`): 9 top-level services plus supporting analytics/optimization modules for consultant selection, pattern scoring, chemistry models, and feedback loops.
  - **Application cluster** (`src/services/application/`): 7 services handling registry, lifecycle, performance monitoring, orchestration, and Lean-extracted application orchestration (`analysis_orchestration_service.py`, `system2_classification_service.py`).
  - **Integration cluster** (`src/services/integrations/`): 3 services (flywheel, benchmarking harness, user analytics).
- `src/services/container.py` exposes service wiring used by `DispatchOrchestrator` and `main.py`.
- Services frequently call into `src.engine` for provider access and infrastructure concerns (LLM manager, Perplexity client, quality raters).

### Infrastructure & Platform Layer

- `src/engine/` remains the infrastructure foundation. Key sub-packages:
  - `core/`: resilient manager pattern (`llm_manager.py`, `research_manager.py`, `feature_flags.py`), context intelligence facades, tool decision framework, caching, checkpoint management.
  - `providers/` & `integrations/`: LLM and research providers, DeepSeek/Anthropic/OpenRouter bridges, Perplexity/Exa integrations.
  - `engines/`: specialized engines for synthesis, benchmarking, monitoring, pyramid formatting, and adaptive selection.
  - `api/`: legacy and long-tail FastAPI routers.
  - `agents/`, `calibration/`, `quality/`, `monitoring/`: rating, calibration, analytics, and observability agents.
  - `persistence/`, `database/`, `memory/`: Supabase adapters, event persistence, engagement resume tooling.
- Infrastructure modules routinely import cognitive components (e.g., `UnifiedContextStream`, event bus) for telemetry and auditing. This cross-layer coupling is a known Lean follow-up item.

### Integration, Storage, and Knowledge Subsystems

- `src/integrations/llm/unified_client.py` provides the Lean unified LLM client that wraps providers, security hardening (PII, injection firewall, output contracts), and caching.
- `src/storage/` handles Supabase checkpoints (`supabase_store.py`), Zep memory (`zep_memory.py`), and hybrid storage strategies (`hybrid_storage.py`).
- Retrieval and RAG functionality live in `src/rag/` (pipeline, retriever, embeddings, project-specific flows).
- Shared utilities, telemetry, schemas, and models are under `src/utils/`, `src/telemetry/`, `src/schema/`, `src/models/`, and `src/config/`.

---

## Cross-Cutting Systems

- **Unified Context & Events**: `src/core/unified_context_stream.py`, `src/core/event_bus.py`, and supporting services manage event emission, persistence, and transparency dossiers.
- **Feature Flags & Experimentation**: `src/engine/core/feature_flags.py` together with `src/config/feature_flags.py` and API endpoints in `src/api/feature_flags_api.py`.
- **Security & Compliance**: PII redaction, sensitivity routing, injection firewall, and compliance APIs live in `src/engine/security/` and `src/api/compliance_api.py`.
- **Memory & Persistence**: Zep memory integration (`src/storage/zep_memory.py`), Supabase persistence (`src/services/orchestration_infra`, `src/engine/persistence/`), checkpoint services in both `src/core` and `src/engine`.
- **Telemetry & Monitoring**: Confidence scoring (`src/telemetry/confidence.py`), decision quality ribbon (`src/api/decision_quality_ribbon_api.py`), monitoring dashboards (`src/engine/monitoring/`).
- **LLM Resiliency**: Unified client resilience lives in `src/integrations/llm/` (retry/circuit breaker helpers in `resiliency.py`, observability in `observability.py`, config via `METIS_LLM_*` env vars). Fallback chains emit structured logs + `ContextEventType.LLM_PROVIDER_FALLBACK` events and can be summarised via `scripts/summarize_llm_attempts.py`.
- **Testing**: `tests/api/`, `tests/core/`, `tests/services/` provide unit/integration coverage for selected modules. Coverage is uneven; new modules often lack tests.

---

## Dependency Expectations vs Reality

- **Intended flow**: Interface → Orchestration → Core → Services → Engine → External providers.
- **Current reality**:
  - `src/engine` imports numerous `src/core` classes (`UnifiedContextStream`, event emitters, pipeline contracts) for telemetry and orchestration support.
  - `src/orchestration` and `src/services` both import infrastructure helpers from `src/engine`.
  - `src/api/routes` reference orchestration, services, engine, and integrations directly for lazy initialization.
- **Guidance**:
  - When adding new features, prefer calling downward (higher-level module depending on lower-level contracts). Avoid introducing *new* upward imports from `src/engine` into `src/core` or `src/orchestration`.
  - If an infrastructure feature truly needs core context (e.g., quality scoring), depend on the shared interfaces in `src/interfaces/` (`context_stream_interface.py`, `pipeline_orchestrator_interface.py`, `llm_manager_interface.py`) and provide adapters rather than importing core modules directly.
  - Newly added contracts `ContextMetrics` and `EvidenceExtractor` (with adapters) unlock metrics/evidence access without touching `src/core/services/*`. Prefer these adapters for telemetry, transparency, or monitoring features inside `src/engine`.
  - Before creating a new manager/provider in `src/core`, verify whether an equivalent already exists in `src/engine/core/`.

---

## Development Guidelines

- **API work**: add new customer-facing endpoints under `src/api/routes/` and register the router in `src/main.py`. Leave `src/engine/api/` for legacy maintenance only.
  - V5.3 stateful analysis (`/api/v53/analysis/*`) and progressive questions now live under Lean routes (`src/api/routes/stateful_analysis_routes.py`, `src/api/routes/progressive_questions.py`).
- **Dispatch & system-2**: contribute to `src/orchestration/` modules. Keep consultant database logic, task classification, and NWAY orchestration inside dispatch seams.
- **Pipeline stages**: extend or create executors under `src/core/stage_executors/` and update `StatefulPipelineOrchestrator`.
- **Context management**: use the services in `src/core/services/` when working on validation, formatting, metrics, or persistence. Avoid duplicating this logic in other modules.
- **Business logic**: model enhancements belong in the appropriate service cluster under `src/services/`. Favor constructor injection via the service container.
- **Infrastructure**: integrate new providers, caching layers, or resilience features inside `src/engine/` (core/providers/engines). Keep API routers in `src/engine/api/` only if the feature cannot yet migrate to the Lean `src/api` structure.
- **Integrations**: use `src/integrations/` for external client wrappers (LLMs, research). These clients may depend on `src/engine` security features but should expose simple interfaces to services and orchestration layers.

---

## Testing & Quality Expectations

- Unit tests for services and executors belong in `tests/services/` and `tests/core/`.
- API contract tests reside in `tests/api/`; prefer exercising the Lean routers under `src/api/routes`.
- When touching `UnifiedContextStream` or related services, add targeted tests (fixtures exist in `tests/core/services/`).
- Run `make test-architecture` to enforce dependency direction and `main.py` budgets before submitting PRs.
- LLM security and resiliency guards live under `tests/security/` and `tests/integrations/`; run `pytest -m "security or architecture"` after touching the unified client.
- Operation Lean is moving toward higher automation coverage. New modules should include tests or documented plans for follow-up suites.

---

## Known Gaps & Active Refactors

- `src/core/unified_context_stream.py` remains a 1.3k LOC god file; the extracted services reduce churn but do not yet isolate persistence, metrics, and evidence handling from the stream’s core responsibilities.
- `src/main.py` still exceeds 800 LOC even after route extraction. Remaining startup logic can be decomposed into dedicated bootstrapping modules.
- Duplicate analysis orchestration exists between `src/api/routes/analyze_routes.py` and `src/services/application/analysis_orchestration_service.py`. Consolidation is required to avoid diverging behaviour.
- Two API stacks (`src/api` and `src/engine/api`) increase maintenance burden; the Lean router migration should continue until legacy routes can be retired or wrapped. Legacy responses now emit deprecation headers via `src/api/middleware.py` to help track traffic during the sunset campaign.
- Legacy entry points and experimental scaffolding (`src/engine/main.py`, `src/experiments/`) remain; catalog them as you encounter them and migrate or archive with feature flags.
- Layering violations (`src/engine` importing `src/core`, services wiring into infrastructure) should be addressed via shared interfaces and dependency injection improvements.
- Test coverage is inconsistent. Core orchestration refactors need regression suites before further extractions.

---

## Working Checklist for New Contributors

- Read this guide and review `LEAN_ROADMAP.md` before choosing an entry point.
- Confirm whether the functionality already exists in another module; the codebase contains parallel implementations from earlier phases.
- When unsure about placement, prefer adding façade or interface definitions in `src/interfaces/` and discuss with the architecture team.
- Run targeted tests (`make test`, `pytest tests/<domain>`) after changes. Operation Lean expects “verify before merge.”

---

This document will be reviewed whenever major architectural shifts land (new service clusters, removal of legacy routers, or completion of UnifiedContextStream refactors). Ping the architecture channel if you spot discrepancies or need clarifications.
