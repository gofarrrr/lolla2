# PRD: Engine ↔ Core Dependency Inversion Initiative

**Campaign**: Operation Lean – Architecture Guardrails  
**Priority Score**: 4.9 (Critical Architectural Debt)  
**Target Scope**: `src/engine/*` import dependencies on `src/core/*`  
**Baseline**: 146 file-level violations (`rg --files-with-matches "from src\.core" src/engine`)  
**Date**: 2025-10-20

---

## 1. Overview

The infrastructure layer (`src/engine`) currently imports core pipeline code (`src/core`) in 146 files, violating the intended dependency direction (`orchestration → core → services → engine`). These upward imports limit testability, block modular deployment, and make guardrails ineffective. The initiative introduces stable interfaces, migrates infrastructure code to depend on them, and ratchets the violation count down to zero with automated enforcement.

**Problems Identified**
1. Infrastructure modules reach into core implementation details (e.g., `UnifiedContextStream`, `StatefulPipelineOrchestrator`), creating tight coupling.
2. Cross-layer imports prevent reusing `src/core` without `src/engine`, hindering experimentation and microservice extraction.
3. Lack of shared interfaces causes infrastructure to depend on concrete classes instead of contracts.
4. Regression risk: without automated checks, new violations creep in unnoticed.

**Desired Outcome**
- Introduce interface contracts in `src/interfaces/` and migrate `src/engine` to consume them.
- Reduce file-level violations from 146 → 0 in controlled increments.
- Enforce dependency direction via architecture guardrail tests.

---

## 2. Goals

1. **Interface Coverage**: Provide protocol interfaces for core constructs consumed by `src/engine` (`ContextStream`, `PipelineOrchestrator`, `EvidenceEmitter`, `LLMManager`, etc.).
2. **Violation Reduction**: Cut cross-layer imports from 146 → ≤40 within the initiative, with a roadmap to zero.
3. **Automated Enforcement**: Architecture tests fail when new violations are introduced or when counts do not decrease according to milestone plan.
4. **Refactoring Safety**: Maintain runtime behaviour and public APIs; no breaking changes to existing services.
5. **Documentation & Adoption**: Update architecture guide with dependency rules and interface usage examples.

---

## 3. User Stories

**US-1 – Infrastructure Developer**  
*I want to depend on a stable interface for the context stream so I can evolve infrastructure features without coupling to core internals.*

**Acceptance Criteria**
- `src/interfaces/context_stream.py` exposes a protocol used by infrastructure modules.
- Infrastructure code no longer imports `src/core/unified_context_stream`.

**US-2 – Platform Architect**  
*I want automated guardrails that catch regression imports, so that architectural boundaries remain intact.*

**Acceptance Criteria**
- Architecture pytest fails when `src/engine` adds new `src/core` imports.
- Baseline counts tracked in code and CI.

**US-3 – Service Owner**  
*I want migration guidance so I can refactor my module without breaking production.*

**Acceptance Criteria**
- Task list specifies module-by-module migration order.
- PR template checklist ensures tests and regression metrics are updated.

---

## 4. Functional Requirements

### 4.1 Interface Definition
- Define protocols under `src/interfaces/` for core constructs referenced by infrastructure:
  - `ContextStream` (`add_event`, `get_events`, `format_for_llm`, `create_checkpoint`)
  - `PipelineOrchestrator` (execute/run entry points)
  - `ContextMetrics` (summary metrics, confidence)
  - `EvidenceProvider` (evidence extraction operations)
  - `LLMManager` (call semantics, fallbacks)
- Provide default adapters that wrap existing core implementations.

### 4.2 Migration Workflow
- Each migration PR replaces direct imports with interface contracts and adapters.
- Introduce shims (facades) where infrastructure requires subset of functionality.
- Deprecated imports recorded per module (tracking sheet) until eliminated.

### 4.3 Guardrails
- Architecture pytest updated with milestone thresholds (Baseline: 146; Milestone 1: ≤120; Milestone 2: ≤80; Milestone 3: ≤40; Final: 0).
- CI job fails if counts exceed target threshold for current milestone.
- Add scripts to produce diff of violating files for visibility.

### 4.4 Documentation
- Architecture guide section describing dependency rule, interface usage, and workflow.
- ADR or playbook summarizing interface inventory and adoption pattern.
- Developer checklist for new infrastructure modules.

---

## 5. Non-Goals
- Removing legitimate downward imports (`src/core` depending on `src/services` or `src/engine`).
- Full rewrite of infrastructure modules (only dependency surface changes).
- Introducing runtime plugin system or dynamic loading.
- Modifying orchestrator logic or replacing `UnifiedContextStream` internals.

---

## 6. Success Metrics

| Metric | Baseline | Milestone | Target |
|--------|----------|-----------|--------|
| File-level dependency violations | 146 | ≤40 (Phase 3) | 0 |
| Interfaces defined & adopted | 0 | 5 | ≥8 |
| Infrastructure modules migrated | 0 | 30% | 100% |
| Architecture tests in CI | Partial | Enabled | Required & blocking |
| Documentation coverage | Sparse | Section added | Guide + ADR published |

---

## 7. Timeline & Milestones

1. **Phase 1 (Week 1)** – Interface scaffolding + baseline guardrails
2. **Phase 2 (Weeks 2-3)** – Migrate high-traffic modules (`api/`, `core/service facades`)
3. **Phase 3 (Weeks 4-5)** – Migrate remaining modules, reduce violations to ≤40
4. **Phase 4 (Week 6+)** – Finish long-tail dependencies, reach zero, remove shims

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Hidden runtime coupling when swapping implementations | Start with adapter wrappers and run full regression tests |
| Team unaware of new rules | Document in guide, announce in stand-up, add PR checklist |
| Moving target (new imports added) | Guardrails block merges; baseline updated only when violations decrease |
| Interface explosion | Prioritize top 5 constructs; audit new requests during Phase 2 |

---

## 9. Open Questions
- Do we need dedicated adapters for async vs sync context stream operations?
- Should adapters live in `src/interfaces/adapters/` or alongside implementations?
- How to coordinate milestone thresholds with release calendar?
- Who owns updating the baseline counts after each migration?

---

**Status**: PRD Approved – move to task breakdown and execution plan.
