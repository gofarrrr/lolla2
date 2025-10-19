# PRD: Legacy Engine API Sunset & Lean Router Migration

**Campaign**: Operation Lean – API Consolidation  
**Priority Score**: 4.6 (High Operational Impact)  
**Scope**: Migrate endpoints from `src/engine/api/` to `src/api/routes/` and retire legacy stack  
**Baseline Traffic**: TBD via `scripts/measure_route_traffic.py`  
**Date**: 2025-10-20

---

## 1. Overview

The codebase hosts two API stacks: the legacy FastAPI routers in `src/engine/api/` and the Lean routers in `src/api/routes/`. Maintaining both increases onboarding friction, duplicates business logic, and complicates quality assurance. This initiative executes the deprecation plan (see `DEPRECATION_PLAN.md`) by migrating remaining high-traffic routes, adding parity tests, and removing the legacy stack once traffic is near zero.

**Key Drivers**
- Reduce cognitive load and routing inconsistencies.
- Centralize API governance (auth, rate limiting, observability).
- Simplify testing — new suites already target Lean routes.
- Prepare for future service decomposition.

---

## 2. Goals

1. **Traffic Migration**: Move ≥95% of production traffic to Lean routers.  
2. **Parity Assurance**: Provide route parity tests ensuring identical responses.  
3. **Deprecation Enforcement**: Add headers/logs warning clients of removal timelines.  
4. **Legacy Removal**: Delete `src/engine/api/` once traffic consistently <5% for 30 days.  
5. **Documentation & Communication**: Maintain a clear migration plan and notify stakeholders.

---

## 3. User Stories

**US-1 – API Consumer**  
*I need clear notice and equivalent endpoints so I can migrate without production risk.*

**Acceptance Criteria**
- Deprecation headers clearly state migration deadline.
- New endpoints documented with request/response examples.
- Clients receive migration guides or automated notifications.

**US-2 – Backend Developer**  
*I want a single place (`src/api/routes/`) to add or modify endpoints, so development is predictable.*

**Acceptance Criteria**
- New features land only in Lean routers.
- Legacy routers point to deprecation docs.
- API unit/integration tests cover Lean endpoints only.

**US-3 – QA Engineer**  
*I want parity tests that compare legacy and new endpoints to ensure behaviour matches.*

**Acceptance Criteria**
- Integration tests hitting both stacks produce identical responses (status code, payload, headers).
- Tests run in CI pipeline.

---

## 4. Functional Requirements

### 4.1 Migration Workflow
- Identify top-priority routes using access logs.
- For each route:
  - Reproduce functionality in `src/api/routes/<module>.py`.
  - Share dependencies via services/interfaces — no logic duplication.
  - Add swagger tagging and metadata consistent with Lean style.

### 4.2 Parity Testing
- Build fixtures to call both legacy and new endpoints (against local app).
- Compare responses (status, JSON body, headers).  
- Document known intentional differences (e.g., header casing).

### 4.3 Deprecation Instrumentation
- Middleware injects `X-Api-Version: legacy` and `X-Api-Deprecation` headers.
- Log warning on each legacy route invocation with structured payload (route name, client).
- Optional: emit metric to observability stack.

### 4.4 Communication
- Update API docs with migration timeline.
- Publish changelog entries and internal announcements.
- Track client migration status (internal/external).

### 4.5 Deletion Criteria
- After 30 consecutive days with <5% traffic on legacy routes, remove:
  - `src/engine/api/**`
  - Associated tests, dependencies, and registration code in `src/main.py`
- Update Makefile/test suites to ensure no legacy references remain.

---

## 5. Non-Goals
- Rewriting request/response contracts (should remain backward-compatible).
- Changing authentication/authorization rules.
- Implementing entirely new features during migration (deliver parity first).
- Splitting API into microservices (future consideration).

---

## 6. Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Traffic served by Lean routes | TBD | ≥95% |
| Legacy routers remaining | 70+ | 0 |
| Parity test coverage | 0 | ≥90% of migrated routes |
| Deprecation warnings logged | 0 | 100% of legacy hits |
| Documentation accuracy | Partial | Updated & cross-linked |

---

## 7. Timeline

1. **Week 1** – Instrument traffic measurement, prioritize top routes.  
2. **Weeks 2-4** – Migrate routes in batches, add parity tests.  
3. **Week 5** – Enforce deprecation headers/logs, monitor metrics.  
4. **Week 6+** – When legacy traffic <5%, remove stack and celebrate.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Hidden clients relying on legacy semantics | Use traffic logs + contact lists; provide staging environment for testing |
| Parity differences due to undocumented behaviour | Write integration tests early, compare diff outputs with stakeholders |
| Migration stalls due to resource constraints | Prioritize by traffic; address top routes first |
| Dual maintenance continues indefinitely | Hard deadlines in deprecation plan, guardrails blocking new legacy routes |

---

## 9. Open Questions
- Do we provide temporary proxies forwarding legacy paths to new handlers during cutover?
- Are there third-party clients needing special support or early communication?
- Should we introduce versioned API docs (v1 vs v53) or collapse into single canonical doc?

---

**Status**: PRD Approved – ready for task decomposition and execution.
