# Tasks: Engine ↔ Core Dependency Inversion Initiative

**PRD**: `6-prd-engine-core-dependency-inversion.md`  
**Priority**: Critical Architectural Debt  
**Estimated Duration**: 6-8 weeks (phased rollout)

---

## Relevant Assets

- `src/interfaces/` – target home for shared protocols/adapters  
- `src/engine/**` – infrastructure layer with 146 direct `src/core` imports  
- `src/core/**` – existing implementations to be wrapped/adapted  
- `tests/architecture/test_dependency_direction.py` – guardrail test (baseline: 146)  
- `ARCHITECTURE_GUIDE.md`, `docs/ARCHITECTURE_AUDIT_REPORT_2025-10-20.md` – documentation to update  
- `Makefile` – `test-architecture` target (CI integration)

Command references:
- Count violations: `rg --files-with-matches "from src\.core" src/engine | wc -l`
- List files: `rg --files-with-matches "from src\.core" src/engine`
- Run guardrail tests: `make test-architecture`

---

## Task Breakdown

### 1.0 Planning & Baseline
- [x] 1.1 Audit current violation list (export to CSV for tracking)  
- [x] 1.2 Categorize violations by subsystem (`api`, `core`, `services`, `quality`, etc.)  
- [x] 1.3 Identify top five core constructs consumed by infrastructure (ContextStream, PipelineOrchestrator, etc.)  
- [x] 1.4 Confirm guardrail baseline in `tests/architecture/test_dependency_direction.py` (146)  
- [x] 1.5 Draft architecture update note summarizing target milestones and owners

### 2.0 Interface Scaffolding
- [x] 2.1 Create `src/interfaces/context_stream.py` (Protocol + adapter)  
- [x] 2.2 Create `src/interfaces/pipeline_orchestrator.py`  
- [x] 2.3 Create `src/interfaces/llm_manager.py` or move existing interface if available  
- [x] 2.4 Add shared typings for evidence and metrics (`src/interfaces/evidence.py`, `context_metrics.py`)  
- [x] 2.5 Export interfaces from `src/interfaces/__init__.py`  
- [x] 2.6 Write unit tests for adapters (ensuring they proxy to existing implementations)

### 3.0 Guardrail Enhancements
- [x] 3.1 Extend architecture test to emit diff of violating files for debugging
- [x] 3.2 Encode milestone thresholds (e.g., `TARGET_PHASE_1 = 120`)
- [x] 3.3 Add CI job (GitHub Actions workflow) running `make test-architecture` (in .github/workflows/security-tests.yml)
- [x] 3.4 Document failure triage steps in README or CONTRIBUTING (in ARCHITECTURE_GUIDE.md)
- [ ] 3.5 Announce guardrail adoption in engineering channel (communication pending)

### 4.0 Migration Phase 1 (Baseline → ≤120)
- [ ] 4.1 Replace imports in `src/engine/api/**` with context stream interface  
- [ ] 4.2 Update quality/analytics modules that only read context data  
- [ ] 4.3 Update service facades that use orchestrator (`src/engine/services/**`)  
- [ ] 4.4 Verify count ≤120 and update baseline constant accordingly  
- [ ] 4.5 Log progress in tracking document (include list of refactored modules)

### 5.0 Migration Phase 2 (≤80)
- [ ] 5.1 Migrate monitoring + calibration subsystems to interfaces  
- [ ] 5.2 Address persistence-related imports (checkpoint manager, event bridge)  
- [ ] 5.3 Refactor integration providers (LLM/research) to use interface adapters  
- [ ] 5.4 Reduce baseline constant to ≤80 and confirm guardrail passes  
- [ ] 5.5 Capture lessons learned + unblockers

### 6.0 Migration Phase 3 (≤40)
- [ ] 6.1 Tackle remaining long-tail modules (e.g., flywheel, experiments)  
- [ ] 6.2 Introduce fallback shims only where absolutely needed; file tickets for follow-up removals  
- [ ] 6.3 Drop baseline to ≤40 once tests pass  
- [ ] 6.4 Audit for unused adapters or dead code created during migration

### 7.0 Finalization (0 Violations)
- [ ] 7.1 Remove temporary adapters/shims once no longer referenced  
- [ ] 7.2 Set guardrail baseline to 0 and enforce strictly  
- [ ] 7.3 Archive violation tracking sheet with before/after metrics  
- [ ] 7.4 Update architecture guide with final dependency diagram  
- [ ] 7.5 Record outcomes in architecture ADR (success metrics, rollout notes)

### 8.0 Communication & Documentation
- [ ] 8.1 Update `ARCHITECTURE_GUIDE.md` with dependency inversion guidance  
- [ ] 8.2 Add section to `docs/ARCHITECTURE_AUDIT_REPORT_2025-10-20.md` summarizing progress  
- [ ] 8.3 Provide migration playbook (step-by-step instructions with code examples)  
- [ ] 8.4 Create internal presentation / Loom for knowledge sharing  
- [ ] 8.5 Schedule milestone reviews (Phase 1/2/3) with stakeholders

### 9.0 Post-Migration Hardening
- [ ] 9.1 Monitor CI for false positives; adjust tests if necessary  
- [ ] 9.2 Add pre-commit hook optional check (informational)  
- [ ] 9.3 Plan follow-up cleanup: convert infrastructure modules to pure adapters where feasible  
- [ ] 9.4 Evaluate opportunity to move select `src/engine` modules into higher layers now that dependencies are clean

---

## Deliverables Checklist
- [x] Interfaces created and exported (src/interfaces/)
- [x] Architecture tests updated with milestones (tests/architecture/)
- [ ] Violation count reduced to target phases (120 → 80 → 40 → 0) - Currently at 139, target Phase 1: ≤120
- [x] Documentation + ADRs updated (ARCHITECTURE_GUIDE.md, docs/ENGINE_CORE_DEPENDENCY_BASELINE.md)
- [ ] Tracking sheet/metric dashboard shared with team (baseline documented, dashboard pending)
- [ ] Post-migration retrospective completed (pending Phase 1 completion)
