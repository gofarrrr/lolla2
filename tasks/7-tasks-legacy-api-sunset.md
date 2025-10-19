# Tasks: Legacy Engine API Sunset & Lean Router Migration

**PRD**: `7-prd-legacy-api-sunset.md`  
**Priority**: High Operational Impact  
**Estimated Duration**: 6 weeks (can overlap with other initiatives)

---

## Key References

- `DEPRECATION_PLAN.md` – migration phases, owners, traffic tracking  
- `scripts/measure_route_traffic.py` – access log analyzer  
- `src/api/routes/` – Lean router destination  
- `src/engine/api/` – legacy router source  
- `ARCHITECTURE_GUIDE.md` – updated development workflow  
- `tests/api/` – target home for parity/integration tests

Commands:
- Measure traffic: `LOG_FILE=/path/to/access.log scripts/measure_route_traffic.py`  
- Run parity tests (to be created): `pytest tests/api/test_route_parity.py -m parity -v`  
- Run architecture suite: `make test-architecture`

---

## Task Breakdown

### 1.0 Preparation
- [x] 1.1 Validate `scripts/measure_route_traffic.py` on staging or production logs  
- [x] 1.2 Produce 30-day traffic report and populate `DEPRECATION_PLAN.md` traffic column  
- [x] 1.3 Identify route owners and add them to the plan  
- [x] 1.4 Communicate migration timeline and expectations to engineering + product  
- [x] 1.5 Add deprecation plan link to onboarding docs / API README

### 2.0 Instrumentation & Guardrails
- [x] 2.1 Implement `DeprecationHeaderMiddleware` to tag legacy responses  
- [x] 2.2 Add structured logging for every legacy route invocation  
- [ ] 2.3 Optional: emit count metric to observability (if infrastructure available)  
- [x] 2.4 Add lint/CI check blocking new files in `src/engine/api/`  
- [x] 2.5 Document failure triage steps in deprecation plan

### 3.0 Route Migration Batches

#### Batch Criteria
- Group routes by domain or shared dependencies.
- Target high-traffic routes first to reduce operational risk.

- [ ] 3.1 Batch A – Core analysis endpoints (e.g., `/api/analysis_execution`, `/api/enhanced_foundation`)  
  - [x] 3.1.1 Recreate handlers under `src/api/routes/` using services  
  - [x] 3.1.2 Add unit tests and integration tests hitting only Lean routes  
  - [x] 3.1.3 Add parity tests comparing legacy vs new responses  
  - [x] 3.1.4 Update `main.py` router registration  
  - [ ] 3.1.5 Monitor traffic and error logs post-cutover
- [ ] 3.2 Batch B – Supporting utilities (progressive questions, devils advocate, transparency)  
- [ ] 3.3 Batch C – Engagement/Enterprise-specific routes  
- [ ] 3.4 Batch D – Long-tail / low-traffic endpoints (migrate or retire)

### 4.0 Testing & Quality
- [x] 4.1 Create `tests/api/test_route_parity.py` covering migrated endpoints  
- [x] 4.2 Add pytest marker `parity` and include in CI  
- [ ] 4.3 Ensure Lean route tests use realistic fixtures (auth, payloads)  
- [ ] 4.4 Run smoke tests after each batch: `pytest tests/api -m "parity or integration"`  
- [ ] 4.5 If parity diverges, document intentional differences in test fixture

### 5.0 Monitoring & Communication
- [ ] 5.1 Track daily traffic ratio (legacy vs Lean) – add chart to team dashboard  
- [ ] 5.2 Send weekly updates summarizing progress and remaining routes  
- [ ] 5.3 Notify clients before major cutovers; provide fallback plan  
- [ ] 5.4 Confirm client migrations complete (internal trackers)

### 6.0 Sunset & Cleanup
- [ ] 6.1 When legacy traffic <5% for 30 consecutive days, schedule removal window  
- [ ] 6.2 Remove `src/engine/api/**` and associated imports from `src/main.py`  
- [ ] 6.3 Delete parity tests that reference legacy stack (keep Lean tests)  
- [ ] 6.4 Update Makefile / scripts to remove legacy references  
- [ ] 6.5 Update documentation to mark migration complete  
- [ ] 6.6 Tag release / changelog entry celebrating consolidation

### 7.0 Post-Mortem & Lessons Learned
- [ ] 7.1 Conduct retrospective on migration process  
- [ ] 7.2 Capture playbook for future API evolutions  
- [ ] 7.3 Archive metrics and parity test diffs for reference  
- [ ] 7.4 Evaluate whether additional API versioning or tooling is needed

---

## Deliverables Checklist
- [ ] Updated `DEPRECATION_PLAN.md` with traffic, owners, target dates  
- [ ] Deprecation headers + logging live in production  
- [ ] Parity tests covering migrated routes (CI enforced)  
- [ ] Legacy traffic consistently <5% for 30 days  
- [ ] Legacy stack removed from repository  
- [ ] Documentation & change logs updated  
- [ ] Retrospective outcomes shared with stakeholders
