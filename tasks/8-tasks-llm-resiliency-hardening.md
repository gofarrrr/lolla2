# Tasks: LLM Resiliency & Security Hardening

**PRD**: `8-prd-llm-resiliency-hardening.md`  
**Priority**: High Risk Exposure  
**Estimated Duration**: 6 weeks

---

## Relevant Components

- `src/integrations/llm/unified_client.py` – orchestration of provider calls  
- `src/engine/core/llm_manager.py` – manager pattern entry point  
- `src/engine/security/` – PII redaction, injection firewall, grounding contract  
- `src/config/**` – configuration loading  
- `tests/integrations/test_unified_llm_client.py` (to be created)  
- `tests/security/test_llm_guards.py` (to be created)

Commands:
- Run unit tests: `pytest tests/integrations/test_unified_llm_client.py -v`  
- Run security suite: `pytest tests/security -m security -v`  
- Measure latency fallback: integration harness (to build)

---

## Task Breakdown

### 1.0 Discovery & Design
- [ ] 1.1 Review current retry/fallback logic in `UnifiedLLMClient` and `LLMManager`  
- [ ] 1.2 Document current provider ordering, timeouts, and error handling  
- [ ] 1.3 Align with SRE on acceptable latency, retry counts, and circuit breaker thresholds  
- [ ] 1.4 Draft design doc (RetryPolicy, CircuitBreaker, SecurityPipeline components)

### 2.0 Configuration Externalization
- [x] 2.1 Add config entries (env vars + defaults) for provider order, timeouts, retries  
- [x] 2.2 Update `src/config/models.py` or equivalent to expose typed settings  
- [x] 2.3 Update `.env.example` with new knobs and documentation  
- [ ] 2.4 Ensure `LLMManager` consumes settings instead of hard-coded values

### 3.0 Retry & Circuit Breaker Implementation
- [x] 3.1 Introduce `RetryPolicy` helper (exponential backoff, jitter, caps)  
- [x] 3.2 Implement circuit breaker class (closed → open → half-open transitions)  
- [x] 3.3 Integrate policies into `UnifiedLLMClient` call path  
- [x] 3.4 Add hooks to bypass/override (useful for tests and debugging)  
- [x] 3.5 Write unit tests covering state transitions, retry exhaustion, cooldown

### 4.0 Observability Enhancements
- [x] 4.1 Emit structured logs per provider attempt (provider, latency, status, fallback reason)  
- [x] 4.2 Add counters/timers (if metrics system available) or stub interface for future integration  
- [x] 4.3 Include fallback metadata in context events (for audit and transparency)  
- [x] 4.4 Provide CLI or script to summarize log output for operators

### 5.0 Security Regression Suite
- [x] 5.1 Build fake provider(s) to simulate malicious prompts/responses
- [x] 5.2 Write tests ensuring injection firewall blocks high-risk prompts
- [x] 5.3 Verify PII redaction removes sensitive content before sending to provider
- [x] 5.4 Validate grounding contract rejects responses breaking schema
- [x] 5.5 Ensure self-verification path triggers retries when confidence low
- [x] 5.6 Add pytest marker `security` and run in CI nightly

### 6.0 Integration & Load Testing
- [x] 6.1 Develop integration harness simulating provider outages (503/429, timeouts)
- [x] 6.2 Validate fallback order executes correctly and respects retry limits
- [ ] 6.3 Measure latency improvements vs baseline; record in report
- [ ] 6.4 Add regression tests ensuring fallback metadata surfaces to callers

### 7.0 Documentation & Rollout
- [x] 7.1 Update `ARCHITECTURE_GUIDE.md` with resiliency workflow and configuration table
- [x] 7.2 Create runbook for SRE (how to adjust thresholds, override circuit breaker)
- [x] 7.3 Document security guarantees and test suite instructions
- [ ] 7.4 Communicate changes to engineering + support teams (Slack + release notes)

### 8.0 Release & Monitoring
- [ ] 8.1 Deploy to staging, monitor logs/metrics for 48 hours  
- [ ] 8.2 Roll out to production with feature flag or staged rollout  
- [ ] 8.3 Monitor fallback frequency, latency, and security warnings  
- [ ] 8.4 Capture anomalies and adjust configuration as needed

### 9.0 Post-Implementation Review
- [ ] 9.1 Hold retrospective; capture lessons learned  
- [ ] 9.2 Update backlog with additional opportunities (e.g., provider-specific tuning)  
- [ ] 9.3 Archive metrics + reports for future audits  
- [ ] 9.4 Mark PRD goals as achieved and close initiative

---

## Deliverables Checklist
- [x] Configurable retry + circuit breaker logic
- [x] Structured logging/metrics for provider attempts
- [x] Security regression tests (PII, injection, grounding) in CI
- [x] Documentation and runbook updated
- [ ] Observed latency/fallback improvements validated in staging & prod (requires production deployment)
- [ ] Post-mortem completed and filed (post-deployment)
