# PRD: LLM Resiliency & Security Hardening (DeepSeek Primary)

**Campaign**: Operation Lean – Infrastructure Reliability  
**Priority Score**: 4.4 (High Risk Exposure)  
**Scope**: `src/integrations/llm/unified_client.py`, `src/engine/core/llm_manager.py`, security middleware, retry/circuit-breaker logic  
**Date**: 2025-10-20

---

## 1. Overview

DeepSeek V3.1 is the primary LLM provider. Production traffic reports intermittent 503/429 responses, high latency (>150s) and server overload. Current fallback logic (Claude/Anthropic/OpenRouter) lacks robust metrics, retry limits, and contract tests. Additionally, enterprise security features (PII redaction, injection firewall, grounding contracts) are injected inline with limited automated validation.

This initiative fortifies the Unified LLM Client by hardening retry/circuit breaker strategies, codifying fallbacks, and adding security regression tests.

---

## 2. Goals

1. **Resiliency**: Guarantee automatic fallback to secondary providers within configurable thresholds; cap retries to prevent cascading failures.
2. **Observability**: Capture metrics/logs for each provider attempt (latency, status, fallback reason).
3. **Security**: Enforce automated tests for injection firewall, PII redaction, and grounding contracts.
4. **Configurability**: Externalize provider ordering, retry counts, and timeouts via settings/config.
5. **Documentation**: Update architecture guide with resiliency workflow and operational runbook.

---

## 3. User Stories

**US-1 – Platform SRE**  
*I need circuit breakers and clear metrics so I can observe and mitigate provider outages quickly.*

**Acceptance Criteria**
- Circuit breaker thresholds configurable (fail fast after N errors).
- Metrics emitted per provider fallback.
- Dashboard or logs show fallback reason.

**US-2 – Backend Engineer**  
*I want deterministic tests that prove the Unified LLM Client enforces security guards (PII redaction, injection firewall) so regressions are caught early.*

**Acceptance Criteria**
- Security tests fail if the firewall allows malicious prompts.
- PII redaction tests verify scrubbing before provider call.

**US-3 – Product Owner**  
*I want assurance that user-facing latency remains acceptable even during deep provider outages.*

**Acceptance Criteria**
- Request timeout reduced to safe default (configurable).
- Fallback response includes metadata to inform front-end if quality degraded.

---

## 4. Functional Requirements

### 4.1 Resiliency Mechanics
- Implement circuit breaker pattern (half-open/open states) per provider.
- Configure retry strategy: exponential backoff with max attempt count.
- Timeouts: enforce configurable request timeout per provider (default ≤45s).
- Fallback order default: DeepSeek → Anthropic → OpenRouter → Claude (configurable).
- Surface fallback metadata to context stream for audit/logging.

### 4.2 Observability
- Emit structured logs for each attempt with provider, latency, status, fallback reason.
- Publish counters for provider success/failure/timeout (if metrics stack available).
- Add debug mode to collect sample prompts/responses (redacted) for triage.

### 4.3 Security Validation
- PII redaction enforced before provider invocation (test against sample data).
- Injection firewall blocks high-risk prompts with sanitized output.
- Grounding contract ensures responses satisfy schema; tests should simulate failures.
- Self-verification path ensures high-risk responses trigger revalidation.

### 4.4 Configuration
- Move provider settings to config (e.g., `METIS_LLM_PROVIDER_ORDER`, `METIS_LLM_TIMEOUT_MS`, `METIS_LLM_MAX_RETRIES`).
- Provide default `.env` stanza and update docs.
- Support overrides per environment (dev, staging, prod).

### 4.5 Testing
- Unit tests for retry logic and circuit breaker state transitions.
- Integration tests simulating provider failure cascade.
- Security tests verifying injection firewall, redaction, grounding.
- Contract tests verifying fallback results surfaced to caller.

---

## 5. Non-Goals
- Building a new provider from scratch.
- Changing business logic of analysis pipeline.
- Removing DeepSeek entirely (still primary).
- Replacing existing service architecture or DI container.

---

## 6. Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Mean request latency under DeepSeek outage | >150s | <60s |
| Provider fallback success rate | Unknown | ≥99% of failed primary attempts |
| Circuit breaker accuracy | None | Detect 3 consecutive failures, open for configurable cooldown |
| Security regression tests | 0 | ≥5 covering PII/injection/grounding |
| Config coverage | Hard-coded | 100% settings externalized |

---

## 7. Timeline

1. **Week 1** – Design circuit breaker & retry strategy, align with SRE.  
2. **Weeks 2-3** – Implement resiliency logic, add observability hooks.  
3. **Week 4** – Build security regression suite and integrate into CI.  
4. **Week 5** – Roll out config changes, update docs, monitor in staging.  
5. **Week 6** – Production rollout with fallback monitoring.

---

## 8. Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Circuit breaker misconfiguration causing false positives | Start with disabled state, enable after testing; provide admin override |
| Tests require network access | Use mock providers/fakes for deterministic behaviour |
| Configuration drift across environments | Document settings in `.env.example`, enforce via config loader |
| Increased complexity in UnifiedLLMClient | Modularize logic into helper classes (RetryPolicy, CircuitBreaker, SecurityPipeline) |

---

## 9. Open Questions
- Do we need provider-specific retry strategies (e.g., fewer retries for rate-limited providers)?  
- Should fallback metadata be exposed to front-end customers or internal only?  
- Can we log prompt summaries safely without leaking PII?  
- What is acceptable max latency under combined fallback scenarios?

---

**Status**: PRD Approved – ready for task plan and execution.
