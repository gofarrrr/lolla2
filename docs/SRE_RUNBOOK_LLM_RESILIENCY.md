# SRE Runbook: LLM Resiliency & Circuit Breaker Operations

**Owner**: SRE Team
**Last Updated**: 2025-10-19
**Status**: Production Ready

---

## Overview

This runbook covers operational procedures for managing the METIS V5.3 LLM resiliency system, including circuit breakers, retry policies, and provider fallback chains.

## Architecture Summary

**Provider Chain**: OpenRouter (Grok-4-Fast) → Anthropic (Claude) → DeepSeek → OpenAI

**Resiliency Features**:
- Retry policy with exponential backoff
- Circuit breakers per provider
- Automatic fallback to secondary providers
- Structured logging and metrics
- PII redaction and injection firewall

---

## Configuration Reference

### Environment Variables

Located in `.env`:

```bash
# Provider Priority
METIS_LLM_PROVIDER_ORDER="deepseek,anthropic,openrouter,claude"

# Retry Configuration
METIS_LLM_TIMEOUT_SECONDS="45"              # Per-provider timeout
METIS_LLM_MAX_RETRIES="3"                   # Max retry attempts
METIS_LLM_RETRY_BASE_DELAY="1.0"            # Base delay for backoff (seconds)

# Circuit Breaker Configuration
METIS_LLM_CIRCUIT_THRESHOLD="3"             # Failures before opening circuit
METIS_LLM_CIRCUIT_COOLDOWN_SECONDS="60"     # Cooldown before retry (seconds)
```

### Default Values

If environment variables are not set:
- Timeout: 45 seconds
- Max Retries: 3
- Base Delay: 1.0 second
- Circuit Threshold: 3 failures
- Circuit Cooldown: 60 seconds

---

## Common Operations

### 1. Check Provider Health

**Symptom**: Users reporting slow responses or errors

**Diagnosis**:
```bash
# Check backend logs for provider failures
tail -f backend_live.log | grep -E "(provider|circuit|fallback)"

# Check for circuit breaker state
grep "circuit.*open" backend_live.log | tail -20

# Check recent LLM attempts
tail -100 backend_live.log | grep "LLM_ATTEMPT"
```

**Expected Output**:
```
INFO:src.integrations.llm.unified_client:✅ OpenRouter/Grok-4-Fast provider initialized (PRIMARY)
INFO:src.integrations.llm.unified_client:✅ Claude/Anthropic provider initialized
INFO:src.integrations.llm.unified_client:✅ DeepSeek provider initialized
```

**Red Flags**:
- `ERROR: Circuit breaker OPEN for provider: openrouter`
- `WARNING: Provider openrouter timeout after 45s`
- `ERROR: All providers exhausted - no fallback available`

---

### 2. Adjust Circuit Breaker Thresholds

**When to Adjust**:
- Provider is unstable but recoverable (increase threshold)
- Provider is consistently failing (decrease threshold to fail faster)
- Testing new provider (lower threshold for safety)

**Procedure**:
1. Update `.env`:
   ```bash
   # Be more tolerant (5 failures before opening circuit)
   METIS_LLM_CIRCUIT_THRESHOLD="5"

   # OR be more aggressive (2 failures before opening)
   METIS_LLM_CIRCUIT_THRESHOLD="2"
   ```

2. Restart backend:
   ```bash
   make dev-restart
   python3 src/main.py
   ```

3. Monitor for 15 minutes:
   ```bash
   tail -f backend_live.log | grep "circuit"
   ```

4. Verify circuit behavior matches expectations

---

### 3. Increase Provider Timeout

**Symptom**: Provider requests timing out prematurely

**When to Adjust**:
- Provider is slow but reliable
- Large/complex prompts requiring more time
- Temporary network latency

**Procedure**:
1. Update `.env`:
   ```bash
   # Increase from 45s to 90s
   METIS_LLM_TIMEOUT_SECONDS="90"
   ```

2. Restart backend

3. Monitor response times:
   ```bash
   grep "generation_time_ms" backend_live.log | tail -50
   ```

**Warning**: Timeouts >120s may impact user experience. Consider if provider is appropriate.

---

### 4. Force Provider Fallback Order

**When to Use**:
- Primary provider is down
- Cost optimization (switch to cheaper provider)
- Testing fallback chain

**Procedure**:
1. Update `.env` provider order:
   ```bash
   # Original: OpenRouter first
   METIS_LLM_PROVIDER_ORDER="openrouter,anthropic,deepseek,openai"

   # Force Claude first (more expensive but reliable)
   METIS_LLM_PROVIDER_ORDER="anthropic,openrouter,deepseek,openai"

   # Force DeepSeek first (cheapest)
   METIS_LLM_PROVIDER_ORDER="deepseek,anthropic,openrouter,openai"
   ```

2. Restart backend

3. Verify provider selection in logs:
   ```bash
   grep "PRIMARY" backend_live.log | tail -1
   ```

**Expected**: Should show new primary provider

---

### 5. Handle Circuit Breaker "OPEN" State

**Symptom**: Circuit breaker stuck in OPEN state

**Diagnosis**:
```bash
# Check circuit state
grep "circuit.*OPEN" backend_live.log

# Check when it opened
grep "Opening circuit breaker for provider" backend_live.log | tail -1
```

**Recovery Options**:

**Option A: Wait for Cooldown (Recommended)**
- Circuit automatically tries HALF_OPEN after cooldown period
- Default: 60 seconds
- No action needed - system self-heals

**Option B: Reduce Cooldown Time**
```bash
# In .env - reduce to 30 seconds
METIS_LLM_CIRCUIT_COOLDOWN_SECONDS="30"
```

**Option C: Restart Backend** (Nuclear option)
```bash
make dev-restart
python3 src/main.py
```
**Warning**: Clears all circuit breaker state. Use only if provider is confirmed healthy.

---

### 6. Investigate Repeated Fallbacks

**Symptom**: Primary provider consistently failing over to secondary

**Diagnosis**:
```bash
# Count fallback frequency
grep "Falling back to provider" backend_live.log | wc -l

# Check fallback reasons
grep "Falling back" backend_live.log | tail -20

# Check provider-specific errors
grep "provider: openrouter.*ERROR" backend_live.log | tail -20
```

**Common Causes**:
1. **Rate Limiting**: `429 Rate Limit Exceeded`
   - **Fix**: Increase `METIS_LLM_RETRY_BASE_DELAY` to 2.0
   - **Or**: Switch provider priority

2. **Server Overload**: `503 Service Unavailable`
   - **Fix**: DeepSeek-specific (20M+ daily users)
   - **Action**: Already have retry logic; monitor if improves

3. **Timeout**: `Timeout after 45s`
   - **Fix**: Increase `METIS_LLM_TIMEOUT_SECONDS`
   - **Or**: Simplify prompts

4. **API Key Issues**: `401 Unauthorized`
   - **Fix**: Check API key in `.env`
   - **Action**: Verify key hasn't expired

---

### 7. Enable/Disable Security Features

**PII Redaction**:
```python
# Located in src/integrations/llm/unified_client.py
# Currently always enabled - to disable, modify code:
self.pii_redaction_enabled = False
```

**Injection Firewall**:
```python
# Located in src/integrations/llm/unified_client.py
self.injection_firewall_enabled = False
```

**Grounding Contract**:
```python
# Located in src/integrations/llm/unified_client.py
self.grounding_contract_enabled = False
```

**⚠️ Warning**: Disabling security features reduces protection. Only disable for debugging.

---

## Monitoring & Alerts

### Key Metrics to Monitor

1. **Provider Success Rate**:
   ```bash
   grep "LLM_ATTEMPT.*success" backend_live.log | wc -l
   grep "LLM_ATTEMPT.*failed" backend_live.log | wc -l
   ```

2. **Fallback Frequency**:
   ```bash
   grep "Falling back to provider" backend_live.log | wc -l
   ```

3. **Circuit Breaker State**:
   ```bash
   grep "circuit" backend_live.log | grep -E "(OPEN|CLOSED|HALF_OPEN)"
   ```

4. **Response Times**:
   ```bash
   grep "generation_time_ms" backend_live.log | awk -F'"' '{print $4}' | sort -n | tail -20
   ```

### Recommended Alerts

**Critical** (Page SRE):
- All providers exhausted (no fallback)
- Circuit breakers OPEN for all providers >5 minutes
- Error rate >10% for 5 consecutive minutes

**Warning** (Slack notification):
- Circuit breaker OPEN for primary provider
- Fallback rate >25% of requests
- Response time p95 >10 seconds

---

## Troubleshooting Guide

### Problem: All Requests Failing

**Check**:
1. Backend running? `ps aux | grep "python3 src/main.py"`
2. API keys configured? `grep "API_KEY" .env | grep -v "your_"`
3. Network connectivity? `curl -I https://api.openai.com`

**Fix**:
```bash
# Restart backend
make dev-restart
python3 src/main.py

# Check health endpoint
curl http://localhost:8000/api/v53/health
```

---

### Problem: Extremely Slow Responses

**Check**:
1. Which provider is being used?
   ```bash
   grep "PRIMARY" backend_live.log | tail -1
   ```

2. Response times:
   ```bash
   grep "generation_time_ms" backend_live.log | tail -20
   ```

**Fix**:
- If DeepSeek: Known slow due to server overload (60-90s)
- Switch to faster provider (OpenRouter/Anthropic)
- Increase timeout if needed

---

### Problem: Circuit Breaker Won't Close

**Check**:
1. When did it open?
   ```bash
   grep "Opening circuit" backend_live.log | tail -1
   ```

2. Has cooldown period passed? (Default: 60s)

3. Is provider still failing?
   ```bash
   grep "provider: openrouter.*ERROR" backend_live.log | tail -5
   ```

**Fix**:
- Wait for cooldown (recommended)
- Reduce cooldown in `.env`
- If provider confirmed healthy, restart backend

---

## Emergency Procedures

### Incident: Primary Provider Down

**Immediate Actions** (< 5 min):
1. Verify fallback is working:
   ```bash
   curl -X POST http://localhost:8000/api/progressive-questions/generate \
     -H "Content-Type: application/json" \
     -d '{"statement":"Test","context":{}}'
   ```

2. If fallback working: **No action needed** (system self-healing)

3. If fallback also failing:
   ```bash
   # Force switch to known-good provider
   # Edit .env:
   METIS_LLM_PROVIDER_ORDER="anthropic,openai,deepseek,openrouter"

   # Restart
   make dev-restart
   python3 src/main.py
   ```

**Communication**:
- Post to #incidents Slack channel
- Update status page if customer-facing
- Log incident for post-mortem

---

### Incident: All Providers Failing

**Immediate Actions**:
1. Check API keys:
   ```bash
   env | grep "_API_KEY"
   ```

2. Test provider connectivity:
   ```bash
   curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
     https://api.anthropic.com/v1/messages
   ```

3. Check for global outage (status pages):
   - OpenRouter: https://status.openrouter.ai
   - Anthropic: https://status.anthropic.com
   - OpenAI: https://status.openai.com

**Escalation**:
- If keys valid + providers up → Escalate to engineering
- If global outage → Wait for provider recovery, communicate to users

---

## Post-Incident Review

After any incident involving LLM resiliency:

1. **Collect Logs**:
   ```bash
   grep -A 5 -B 5 "ERROR" backend_live.log > incident_logs.txt
   ```

2. **Analyze Metrics**:
   - Failure rate timeline
   - Circuit breaker transitions
   - Fallback frequency

3. **Update Configuration**:
   - Adjust thresholds based on learnings
   - Update provider priority if needed

4. **Document**:
   - Add to this runbook
   - Update team knowledge base

---

## Testing Resiliency

### Simulate Provider Failure

Use fake providers from test suite:

```python
# In tests/security/fake_malicious_provider.py
provider = FakeFailingProvider(failure_mode="503", fail_count=2)

# Simulates:
# - First call: 503 error
# - Second call: 503 error
# - Third call: Success
```

### Test Circuit Breaker

```bash
# Run security test suite
pytest tests/security/test_llm_guards.py::test_fake_failing_provider_retries -v
```

### Verify Fallback Chain

```bash
# Temporarily disable primary in .env
# ANTHROPIC_API_KEY=""  # Comment out

# Restart and test
make dev-restart
python3 src/main.py

# Should fallback to next provider automatically
```

---

## Appendix: Log Message Reference

### Success Messages
- `✅ OpenRouter/Grok-4-Fast provider initialized (PRIMARY)`
- `✅ LLM providers initialized: ['openrouter', 'anthropic', 'deepseek', 'openai']`
- `LLM_ATTEMPT: provider=openrouter, status=success, latency=1.2s`

### Warning Messages
- `⚠️ Circuit breaker HALF_OPEN for provider: openrouter (testing)`
- `⚠️ Provider timeout after 45s, attempting retry`
- `⚠️ Falling back to provider: anthropic (reason: timeout)`

### Error Messages
- `❌ Circuit breaker OPEN for provider: openrouter`
- `❌ Provider failed: 503 Service Unavailable`
- `❌ All providers exhausted - no fallback available`

---

## Contact & Escalation

**On-Call SRE**: Check PagerDuty rotation
**Engineering Lead**: @engineering-team
**Slack Channels**: #sre-incidents, #backend-alerts

**Escalation Path**:
1. On-call SRE (immediate)
2. Backend Lead (< 30 min)
3. CTO (critical/all-providers-down)

---

**Document Version**: 1.0
**Next Review**: 2025-11-19 (monthly)
