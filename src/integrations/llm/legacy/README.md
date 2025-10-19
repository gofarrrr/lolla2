# Legacy LLM Client Code Archive

**Created**: 2025-10-18
**Purpose**: Preserve pre-pipeline implementation for emergency rollback
**Retention**: Keep for 2 production sprints, then move to git history

---

## Why This Archive Exists

During Operation Lean, we completed the pipeline refactoring of `unified_client.py`. The legacy fallback code (372 lines) was removed and archived here for:

1. **Emergency Rollback**: If pipeline has critical bugs, we can restore legacy code
2. **Reference Documentation**: Understand what the old implementation did
3. **Compliance**: Maintain audit trail of major architectural changes
4. **Knowledge Transfer**: Help developers understand evolution of codebase

---

## What Was Removed

### Files in This Archive

1. `legacy_call_llm_pre_pipeline.py` - Pre-LLM pipeline logic (lines 614-654, 700-751)
   - Legacy injection firewall
   - Legacy PII redaction
   - Legacy output contracts
   - Legacy RAG context injection
   - Legacy sensitivity routing
   - Legacy provider-specific parameter filtering
   - Legacy reasoning mode selection

2. `MIGRATION_NOTES.md` - Detailed migration notes and mapping to new pipeline

### What Was NOT Removed

The following logic remains in `unified_client.py` because it's not duplicated in pipeline:

1. **Pre-Pipeline Validation** (kept in call_llm):
   - Provider normalization
   - Provider availability checks
   - Model registry validation

2. **LLM Call Orchestration** (kept in call_llm):
   - Actual provider.call_llm() invocation
   - Response metadata initialization

3. **Post-LLM Processing** (kept in call_llm for now):
   - Prompt version registry
   - Schema version registry
   - Shadow self-consistency (feature flag)
   - Confidence scoring (partial - verify ConfidenceEscalationStage)
   - Quality scores merge

---

## How to Resurrect (Emergency Rollback)

If you need to temporarily restore legacy code due to pipeline bugs:

### Step 1: Disable Pipeline
```python
# In unified_client.py __init__
self.pipeline_enabled = False  # Disable pipeline, use legacy fallback
```

### Step 2: Copy Legacy Code Back
```bash
# Copy pre-pipeline logic back to unified_client.py
cat src/integrations/llm/legacy/legacy_call_llm_pre_pipeline.py >> src/integrations/llm/unified_client.py
```

### Step 3: Verify Tests Pass
```bash
pytest tests/integrations/llm/ -v
```

### Step 4: Deploy with Monitoring
Monitor error rates, latency, response quality closely.

### Step 5: File Bug Report
Document exactly what failed in pipeline and why legacy was restored.

---

## Deletion Timeline

| Date | Action |
|------|--------|
| 2025-10-18 | Legacy code archived |
| 2025-10-25 | Sprint 1 monitoring - keep archive |
| 2025-11-08 | Sprint 2 monitoring - keep archive |
| 2025-11-15 | If no issues, archive can be deleted (move to git history) |

**After 2 sprints of production stability**, this archive can be deleted. The code will remain in git history if needed.

---

## Migration Mapping

### Old → New Implementation

| Old Code (Legacy) | New Code (Pipeline) | File |
|-------------------|---------------------|------|
| `_check_injection_firewall()` | `InjectionFirewallStage` | `pipeline/stages/injection_firewall.py` |
| `_redact_pii_from_messages()` | `PIIRedactionStage` | `pipeline/stages/pii_redaction.py` |
| `_append_contract_prompt()` | `OutputContractStage` | `pipeline/stages/output_contract.py` |
| RAG context inline code | `RAGContextInjectionStage` | `pipeline/stages/rag_context_injection.py` |
| `_apply_sensitivity_routing()` | `SensitivityRoutingStage` | `pipeline/stages/sensitivity_routing.py` |
| Provider param filtering | `ProviderAdapterStage` | `pipeline/stages/provider_adapter.py` |
| Reasoning mode selection | `ReasoningModeStage` | `pipeline/stages/reasoning_mode.py` |
| Style gate inline code | `StyleGateStage` | `pipeline/stages/style_gate.py` |
| `_escalate_on_low_confidence()` | `ConfidenceEscalationStage` | `pipeline/stages/confidence_escalation.py` |

---

## Contact

If you have questions about this archive or need help with rollback:
- Review `PIPELINE_MIGRATION_ANALYSIS.md` in `docs/`
- Check git history for detailed commit messages
- Consult LEAN_ROADMAP.md for refactoring context

---

*This archive is temporary. After 2 sprints of production stability, it can be safely deleted.*
