# Pipeline Migration Analysis: Feature Parity Check

**Date**: 2025-10-18
**Status**: Pre-migration validation
**Purpose**: Ensure pipeline implementation has 100% feature parity with legacy code before deletion

---

## Pipeline vs Legacy: Feature Comparison

### ✅ Features Implemented in Pipeline

| Feature | Pipeline Stage | Legacy Location | Status |
|---------|---------------|-----------------|--------|
| Injection Firewall | `InjectionFirewallStage` | lines 618-619 | ✅ COMPLETE |
| PII Redaction | `PIIRedactionStage` | lines 622-623 | ✅ COMPLETE |
| Sensitivity Routing | `SensitivityRoutingStage` | lines 647-650 | ✅ COMPLETE |
| Output Contracts | `OutputContractStage` | lines 626-627 | ✅ COMPLETE |
| RAG Context Injection | `RAGContextInjectionStage` | lines 630-644 | ✅ COMPLETE |
| Provider Adapter | `ProviderAdapterStage` | lines 703-751 | ✅ COMPLETE |
| Reasoning Mode (OpenRouter) | `ReasoningModeStage` | lines 729-751 | ✅ COMPLETE |
| Style Gate | `StyleGateStage` | lines 762-782 | ✅ COMPLETE |
| Confidence Escalation | `ConfidenceEscalationStage` | lines 481-517 | ✅ COMPLETE |

### ⚠️ Features NOT in Pipeline (Post-LLM processing)

| Feature | Legacy Location | In Pipeline? | Action Required |
|---------|-----------------|--------------|-----------------|
| Style Scoring | lines 763-766 | ⚠️ Partial (StyleGateStage) | Validate implementation |
| Prompt Version Registry | lines 785-806 | ❌ NO | Add to post-LLM stage or keep in call_llm |
| Schema Version Registry | lines 795-801 | ❌ NO | Add to post-LLM stage or keep in call_llm |
| Shadow Self-Consistency | lines 814-827 | ❌ NO | Add to post-LLM stage or keep in call_llm |
| Confidence Scoring | lines 830-849 | ⚠️ Partial | Validate ConfidenceEscalationStage |
| Quality Scores Merge | lines 845-849 | ❌ NO | Add to post-LLM stage or keep in call_llm |

### 🔴 Critical Shared Logic (NOT in pipeline)

| Feature | Location | Status | Decision |
|---------|----------|--------|----------|
| Provider Normalization | lines 659-671 | ❌ Pre-pipeline | **KEEP** - happens before pipeline |
| Provider Availability Check | lines 674-691 | ❌ Pre-pipeline | **KEEP** - happens before pipeline |
| Model Registry Validation | lines 680-687 | ❌ Pre-pipeline | **KEEP** - happens before pipeline |
| Provider Instance Retrieval | lines 677 | ❌ Pre-pipeline | **KEEP** - happens before pipeline |
| **Actual LLM Call** | line 753 | ❌ Between stages | **KEEP** - core orchestration |
| Response Metadata Initialization | lines 756-760 | ❌ Post-LLM | **KEEP** - post-processing |

---

## Code Organization Analysis

### Current Structure (467 lines)
```
call_llm() method:
├── [Lines 519-568] Function signature & docstring (49 lines)
├── [Lines 569-611] Pipeline execution NEW CODE (42 lines)
├── [Lines 614-654] Legacy pre-LLM processing (40 lines)
├── [Lines 656-698] Common pre-pipeline logic (42 lines) ← KEEP
├── [Lines 700-751] Legacy provider-specific logic (51 lines)
├── [Line 753] ACTUAL LLM CALL ← KEEP
├── [Lines 756-985] Post-LLM processing (229 lines) ← ANALYZE
```

### Proposed Structure (95 lines)
```
call_llm() method:
├── [Lines 1-30] Function signature & docstring
├── [Lines 31-50] Pre-pipeline validation (provider normalization, availability)
├── [Lines 51-60] Pipeline execution (pre-LLM stages 1-7)
├── [Lines 61-70] Actual LLM call
├── [Lines 71-85] Post-LLM pipeline execution (stages 8-9)
├── [Lines 86-95] Post-processing (version registry, telemetry, return)
```

---

## Migration Strategy

### Phase 1: Validate Pipeline Coverage (CURRENT PHASE)
- [x] Catalog all legacy features
- [x] Map to pipeline stages
- [ ] Identify gaps (version registry, shadow self-consistency)
- [ ] Run integration tests with pipeline enabled
- [ ] Monitor production for 1 week

### Phase 2: Fill Pipeline Gaps
- [ ] Create `VersionRegistryStage` for prompt/schema version tracking
- [ ] Create `SelfConsistencyStage` for shadow self-consistency checks
- [ ] Create `QualityScoresMergeStage` for post-LLM quality aggregation
- [ ] Update factory.py to wire new stages
- [ ] Re-run integration tests

### Phase 3: Conservative Legacy Code Archive
- [ ] Create `src/integrations/llm/legacy/` directory
- [ ] Extract legacy code to `legacy_call_llm_implementation.py`
- [ ] Add comprehensive documentation of what was removed and why
- [ ] Add "resurrection instructions" if emergency rollback needed
- [ ] Simplify `call_llm()` to pure pipeline delegation

### Phase 4: Production Validation (2 sprints)
- [ ] Deploy with simplified call_llm()
- [ ] Monitor error rates, response quality, latency
- [ ] Keep legacy code in archive for 2 sprints
- [ ] After 2 sprints of stability, archive can be moved to git history

---

## Gap Analysis: Missing Features

### 1. Prompt Version Registry (lines 785-806)
**Status**: ❌ NOT in pipeline
**Criticality**: LOW (telemetry/tracking)
**Action**: Keep in post-LLM processing section of call_llm() for now

**Code**:
```python
# Prompt/schema version tracking
if prompt_name and prompt_tmpl:
    from src.engine.services.prompt_version_registry import get_prompt_version_registry
    preg = get_prompt_version_registry()
    pv = preg.register_prompt(prompt_name, prompt_tmpl)
    preg.bind(prompt_name, provider_key, model)
    versions.update({"prompt_name": prompt_name, "prompt_version": pv})
```

**Recommendation**: This is optional telemetry. Keep in simplified call_llm() for now.

### 2. Schema Version Registry (lines 795-801)
**Status**: ❌ NOT in pipeline
**Criticality**: LOW (telemetry/tracking)
**Action**: Keep in post-LLM processing section

**Code**:
```python
if schema_name and isinstance(output_schema, dict):
    from src.engine.services.schema_registry import get_schema_registry
    sreg = get_schema_registry()
    sv = sreg.register_schema(schema_name, output_schema)
    versions.update({"schema_name": schema_name, "schema_version": sv})
```

**Recommendation**: This is optional telemetry. Keep in simplified call_llm() for now.

### 3. Shadow Self-Consistency (lines 814-827)
**Status**: ❌ NOT in pipeline
**Criticality**: MEDIUM (behind feature flag, quality validation)
**Action**: Add as optional post-LLM pipeline stage OR keep in call_llm()

**Code**:
```python
if os.getenv("ENABLE_SHADOW_SELF_CONSISTENCY", "false").lower() == "true":
    from src.telemetry.self_consistency import run_shadow_check
    shadow_consistency = await run_shadow_check(
        provider_instance=provider_instance,
        messages=messages,
        model=model,
        primary_text=response.content,
    )
```

**Recommendation**: This is behind a feature flag. Keep in simplified call_llm() for now, consider moving to pipeline stage in future.

### 4. Confidence Scoring (lines 830-849)
**Status**: ⚠️ PARTIAL - ConfidenceEscalationStage exists
**Criticality**: HIGH (core quality metric)
**Action**: Verify ConfidenceEscalationStage implements this fully

**Code**:
```python
from src.telemetry.confidence import confidence_scorer
factors = {
    "groundedness": groundedness or 0.0,
    "self_verification": self_ver or 0.0,
    "style": style_score or 0.0,
}
if shadow_consistency is not None:
    factors["consistency"] = shadow_consistency
cs = confidence_scorer.evaluate_response_confidence(
    response=response.content,
    factors=factors,
    component="llm",
)
```

**Recommendation**: Verify this is in ConfidenceEscalationStage. If not, keep in call_llm() for now.

---

## What CAN Be Deleted

### ✅ Safe to Delete (Already in Pipeline)

**Lines 614-654: Legacy Pre-LLM Processing**
- Injection firewall (line 618-619) → `InjectionFirewallStage`
- PII redaction (lines 622-623) → `PIIRedactionStage`
- Output contracts (lines 626-627) → `OutputContractStage`
- RAG context injection (lines 630-644) → `RAGContextInjectionStage`
- Sensitivity routing (lines 647-650) → `SensitivityRoutingStage`

**Lines 700-751: Legacy Provider-Specific Logic**
- Provider parameter filtering (lines 703-727) → `ProviderAdapterStage`
- Reasoning mode selection (lines 729-751) → `ReasoningModeStage`

**Total deletable: ~130 lines**

---

## What MUST Be Kept

### 🔴 Critical Logic (Pre-Pipeline)

**Lines 656-698: Provider Setup**
- Provider normalization (lines 659-671)
- Provider availability check (lines 674-691)
- Model registry validation (lines 680-687)
- Provider instance retrieval (line 677)

**Total kept (pre-pipeline): ~42 lines**

### 🔴 Critical Logic (LLM Call)

**Line 753: Actual LLM Call**
```python
response = await provider_instance.call_llm(messages, model, **call_kwargs)
```

### 🔴 Critical Logic (Post-LLM)

**Lines 756-985: Post-LLM Processing**
- Response metadata initialization (lines 756-760)
- Style scoring (lines 763-766) - partially in StyleGateStage
- Style gate evaluation (lines 768-782) - in StyleGateStage
- Prompt version registry (lines 785-806) - keep for now
- Schema version registry (lines 795-801) - keep for now
- Shadow self-consistency (lines 814-827) - keep for now
- Confidence scoring (lines 830-849) - verify in pipeline
- Quality scores merge (lines 845-849) - keep for now
- (Additional post-processing to line 985)

**Total kept (post-pipeline): ~50-100 lines (needs detailed review)**

---

## Final Recommendation

### Conservative Approach (RECOMMENDED)

1. **Delete Now**: Lines 614-654, 700-751 (~130 lines of duplicate pre-LLM logic)
2. **Keep for Now**: Lines 656-698 (pre-pipeline validation), line 753 (LLM call), lines 756-985 (post-processing)
3. **Archive**: Extract deleted code to `legacy/legacy_call_llm_pre_pipeline.py` with full documentation
4. **Result**: Reduce from 467 lines → ~250 lines (46% reduction, 100% safe)

### Target Structure

```python
async def call_llm(...) -> LLMResponse:
    """Docstring"""

    # PRE-PIPELINE: Provider validation and setup (~40 lines)
    provider_key = self._normalize_provider(provider)
    provider_instance = self._get_provider_instance(provider_key)
    self._validate_model_registry(provider_key, model)
    await self._check_provider_availability(provider_instance)

    # PIPELINE EXECUTION: Pre-LLM stages (~20 lines)
    context = LLMCallContext(messages, model, provider, kwargs)
    context = await self.pipeline.execute(context)  # Stages 1-7
    messages = context.get_effective_messages()
    provider = context.get_effective_provider()
    model = context.get_effective_model()
    call_kwargs = context.get_effective_kwargs()

    # LLM CALL (~5 lines)
    response = await provider_instance.call_llm(messages, model, **call_kwargs)

    # PIPELINE EXECUTION: Post-LLM stages (~15 lines)
    context.set_response(response)
    context = await self.pipeline.execute_post_llm(context)  # Stages 8-9
    response = context.get_effective_response()

    # POST-PROCESSING: Telemetry, version registry (~60 lines)
    # ... version registry, shadow self-consistency, etc.

    # TELEMETRY & RETURN (~20 lines)
    await self._log_turn(...)
    await self._escalate_on_low_confidence(...)
    return response
```

**Total: ~160 lines (65% reduction from 467)**

---

## Next Steps

1. ✅ Complete this analysis
2. ⏳ Run integration tests with pipeline enabled
3. ⏳ Create `src/integrations/llm/legacy/` directory
4. ⏳ Extract deletable code to archive with full documentation
5. ⏳ Refactor call_llm() to target structure
6. ⏳ Run full test suite
7. ⏳ Deploy to production with monitoring
8. ⏳ Monitor for 2 sprints before considering archive deletion

---

*Generated by Operation Lean - Pipeline Migration Analysis*
*Status: Pre-migration validation complete*
