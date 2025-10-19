# UnifiedContextStream Method Categorization

**Date**: 2025-10-18
**Purpose**: Task 1.2-1.3 - Catalog all methods by category for service extraction
**Total Methods**: 44 methods analyzed

---

## Method Distribution

### Category 1: Event Validation (5 methods → EventValidationService)
1. `_load_event_allowlist()` - Line 336
2. `_validate_event_schema()` - Line 350
3. `_validate_event_transition()` - Line 367
4. `_scrub_pii()` - Line 412 (PII scrubbing)
5. `_scrub_structure()` - Line 445 (Nested PII scrubbing)

**Total LOC**: ~120 lines

### Category 2: Evidence Extraction (7 methods → EvidenceExtractionService)
1. `get_evidence_events()` - Line 629
2. `get_consultant_selection_evidence()` - Line 658
3. `get_synergy_evidence()` - Line 666
4. `get_coreops_evidence()` - Line 674
5. `get_contradiction_evidence()` - Line 682
6. `get_evidence_summary()` - Line 690
7. `_summarize_evidence_event()` - Line 1014 (Helper)
8. `export_evidence_for_api()` - Line 1103 (API formatting)

**Total LOC**: ~150 lines

### Category 3: Context Formatting (4 methods → ContextFormattingService)
1. `format_for_llm()` - Line 769 (Main formatting method)
2. `_format_as_xml()` - Line 854
3. `_format_compressed()` - Line 882
4. `_format_as_json()` - Line 902

**Total LOC**: ~80 lines

### Category 4: Persistence (5 methods → ContextPersistenceService)
1. `create_checkpoint()` - Line 1254
2. `restore_from_checkpoint()` - Line 1266
3. `persist_to_database()` - Line 1429 (async)
4. `set_final_analysis_text()` - Line 1449
5. `_build_persistence_record()` - Line 1461

**Total LOC**: ~100 lines

### Category 5: Metrics & Analytics (6 methods → ContextMetricsService)
1. `get_performance_metrics()` - Line 1294
2. `_calculate_session_duration()` - Line 1205
3. `_extract_confidence_from_event()` - Line 1215
4. `_extract_processing_time_from_event()` - Line 1220
5. `_calculate_summary_metrics()` - Line 1354 ⚠️ **HIGHEST COMPLEXITY (CC=18)**
6. `_summarize_event()` - Line 1225 (Helper)

**Total LOC**: ~120 lines

### Category 6: Core Event Management (KEEP in UnifiedContextStream)
1. `__init__()` - Line 259 (Constructor)
2. `add_event()` - Line 459 (Main event recording)
3. `record_event()` - Line 799 (Async wrapper)
4. `get_relevant_context()` - Line 582
5. `get_recent_events()` - Line 606
6. `get_events()` - Line 618
7. `_calculate_initial_relevance()` - Line 907
8. `_recalculate_relevance()` - Line 932
9. `_compress_old_events()` - Line 967
10. `_notify_subscribers()` - Line 1236
11. `subscribe()` - Line 1245

**Total LOC**: ~200 lines

### Category 7: Agent & Engagement Context (KEEP in UnifiedContextStream)
1. `set_agent_context()` - Line 400
2. `clear_agent_context()` - Line 408
3. `set_engagement_context()` - Line 1314
4. `complete_engagement()` - Line 1331

**Total LOC**: ~40 lines

### Category 8: Utility Methods (KEEP in UnifiedContextStream)
1. `get_current_timestamp()` - Line 1290

**Total LOC**: ~5 lines

### Category 9: Singleton Functions (Module Level - Update)
1. `get_unified_context_stream()` - Line 1520
2. `create_new_context_stream()` - Line 1538

---

## Extraction Summary

| Category | Methods | LOC | Target Service | Priority |
|----------|---------|-----|----------------|----------|
| Event Validation | 5 | ~120 | EventValidationService | HIGH |
| Evidence Extraction | 8 | ~150 | EvidenceExtractionService | HIGH |
| Context Formatting | 4 | ~80 | ContextFormattingService | MEDIUM |
| Persistence | 5 | ~100 | ContextPersistenceService | MEDIUM |
| Metrics & Analytics | 6 | ~120 | ContextMetricsService | HIGH (CC=18) |
| **Core (Keep)** | 16 | ~245 | UnifiedContextStream | - |
| **Total Extracted** | **28** | **570** | **5 Services** | - |
| **Total Kept** | **16** | **245** | **Core** | - |

---

## Expected Results

**Before Refactoring**:
- Total LOC: 815 (current file size from wc -l)
- Methods: 44
- Max CC: 18 (`_calculate_summary_metrics()`)

**After Refactoring**:
- UnifiedContextStream: ~245 LOC (core only)
- 5 new services: ~570 LOC distributed
- Max CC in core: <8
- Max CC per service: <10

**Reduction**: 815 LOC → 245 LOC in core (70% reduction) ✅

---

## High-Risk Methods

### 1. `_calculate_summary_metrics()` (Line 1354, CC=18)
- **Complexity**: HIGHEST in file
- **Target Service**: ContextMetricsService
- **Risk**: Complex logic with many branches
- **Mitigation**: Comprehensive unit tests with all edge cases

### 2. `add_event()` (Line 459)
- **Calls**: Validation, scrubbing, transition checks
- **Target**: Delegates to EventValidationService
- **Risk**: Central method, many dependencies
- **Mitigation**: Keep in core, delegate validation only

### 3. `get_evidence_summary()` (Line 690)
- **Complexity**: Aggregates multiple evidence types
- **Target Service**: EvidenceExtractionService
- **Risk**: Used by many consumers
- **Mitigation**: Maintain exact same return format

---

## Dependency Analysis

### Services That Depend on Each Other
- ❌ **NO circular dependencies** - All services operate independently on events list
- ✅ **EventValidationService** → Used by core `add_event()`
- ✅ **EvidenceExtractionService** → Reads from events (no writes)
- ✅ **ContextFormattingService** → Reads from events (no writes)
- ✅ **ContextPersistenceService** → Writes checkpoints (independent)
- ✅ **ContextMetricsService** → Reads from events (no writes)

### External Dependencies
- **EventValidationService**: Needs PII patterns, event allowlist
- **ContextPersistenceService**: Needs IEventPersistence interface
- **All Services**: Need access to events list or specific event data

---

## Next Steps (Task 1.4-1.5)

- [ ] 1.4 Verify all 184 import sites
- [ ] 1.5 Create baseline test coverage report

---

*Generated for Task 1.2-1.3 - Method Categorization Complete*
