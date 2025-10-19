# Tasks: Unified Context Stream Service Layer Extraction

**PRD**: `1-prd-unified-context-stream-refactoring.md`
**Priority**: HIGHEST (Score: 5.41)
**Estimated Duration**: 2-3 sprints

---

## Relevant Files

- `src/core/unified_context_stream.py` - Current God Object (384 LOC, 60+ methods) - TO BE REFACTORED
- `src/core/services/event_validation_service.py` - NEW: Event validation logic
- `src/core/services/evidence_extraction_service.py` - NEW: Evidence extraction logic
- `src/core/services/context_formatting_service.py` - NEW: Formatting logic (XML/JSON)
- `src/core/services/context_persistence_service.py` - NEW: Checkpoint and DB persistence
- `src/core/services/context_metrics_service.py` - NEW: Metrics calculation
- `src/core/services/__init__.py` - NEW: Service exports
- `tests/core/test_event_validation_service.py` - NEW: Validation service tests
- `tests/core/test_evidence_extraction_service.py` - NEW: Evidence service tests
- `tests/core/test_context_formatting_service.py` - NEW: Formatting service tests
- `tests/core/test_context_persistence_service.py` - NEW: Persistence service tests
- `tests/core/test_context_metrics_service.py` - NEW: Metrics service tests
- `tests/core/test_unified_context_stream_integration.py` - Integration tests

### Notes

- This refactoring has HIGH RISK due to 184 import sites across codebase
- Must maintain facade pattern - zero breaking changes allowed
- Run `python3 -m pytest tests/core/ -v` to run all context stream tests
- Use strangler fig pattern: create services alongside existing code, migrate gradually
- Monitor import sites: `grep -r "from src.core.unified_context_stream import" src/`

---

## Tasks

- [ ] **1.0 Setup & Planning**
  - [ ] 1.1 Create `src/core/services/` directory structure
  - [ ] 1.2 Analyze `unified_context_stream.py` and catalog all 60+ methods by category
  - [ ] 1.3 Create method categorization document (validation/evidence/formatting/persistence/metrics)
  - [ ] 1.4 Verify all 184 import sites with automated scan: `grep -r "unified_context_stream" src/`
  - [ ] 1.5 Create baseline test coverage report: `pytest --cov=src.core.unified_context_stream --cov-report=html`

- [ ] **2.0 Create Service Interfaces (Protocols)**
  - [ ] 2.1 Create `src/core/services/protocols.py` with base Protocol definitions
  - [ ] 2.2 Define `IEventValidationService` Protocol with methods: `validate_event_schema()`, `validate_event_transition()`, `is_event_allowed()`, `scrub_pii()`
  - [ ] 2.3 Define `IEvidenceExtractionService` Protocol with methods: `get_consultant_selection_evidence()`, `get_synergy_evidence()`, `get_coreops_evidence()`, `get_contradiction_evidence()`, `get_evidence_summary()`
  - [ ] 2.4 Define `IContextFormattingService` Protocol with methods: `format_as_xml()`, `format_as_json()`, `format_compressed()`, `format_for_llm()`
  - [ ] 2.5 Define `IContextPersistenceService` Protocol with methods: `create_checkpoint()`, `restore_from_checkpoint()`, `persist_to_database()`, `set_final_analysis_text()`
  - [ ] 2.6 Define `IContextMetricsService` Protocol with methods: `calculate_session_duration()`, `extract_confidence()`, `calculate_summary_metrics()`, `get_performance_metrics()`

- [ ] **3.0 Implement EventValidationService**
  - [ ] 3.1 Create `src/core/services/event_validation_service.py` implementing `IEventValidationService`
  - [ ] 3.2 Extract `_validate_event_schema()` logic from UnifiedContextStream (lines ~50-80)
  - [ ] 3.3 Extract `_validate_event_transition()` logic (lines ~80-110)
  - [ ] 3.4 Extract `_load_event_allowlist()` logic (lines ~40-50)
  - [ ] 3.5 Extract `_scrub_pii()` and `_scrub_structure()` logic (lines ~120-160)
  - [ ] 3.6 Create constructor accepting allowlist path and PII patterns
  - [ ] 3.7 Add comprehensive docstrings with examples
  - [ ] 3.8 Write unit tests in `tests/core/test_event_validation_service.py`:
    - [ ] 3.8.1 Test schema validation with valid/invalid events
    - [ ] 3.8.2 Test transition validation with valid/invalid state transitions
    - [ ] 3.8.3 Test allowlist checking
    - [ ] 3.8.4 Test PII scrubbing (emails, phone numbers, SSN patterns)
  - [ ] 3.9 Verify test coverage ≥90%: `pytest --cov=src.core.services.event_validation_service`

- [ ] **4.0 Implement EvidenceExtractionService**
  - [ ] 4.1 Create `src/core/services/evidence_extraction_service.py` implementing `IEvidenceExtractionService`
  - [ ] 4.2 Extract `get_consultant_selection_evidence()` logic (lines ~200-220)
  - [ ] 4.3 Extract `get_synergy_evidence()` logic (lines ~220-230)
  - [ ] 4.4 Extract `get_coreops_evidence()` logic (lines ~230-240)
  - [ ] 4.5 Extract `get_contradiction_evidence()` logic (lines ~240-250)
  - [ ] 4.6 Extract `get_evidence_summary()` logic (lines ~250-290)
  - [ ] 4.7 Extract `_summarize_evidence_event()` helper method (lines ~290-330)
  - [ ] 4.8 Create constructor accepting events list
  - [ ] 4.9 Add comprehensive docstrings with examples
  - [ ] 4.10 Write unit tests in `tests/core/test_evidence_extraction_service.py`:
    - [ ] 4.10.1 Test consultant selection evidence extraction
    - [ ] 4.10.2 Test synergy evidence extraction
    - [ ] 4.10.3 Test CoreOps evidence extraction
    - [ ] 4.10.4 Test contradiction evidence extraction
    - [ ] 4.10.5 Test evidence summarization
    - [ ] 4.10.6 Test empty events handling
  - [ ] 4.11 Verify test coverage ≥90%

- [ ] **5.0 Implement ContextFormattingService**
  - [ ] 5.1 Create `src/core/services/context_formatting_service.py` implementing `IContextFormattingService`
  - [ ] 5.2 Extract `_format_as_xml()` logic (lines ~330-370)
  - [ ] 5.3 Extract `_format_compressed()` logic (lines ~370-400)
  - [ ] 5.4 Extract `_format_as_json()` logic (lines ~400-410)
  - [ ] 5.5 Extract `format_for_llm()` logic (lines ~260-280)
  - [ ] 5.6 Create constructor accepting formatting preferences
  - [ ] 5.7 Add comprehensive docstrings with examples
  - [ ] 5.8 Write unit tests in `tests/core/test_context_formatting_service.py`:
    - [ ] 5.8.1 Test XML formatting with sample events
    - [ ] 5.8.2 Test JSON formatting
    - [ ] 5.8.3 Test compressed formatting
    - [ ] 5.8.4 Test LLM-specific formatting
    - [ ] 5.8.5 Test format switching
  - [ ] 5.9 Verify test coverage ≥90%

- [ ] **6.0 Implement ContextPersistenceService**
  - [ ] 6.1 Create `src/core/services/context_persistence_service.py` implementing `IContextPersistenceService`
  - [ ] 6.2 Extract `create_checkpoint()` logic (lines ~420-430)
  - [ ] 6.3 Extract `restore_from_checkpoint()` logic (lines ~430-445)
  - [ ] 6.4 Extract `persist_to_database()` logic (lines ~450-480)
  - [ ] 6.5 Extract `_build_persistence_record()` helper (lines ~480-510)
  - [ ] 6.6 Extract `set_final_analysis_text()` logic (lines ~515-525)
  - [ ] 6.7 Create constructor accepting persistence interface
  - [ ] 6.8 Add comprehensive docstrings with examples
  - [ ] 6.9 Write unit tests in `tests/core/test_context_persistence_service.py`:
    - [ ] 6.9.1 Test checkpoint creation
    - [ ] 6.9.2 Test checkpoint restoration
    - [ ] 6.9.3 Test database persistence
    - [ ] 6.9.4 Test final analysis storage
    - [ ] 6.9.5 Test error handling (DB unavailable)
  - [ ] 6.10 Verify test coverage ≥90%

- [ ] **7.0 Implement ContextMetricsService**
  - [ ] 7.1 Create `src/core/services/context_metrics_service.py` implementing `IContextMetricsService`
  - [ ] 7.2 Extract `_calculate_session_duration()` logic (lines ~530-540)
  - [ ] 7.3 Extract `_extract_confidence_from_event()` logic (lines ~540-550)
  - [ ] 7.4 Extract `_extract_processing_time_from_event()` logic (lines ~550-560)
  - [ ] 7.5 Extract `_calculate_summary_metrics()` logic (lines ~560-610) - HIGHEST COMPLEXITY (CC=18)
  - [ ] 7.6 Extract `get_performance_metrics()` logic (lines ~615-625)
  - [ ] 7.7 Create constructor accepting events list
  - [ ] 7.8 Add comprehensive docstrings with examples
  - [ ] 7.9 Write unit tests in `tests/core/test_context_metrics_service.py`:
    - [ ] 7.9.1 Test session duration calculation
    - [ ] 7.9.2 Test confidence extraction
    - [ ] 7.9.3 Test processing time extraction
    - [ ] 7.9.4 Test summary metrics calculation (complex logic)
    - [ ] 7.9.5 Test performance metrics aggregation
    - [ ] 7.9.6 Test edge cases (empty events, missing timestamps)
  - [ ] 7.10 Verify test coverage ≥90%

- [ ] **8.0 Refactor UnifiedContextStream to Use Services**
  - [ ] 8.1 Add service dependency injection to `__init__` constructor:
    - [ ] 8.1.1 Add `event_validation_service: IEventValidationService` parameter
    - [ ] 8.1.2 Add `evidence_extraction_service: IEvidenceExtractionService` parameter
    - [ ] 8.1.3 Add `context_formatting_service: IContextFormattingService` parameter
    - [ ] 8.1.4 Add `context_persistence_service: IContextPersistenceService` parameter
    - [ ] 8.1.5 Add `context_metrics_service: IContextMetricsService` parameter
    - [ ] 8.1.6 Add default service creation if None provided (backward compatibility)
  - [ ] 8.2 Refactor validation methods to delegate to EventValidationService:
    - [ ] 8.2.1 Update `add_event()` to call `self.event_validation_service.validate_event_schema()`
    - [ ] 8.2.2 Update `add_event()` to call `self.event_validation_service.validate_event_transition()`
    - [ ] 8.2.3 Update `add_event()` to call `self.event_validation_service.scrub_pii()`
  - [ ] 8.3 Refactor evidence methods to delegate to EvidenceExtractionService:
    - [ ] 8.3.1 Update `get_consultant_selection_evidence()` to delegate
    - [ ] 8.3.2 Update `get_synergy_evidence()` to delegate
    - [ ] 8.3.3 Update `get_coreops_evidence()` to delegate
    - [ ] 8.3.4 Update `get_contradiction_evidence()` to delegate
    - [ ] 8.3.5 Update `get_evidence_summary()` to delegate
  - [ ] 8.4 Refactor formatting methods to delegate to ContextFormattingService:
    - [ ] 8.4.1 Update `format_for_llm()` to delegate
    - [ ] 8.4.2 Update internal XML formatting to delegate
    - [ ] 8.4.3 Update internal JSON formatting to delegate
    - [ ] 8.4.4 Update compressed formatting to delegate
  - [ ] 8.5 Refactor persistence methods to delegate to ContextPersistenceService:
    - [ ] 8.5.1 Update `create_checkpoint()` to delegate
    - [ ] 8.5.2 Update `restore_from_checkpoint()` to delegate
    - [ ] 8.5.3 Update `persist_to_database()` to delegate
    - [ ] 8.5.4 Update `set_final_analysis_text()` to delegate
  - [ ] 8.6 Refactor metrics methods to delegate to ContextMetricsService:
    - [ ] 8.6.1 Update `get_performance_metrics()` to delegate
    - [ ] 8.6.2 Update internal metrics calculation to delegate
  - [ ] 8.7 Remove old inline implementations (mark as deprecated for 1 sprint first)
  - [ ] 8.8 Verify UnifiedContextStream is now ~150 LOC (down from 384)

- [ ] **9.0 Update Singleton Functions**
  - [ ] 9.1 Update `get_unified_context_stream()` to initialize services
  - [ ] 9.2 Update `create_new_context_stream()` to accept optional service overrides
  - [ ] 9.3 Ensure singleton pattern still works with new architecture
  - [ ] 9.4 Add service initialization logging

- [ ] **10.0 Integration Testing**
  - [ ] 10.1 Create `tests/core/test_unified_context_stream_integration.py`
  - [ ] 10.2 Test complete event lifecycle (record → validate → extract evidence → format → persist)
  - [ ] 10.3 Test service interactions (evidence service using formatted data)
  - [ ] 10.4 Test singleton pattern with services
  - [ ] 10.5 Test backward compatibility (all existing tests should pass)
  - [ ] 10.6 Run full test suite: `pytest tests/core/ -v --cov=src.core`
  - [ ] 10.7 Verify zero test failures
  - [ ] 10.8 Verify test coverage ≥85% overall

- [ ] **11.0 Import Site Validation**
  - [ ] 11.1 Scan all 184 import sites: `grep -r "from src.core.unified_context_stream import" src/ | wc -l`
  - [ ] 11.2 Create automated test that imports UnifiedContextStream from 20 random files
  - [ ] 11.3 Verify public API unchanged (no method signature changes)
  - [ ] 11.4 Run smoke tests on key import sites:
    - [ ] 11.4.1 `src/services/stateful_pipeline_orchestrator.py`
    - [ ] 11.4.2 `src/core/enhanced_devils_advocate_system.py`
    - [ ] 11.4.3 `src/engine/orchestration/pipeline_orchestrator.py`
    - [ ] 11.4.4 `src/api/routes/engagements/public.py`
  - [ ] 11.5 Verify zero import errors: `python3 -c "from src.core.unified_context_stream import *"`

- [ ] **12.0 Documentation**
  - [ ] 12.1 Create `docs/UNIFIED_CONTEXT_STREAM_ARCHITECTURE.md` documenting new service layer
  - [ ] 12.2 Create service architecture diagram (ASCII or mermaid.js)
  - [ ] 12.3 Document service responsibilities and interfaces
  - [ ] 12.4 Add migration guide for developers
  - [ ] 12.5 Update UnifiedContextStream docstring with new architecture
  - [ ] 12.6 Add inline comments explaining delegation pattern
  - [ ] 12.7 Create `docs/SERVICE_LAYER_PATTERNS.md` with reusable patterns for future refactorings

- [ ] **13.0 Performance Validation**
  - [ ] 13.1 Create performance benchmark script for event processing
  - [ ] 13.2 Measure baseline performance (before refactoring)
  - [ ] 13.3 Measure new performance (after refactoring)
  - [ ] 13.4 Verify performance regression <1%
  - [ ] 13.5 Profile service delegation overhead
  - [ ] 13.6 Optimize any hot paths if needed

- [ ] **14.0 Deployment & Monitoring**
  - [ ] 14.1 Create feature flag for service layer: `FF_USE_SERVICE_LAYER=true`
  - [ ] 14.2 Deploy with service layer ENABLED but old code preserved
  - [ ] 14.3 Monitor error rates for 1 week
  - [ ] 14.4 Monitor performance metrics for 1 week
  - [ ] 14.5 Monitor memory usage (services may add overhead)
  - [ ] 14.6 If stable, remove old inline implementations
  - [ ] 14.7 If issues, rollback via feature flag

- [ ] **15.0 Cleanup & Finalization**
  - [ ] 15.1 Remove deprecated inline implementations from UnifiedContextStream
  - [ ] 15.2 Remove feature flag (make services mandatory)
  - [ ] 15.3 Update LEAN_ROADMAP.md marking target #2 as COMPLETE
  - [ ] 15.4 Create completion summary document
  - [ ] 15.5 Archive old implementation in `src/core/legacy/` (following unified_client pattern)
  - [ ] 15.6 Final code review with team
  - [ ] 15.7 Celebrate 🎉 - God Object vanquished!

---

## Validation Checkpoints

After each major section, validate:

✅ **After Task 3-7 (Service Creation)**:
- All 5 services have ≥90% test coverage
- All service unit tests passing
- Services independently instantiable

✅ **After Task 8 (Refactoring)**:
- UnifiedContextStream reduced to ~150 LOC
- All delegation methods working
- No inline implementations remain

✅ **After Task 10 (Integration Testing)**:
- All existing tests pass without modification
- Integration tests cover service interactions
- Overall test coverage ≥85%

✅ **After Task 11 (Import Validation)**:
- All 184 import sites verified
- Zero import errors
- Public API unchanged

✅ **After Task 14 (Deployment)**:
- Production stable for 1 week
- No performance regression
- No error rate increase

---

## Risk Mitigation

**High Risk: 184 Import Sites**
- Mitigation: Facade pattern preserves all public methods
- Validation: Automated import scanning before/after

**High Risk: Complex Metrics Logic (CC=18)**
- Mitigation: Comprehensive unit tests for `_calculate_summary_metrics()`
- Validation: Test all edge cases and boundary conditions

**Medium Risk: Service Delegation Overhead**
- Mitigation: Performance benchmarks before/after
- Validation: <1% regression tolerance

**Medium Risk: Circular Dependencies**
- Mitigation: Services depend only on data structures, not each other
- Validation: Import cycle detection

---

## Success Criteria

- [ ] Core file reduced from 384 LOC to ≤150 LOC
- [ ] Max CC reduced from 18 to ≤8
- [ ] 5 services created (~60-100 LOC each)
- [ ] Test coverage ≥85%
- [ ] Zero breaking changes to 184 import sites
- [ ] All existing tests pass without modification
- [ ] Performance regression <1%
- [ ] Deployed to production successfully

---

**Status**: READY FOR EXECUTION
**Estimated Effort**: 80-120 hours (2-3 sprints)
**Risk Level**: HIGH (due to 184 import sites)
**Approach**: Strangler Fig Pattern (gradual migration)
