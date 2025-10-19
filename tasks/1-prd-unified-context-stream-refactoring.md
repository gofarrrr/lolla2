# PRD: Unified Context Stream Service Layer Extraction

**Priority**: HIGHEST (Score: 5.41)
**Campaign**: Operation Lean
**Target File**: `src/core/unified_context_stream.py`
**Date**: 2025-10-18

---

## 1. Introduction/Overview

The UnifiedContextStream is currently a God Object containing 384 lines with 60+ methods handling multiple unrelated concerns. This refactoring will extract specialized services from the monolithic class, improving maintainability, testability, and extensibility.

**Problem Being Solved**:
- Single file with mixed responsibilities (formatting, persistence, metrics, evidence extraction, validation)
- High fan-in (184 imports) makes changes risky
- Difficult to test in isolation
- Violates Single Responsibility Principle

**Overall Goal**:
Break UnifiedContextStream into 5 specialized services while maintaining backward compatibility via facade pattern and preserving the 184-file import dependency chain.

---

## 2. Goals

1. **Reduce Core File Complexity**: Reduce `unified_context_stream.py` from 384 LOC to ~150 LOC (60% reduction)
2. **Lower Cyclomatic Complexity**: Reduce max CC from 18 to ~8 in core file
3. **Improve Testability**: Create 5 independent test suites instead of 1 massive suite
4. **Maintain Backward Compatibility**: Zero breaking changes to existing 184 import sites
5. **Preserve Fan-In**: Maintain stable public API via facade pattern
6. **Enable Parallel Development**: Allow multiple developers to work on different services simultaneously

---

## 3. User Stories

**As a developer working on evidence extraction**,
I want evidence-related logic in a dedicated service,
So that I can modify evidence extraction without touching formatting or persistence logic.

**As a developer adding new event types**,
I want event validation in a separate service,
So that I can extend validation rules without impacting metrics or formatting.

**As a QA engineer writing tests**,
I want to test context formatting independently,
So that I can verify XML/JSON formatting without initializing the entire context stream.

**As a platform architect**,
I want clear service boundaries,
So that I can understand and reason about the system architecture.

**As a developer using UnifiedContextStream**,
I want my existing import statements to continue working,
So that I don't have to update 184 files across the codebase.

---

## 4. Functional Requirements

### 4.1 Service Extraction

**4.1.1** Extract `EventValidationService` from UnifiedContextStream
- Move event schema validation logic
- Move event transition validation logic
- Move event allowlist loading and checking
- Expose methods: `validate_event_schema()`, `validate_event_transition()`, `is_event_allowed()`

**4.1.2** Extract `EvidenceExtractionService` from UnifiedContextStream
- Move consultant selection evidence extraction
- Move synergy evidence extraction
- Move CoreOps evidence extraction
- Move contradiction evidence extraction
- Move evidence summarization logic
- Expose methods: `get_consultant_selection_evidence()`, `get_synergy_evidence()`, `get_coreops_evidence()`, `get_contradiction_evidence()`, `get_evidence_summary()`

**4.1.3** Extract `ContextFormattingService` from UnifiedContextStream
- Move XML formatting logic
- Move JSON formatting logic
- Move compressed format logic
- Move LLM-specific formatting
- Expose methods: `format_as_xml()`, `format_as_json()`, `format_compressed()`, `format_for_llm()`

**4.1.4** Extract `ContextPersistenceService` from UnifiedContextStream
- Move checkpoint save/restore logic
- Move database persistence logic
- Move final analysis text storage
- Expose methods: `create_checkpoint()`, `restore_from_checkpoint()`, `persist_to_database()`, `set_final_analysis_text()`

**4.1.5** Extract `ContextMetricsService` from UnifiedContextStream
- Move session duration calculation
- Move confidence extraction
- Move processing time extraction
- Move summary metrics calculation
- Move performance metrics aggregation
- Expose methods: `calculate_session_duration()`, `extract_confidence()`, `calculate_summary_metrics()`, `get_performance_metrics()`

### 4.2 Core UnifiedContextStream Responsibilities

**4.2.1** Core class should ONLY handle:
- Event stream management (append-only list)
- Event recording with timestamps
- Recent event retrieval
- Event filtering by type
- Subscriber notifications (pub-sub)
- Agent context management
- Engagement lifecycle management

**4.2.2** Core class should delegate to services:
- Validation → `EventValidationService`
- Evidence extraction → `EvidenceExtractionService`
- Formatting → `ContextFormattingService`
- Persistence → `ContextPersistenceService`
- Metrics → `ContextMetricsService`

### 4.3 Backward Compatibility

**4.3.1** Maintain facade pattern in UnifiedContextStream
- All existing public methods must continue to work
- Internal implementation delegates to new services
- No changes required to 184 import sites

**4.3.2** Preserve singleton pattern
- `get_unified_context_stream()` function continues to work
- `create_new_context_stream()` function continues to work

**4.3.3** Maintain CloudEvents compatibility
- Event schema remains unchanged
- All event types preserved
- Data contracts unchanged

### 4.4 Testing Requirements

**4.4.1** Create unit tests for each service:
- `tests/core/test_event_validation_service.py`
- `tests/core/test_evidence_extraction_service.py`
- `tests/core/test_context_formatting_service.py`
- `tests/core/test_context_persistence_service.py`
- `tests/core/test_context_metrics_service.py`

**4.4.2** Maintain integration tests:
- Existing `test_unified_context_stream.py` should pass without modification
- Add new integration tests for service interactions

**4.4.3** Test coverage targets:
- Each service: ≥90% coverage
- Core UnifiedContextStream: ≥85% coverage
- Integration tests: Cover all critical workflows

---

## 5. Non-Goals (Out of Scope)

**5.1** Changing event schema or data contracts
- Event types remain the same
- CloudEvents format unchanged
- No breaking changes to event structure

**5.2** Modifying external API contracts
- Public methods of UnifiedContextStream remain unchanged
- Function signatures preserved
- Return types unchanged

**5.3** Database schema changes
- No changes to how events are persisted
- Checkpoint format remains the same
- Database tables unchanged

**5.4** Performance optimization
- This is a code organization refactoring, not a performance optimization
- Performance should remain the same or improve slightly

**5.5** Adding new features
- Focus is on extracting existing functionality, not adding new capabilities
- New event types or validation rules are out of scope

---

## 6. Design Considerations

### 6.1 Service Layer Architecture

```
UnifiedContextStream (Facade - ~150 LOC)
├── Core Responsibilities:
│   ├── Event stream management (append-only list)
│   ├── Event recording with timestamps
│   ├── Subscriber notifications
│   └── Engagement lifecycle
│
└── Delegates to Services:
    ├── EventValidationService (~80 LOC)
    │   ├── Schema validation
    │   ├── Transition validation
    │   └── Allowlist checking
    │
    ├── EvidenceExtractionService (~100 LOC)
    │   ├── Consultant selection evidence
    │   ├── Synergy evidence
    │   ├── CoreOps evidence
    │   └── Evidence summarization
    │
    ├── ContextFormattingService (~70 LOC)
    │   ├── XML formatting
    │   ├── JSON formatting
    │   ├── Compressed formatting
    │   └── LLM-specific formatting
    │
    ├── ContextPersistenceService (~60 LOC)
    │   ├── Checkpoint management
    │   ├── Database persistence
    │   └── Final analysis storage
    │
    └── ContextMetricsService (~80 LOC)
        ├── Session duration
        ├── Confidence extraction
        ├── Processing time
        └── Summary metrics
```

### 6.2 Dependency Injection

- Services should be injected into UnifiedContextStream constructor
- Allow optional service overrides for testing
- Default to standard implementations if not provided

### 6.3 Interface Design

- Each service should have a clear interface (Protocol/ABC)
- Services should be independently instantiable
- Services should have minimal dependencies on each other

### 6.4 Migration Strategy

1. Create service interfaces and implementations
2. Add services to UnifiedContextStream constructor
3. Refactor UnifiedContextStream methods to delegate to services
4. Update tests to work with new structure
5. Validate all 184 import sites still work
6. Clean up old inline implementations

---

## 7. Technical Considerations

### 7.1 Import Impact Analysis

**High Risk**: 184 files import UnifiedContextStream
- Must maintain facade pattern to avoid breaking imports
- Cannot change public method signatures
- Must preserve singleton pattern

**Mitigation**:
- All existing public methods delegate to services
- Run comprehensive import analysis before and after
- Automated testing of all import sites

### 7.2 PII Scrubbing

- PII scrubbing logic currently mixed with event validation
- Should remain in EventValidationService
- Must preserve scrubbing behavior exactly

### 7.3 CloudEvents Integration

- Evidence events use CloudEvents format
- Formatting service must maintain CloudEvents compatibility
- Event schema validation must support CloudEvents

### 7.4 Performance Considerations

- Service delegation adds slight overhead (method calls)
- Should be negligible (<1% performance impact)
- Consider caching in services if needed

### 7.5 Circular Dependency Prevention

- Services should not depend on each other
- All services depend only on core data structures
- UnifiedContextStream orchestrates service interactions

---

## 8. Success Metrics

### 8.1 Code Quality Metrics
- ✅ Core file reduced from 384 LOC to ≤150 LOC
- ✅ Max cyclomatic complexity reduced from 18 to ≤8
- ✅ 5 new service files created (~60-100 LOC each)
- ✅ Test coverage maintained at ≥85%

### 8.2 Maintainability Metrics
- ✅ Each service independently testable
- ✅ Clear separation of concerns (0 violations)
- ✅ All public methods under 20 lines
- ✅ All methods under CC=5

### 8.3 Backward Compatibility Metrics
- ✅ Zero breaking changes to 184 import sites
- ✅ All existing tests pass without modification
- ✅ Public API unchanged (method signatures identical)
- ✅ Return types unchanged

### 8.4 Developer Productivity Metrics
- ✅ Time to understand a service: <15 minutes (down from 45 minutes for monolith)
- ✅ Time to add new evidence type: <30 minutes (down from 2 hours)
- ✅ Time to modify formatting: <20 minutes (down from 1 hour)

---

## 9. Open Questions

**Q1**: Should we extract PII scrubbing into its own service or keep it in EventValidationService?
- **Decision Needed**: PII is currently in `_scrub_pii()` and `_scrub_structure()` methods
- **Options**: (A) Keep in EventValidationService, (B) Create PIIScrubbingService
- **Recommendation**: Keep in EventValidationService for now, can extract later if needed

**Q2**: How should we handle service initialization in UnifiedContextStream constructor?
- **Decision Needed**: Should services be lazy-loaded or eager-loaded?
- **Options**: (A) Lazy load on first use, (B) Eager load in __init__
- **Recommendation**: Eager load for predictable performance

**Q3**: Should we create service interfaces (Protocols) or just concrete classes?
- **Decision Needed**: Do we need formal interfaces for services?
- **Options**: (A) Create Protocol interfaces for all services, (B) Skip protocols for now
- **Recommendation**: Create Protocols for clarity and future mock testing

**Q4**: How should we handle the `persistence` parameter currently in UnifiedContextStream?
- **Decision Needed**: Should persistence service be optional?
- **Options**: (A) Always required, (B) Optional with None default
- **Recommendation**: Optional to maintain backward compatibility

**Q5**: Should we version the services (e.g., EventValidationServiceV1)?
- **Decision Needed**: Do we need versioning for future evolution?
- **Options**: (A) Version from the start, (B) Add versioning later if needed
- **Recommendation**: Skip versioning for now, YAGNI principle

---

## Implementation Notes

### Migration Checklist

- [ ] Create service interface definitions (Protocols)
- [ ] Create service implementations
- [ ] Add unit tests for each service
- [ ] Update UnifiedContextStream constructor to accept services
- [ ] Refactor UnifiedContextStream methods to delegate
- [ ] Update integration tests
- [ ] Run full test suite
- [ ] Validate all 184 import sites work
- [ ] Update documentation
- [ ] Deploy with monitoring

### Rollback Plan

If issues arise:
1. Services are additive - can disable delegation via feature flag
2. Keep old inline implementations for 1 sprint
3. Gradual cutover: delegate one method at a time
4. Monitor error rates and performance metrics

---

**Status**: READY FOR IMPLEMENTATION
**Next Step**: Generate detailed task list from this PRD
