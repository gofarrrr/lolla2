# Product Requirement Document: main.py Refactoring

**Campaign**: Operation Lean - Target #2
**Priority Score**: 5.08 (2nd Highest)
**Current State**: 822 LOC (LARGEST file in codebase)
**Target State**: ~300 LOC (63% reduction)
**Date**: 2025-10-19

---

## Overview

Refactor `src/main.py` to separate infrastructure/wiring concerns from business logic and API route implementations. Currently, main.py contains 822 lines mixing FastAPI application setup, business logic, stub endpoints, and analysis orchestration. The goal is to transform it into a clean entry point (~300 LOC) that focuses solely on application initialization and dependency injection wiring.

**Current Complexity**:
- **LOC**: 822 (LARGEST file)
- **Max CC**: 6 (low - good, but scattered across many functions)
- **Fan-In**: 0 (entry point - expected)
- **Mixed Concerns**: Infrastructure + Business Logic + API Routes + Stubs

**Key Insight**: Despite low cyclomatic complexity (CC=6), the file violates single responsibility by containing ~522 LOC of business logic that belongs in services and routers.

---

## Goals

### Primary Goals
1. **Reduce LOC from 822 → ~300** (63% reduction)
2. **Extract business logic to service layer** (~200 LOC to services)
3. **Extract API routes to dedicated routers** (~322 LOC to route modules)
4. **Maintain clean entry point** focusing only on app initialization and DI wiring
5. **Zero breaking changes** to existing API endpoints
6. **Improve developer onboarding** with clear separation of concerns

### Secondary Goals
1. Enable parallel development across route modules
2. Improve testability of business logic in isolation
3. Establish pattern for future route additions
4. Reduce cognitive load for new developers
5. Prepare for microservices extraction (if needed)

---

## User Stories

### US-1: As a New Developer
**Story**: As a new developer joining the project, I want to understand the application entry point quickly, so that I can start contributing without getting overwhelmed.

**Acceptance Criteria**:
- main.py contains only infrastructure setup (FastAPI app, CORS, middleware)
- Service dependency injection is clearly documented
- Router registration is self-documenting
- Business logic is clearly separated in service modules

### US-2: As a Backend Developer
**Story**: As a backend developer working on confidence tracing, I want the confidence logic in a dedicated module, so that I can modify it without touching the main entry point.

**Acceptance Criteria**:
- All confidence-related logic is in `src/api/routes/confidence_routes.py`
- Confidence routes are registered via router
- Changes to confidence logic don't require modifying main.py
- Comprehensive tests for confidence routes exist

### US-3: As a Platform Engineer
**Story**: As a platform engineer managing transparency features, I want transparency dossier logic separated, so that I can evolve it independently.

**Acceptance Criteria**:
- Transparency dossier assembly is in `src/api/routes/transparency_routes.py`
- Transparency routes are registered via router
- Dossier logic can be tested in isolation
- Changes don't affect main.py

### US-4: As an AI/ML Engineer
**Story**: As an AI/ML engineer working on analysis orchestration, I want the analysis logic in a service, so that I can reuse it across different entry points.

**Acceptance Criteria**:
- Analysis orchestration is in `src/services/application/analysis_orchestration_service.py`
- Service can be tested independently of FastAPI
- Service can be reused in CLI, background jobs, etc.
- Clear interface for analysis execution

### US-5: As a Maintainer
**Story**: As a codebase maintainer, I want stub endpoints consolidated, so that I can easily identify and remove them when features are implemented.

**Acceptance Criteria**:
- All stub endpoints are in `src/api/routes/stub_routes.py`
- Stubs are clearly marked with TODO comments
- Easy to identify which stubs are still needed
- Gradual removal path is clear

---

## Current State Analysis

### Proper Concerns (~300 LOC)
These should REMAIN in main.py:

1. **FastAPI Application Initialization** (~50 LOC)
   - App instance creation
   - Metadata configuration (title, version, description)
   - CORS configuration
   - Middleware setup

2. **Service Container & Dependency Injection** (~100 LOC)
   - Global container initialization
   - Service registration
   - Adapter configuration
   - Dependency wiring

3. **Router Registration** (~50 LOC)
   - Health check routes
   - API versioning
   - Route module imports and registration
   - WebSocket setup (if applicable)

4. **Application Lifecycle** (~50 LOC)
   - Startup events
   - Shutdown events
   - Context managers
   - Resource cleanup

5. **Server Configuration** (~50 LOC)
   - Uvicorn configuration
   - Port setup
   - Host configuration
   - SSL/TLS setup (if applicable)

### Misplaced Concerns (~522 LOC)
These should be EXTRACTED:

1. **Stub Endpoints** (~200 LOC) → `src/api/routes/stub_routes.py`
   - 40+ `_stub_*` functions
   - Temporary placeholder implementations
   - Should be consolidated for easy removal

2. **Confidence Trace Logic** (~100 LOC) → `src/api/routes/confidence_routes.py`
   - `_get_confidence_data()`
   - `_compute_weighted_score()`
   - `/api/confidence-trace` endpoint
   - Confidence calculation business logic

3. **Transparency Dossier** (~80 LOC) → `src/api/routes/transparency_routes.py`
   - `get_transparency_dossier()`
   - `/api/transparency-dossier` endpoint
   - Evidence assembly logic

4. **Analysis Orchestration** (~100 LOC) → `src/services/application/analysis_orchestration_service.py`
   - `analyze_query()`
   - `_generate_analysis_with_memory()`
   - Core analysis execution logic

5. **System-2 Classification** (~42 LOC) → `src/services/application/system2_classification_service.py`
   - `classify_system2_tier()`
   - Tier classification logic
   - Should be service-based for reusability

---

## Technical Requirements

### TR-1: Infrastructure Layer (main.py)
- **Size**: ~300 LOC maximum
- **Responsibilities**:
  - FastAPI app initialization
  - CORS/middleware configuration
  - Service container setup
  - Router registration
  - Lifecycle management
- **Dependencies**: Services, routers (imports only)
- **No Business Logic**: Zero business logic allowed

### TR-2: Service Layer
- **Location**: `src/services/application/`
- **Services to Create**:
  1. `AnalysisOrchestrationService` (~150 LOC)
     - `async def analyze_query(query, config) -> AnalysisResult`
     - `async def generate_analysis_with_memory(context) -> str`
     - Core analysis execution
  2. `System2ClassificationService` (~60 LOC)
     - `def classify_tier(query, context) -> Tier`
     - Tier classification logic
     - Complexity assessment
- **Testing**: Unit tests for all service methods
- **Contracts**: Clear interfaces for all services

### TR-3: Route Layer
- **Location**: `src/api/routes/`
- **Routers to Create**:
  1. `confidence_routes.py` (~120 LOC)
     - `/api/confidence-trace` endpoint
     - Helper functions for confidence calculation
     - Response formatting
  2. `transparency_routes.py` (~100 LOC)
     - `/api/transparency-dossier` endpoint
     - Dossier assembly logic
     - Evidence formatting
  3. `stub_routes.py` (~220 LOC)
     - All `_stub_*` endpoints consolidated
     - Clear TODO markers
     - Easy removal when implemented
- **Router Pattern**: Use `APIRouter` from FastAPI
- **Dependencies**: Inject services via Depends()
- **Testing**: Integration tests for all routes

### TR-4: Backward Compatibility
- **Requirement**: 100% backward compatibility
- **Validation**:
  - All existing endpoints must work
  - Same request/response formats
  - Same error handling
  - Same authentication/authorization
- **Testing**: E2E tests validate all endpoints

### TR-5: Code Quality
- **Linting**: All new files pass ruff + black
- **Type Hints**: Full type coverage with mypy
- **Documentation**: Docstrings for all public methods
- **Tests**: ≥90% coverage for new modules

---

## Functional Requirements

### FR-1: Service Extraction
- **Analysis Orchestration Service**:
  - Extract `analyze_query()` logic
  - Extract `_generate_analysis_with_memory()` logic
  - Maintain same behavior
  - Add proper error handling
  - Add logging
  - Make async-compatible

- **System-2 Classification Service**:
  - Extract `classify_system2_tier()` logic
  - Support tier classification (1, 2, 3)
  - Complexity assessment
  - Reusable across modules

### FR-2: Route Module Creation
- **Confidence Routes**:
  - `GET /api/confidence-trace`
  - Extract `_get_confidence_data()`
  - Extract `_compute_weighted_score()`
  - Return same response format
  - Add proper error handling

- **Transparency Routes**:
  - `GET /api/transparency-dossier`
  - Extract `get_transparency_dossier()`
  - Assemble evidence events
  - Return formatted dossier

- **Stub Routes**:
  - Consolidate all 40+ stub endpoints
  - Mark with TODO comments
  - Return consistent stub responses
  - Easy to identify for removal

### FR-3: Main.py Restructuring
- **Keep Only**:
  - App initialization
  - CORS configuration
  - Middleware setup
  - Service container setup
  - Router registration
  - Lifecycle hooks

- **Remove**:
  - All business logic
  - All endpoint implementations
  - All stub endpoint definitions
  - All helper functions with business logic

---

## Non-Goals (Out of Scope)

### What We Will NOT Do

1. **Change API Contracts**: No modifications to request/response formats
2. **Add New Features**: Pure refactoring, no feature additions
3. **Microservices Extraction**: Services stay in monolith for now
4. **Database Schema Changes**: No data model modifications
5. **Authentication Refactoring**: Auth logic remains unchanged
6. **WebSocket Refactoring**: WebSocket code (if present) handled separately
7. **Performance Optimization**: Focus is on structure, not performance
8. **Frontend Changes**: No frontend modifications needed
9. **Deployment Changes**: Same deployment process
10. **Configuration Changes**: No new environment variables (unless essential)

---

## Design

### Proposed Architecture

```
main.py (~300 LOC)
├── FastAPI App Initialization
├── CORS & Middleware Setup
├── Service Container (DI)
├── Router Registration
│   ├── Health Routes
│   ├── Confidence Routes
│   ├── Transparency Routes
│   ├── Analysis Routes
│   └── Stub Routes
└── Lifecycle Management

src/services/application/
├── analysis_orchestration_service.py (~150 LOC)
│   ├── AnalysisOrchestrationService
│   ├── analyze_query()
│   └── generate_analysis_with_memory()
└── system2_classification_service.py (~60 LOC)
    ├── System2ClassificationService
    └── classify_tier()

src/api/routes/
├── confidence_routes.py (~120 LOC)
│   ├── router = APIRouter()
│   ├── GET /api/confidence-trace
│   ├── _get_confidence_data()
│   └── _compute_weighted_score()
├── transparency_routes.py (~100 LOC)
│   ├── router = APIRouter()
│   ├── GET /api/transparency-dossier
│   └── get_transparency_dossier()
├── stub_routes.py (~220 LOC)
│   ├── router = APIRouter()
│   └── 40+ stub endpoints
└── analysis_routes.py (if needed)
    ├── router = APIRouter()
    └── POST /api/analyze

tests/
├── services/
│   ├── test_analysis_orchestration_service.py
│   └── test_system2_classification_service.py
└── api/
    ├── test_confidence_routes.py
    ├── test_transparency_routes.py
    └── test_stub_routes.py
```

### Service Layer Contracts

```python
# src/services/application/contracts.py

from typing import Protocol, Dict, Any
from dataclasses import dataclass

class IAnalysisOrchestrationService(Protocol):
    """Analysis orchestration service contract"""

    async def analyze_query(
        self,
        query: str,
        config: Dict[str, Any]
    ) -> AnalysisResult:
        """Execute analysis on query"""
        ...

    async def generate_analysis_with_memory(
        self,
        context: AnalysisContext
    ) -> str:
        """Generate analysis with memory context"""
        ...

class ISystem2ClassificationService(Protocol):
    """System-2 tier classification service contract"""

    def classify_tier(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tier:
        """Classify query into System-2 tier"""
        ...
```

### Router Pattern

```python
# src/api/routes/confidence_routes.py

from fastapi import APIRouter, Depends
from src.services.container import get_service_container

router = APIRouter(prefix="/api", tags=["confidence"])

@router.get("/confidence-trace")
async def get_confidence_trace(
    trace_id: str,
    container = Depends(get_service_container)
):
    """Get confidence trace for a given trace ID"""
    # Implementation moved from main.py
    ...
```

---

## Technical Considerations

### Dependency Injection Strategy
- Use FastAPI's `Depends()` for service injection
- Services registered in global container
- Routers receive services via DI
- Easy to mock for testing

### Error Handling
- Maintain current error handling patterns
- Add structured logging
- Preserve error response formats
- Add monitoring hooks

### Testing Strategy
1. **Unit Tests**: Service layer in isolation
2. **Integration Tests**: Routes with mocked services
3. **E2E Tests**: Full API validation
4. **Smoke Tests**: Health check endpoints
5. **Regression Tests**: Validate no breaking changes

### Migration Path
1. Create service layer first (lowest risk)
2. Create route modules with duplicate code
3. Update main.py to use new routers
4. Validate all endpoints work
5. Remove duplicate code from main.py
6. Run full regression test suite
7. Deploy with rollback plan

### Rollback Strategy
- Keep old main.py as `main_legacy.py` for 2 sprints
- Feature flag for new vs old routing (if needed)
- Quick rollback via symlink swap
- Comprehensive monitoring during migration

---

## Success Metrics

### Code Quality Metrics
- ✅ main.py reduced from 822 LOC → ~300 LOC (≥63% reduction)
- ✅ Max 5 routers registered in main.py
- ✅ Zero business logic functions in main.py
- ✅ All services have ≥90% test coverage
- ✅ All routes have integration tests

### Functional Metrics
- ✅ 100% API backward compatibility
- ✅ Zero breaking changes to existing endpoints
- ✅ All E2E tests passing
- ✅ Response times within ±5% of baseline

### Developer Experience Metrics
- ✅ New developer can understand main.py in <10 minutes
- ✅ Business logic changes don't touch main.py
- ✅ Adding new routes follows clear pattern
- ✅ Service layer is reusable across entry points

### Operational Metrics
- ✅ Zero production incidents during migration
- ✅ Same deployment process
- ✅ No performance regressions
- ✅ Monitoring and logging preserved

---

## Risks & Mitigation

### Risk 1: Breaking API Compatibility
**Impact**: High
**Probability**: Medium
**Mitigation**:
- Comprehensive E2E test suite
- Contract testing for all endpoints
- Gradual migration with duplicate code phase
- Feature flag for rollback

### Risk 2: Performance Regression
**Impact**: Medium
**Probability**: Low
**Mitigation**:
- Performance benchmarks before/after
- Load testing on staging
- Monitoring during rollout
- Easy rollback plan

### Risk 3: Service Dependency Confusion
**Impact**: Medium
**Probability**: Medium
**Mitigation**:
- Clear service contracts
- Comprehensive DI documentation
- Type hints for all dependencies
- Service registration validation

### Risk 4: Incomplete Migration
**Impact**: High
**Probability**: Low
**Mitigation**:
- Detailed task checklist
- Code review checkpoints
- Automated validation scripts
- Migration completion criteria

---

## Open Questions

1. **Q**: Should we use FastAPI's dependency injection or our own container?
   **A**: Use FastAPI's Depends() for routes, global container for services

2. **Q**: How should we handle stub endpoint removal?
   **A**: Consolidate in stub_routes.py, mark with TODOs, remove when features implemented

3. **Q**: Should analysis orchestration be async throughout?
   **A**: Yes, all service methods should be async-compatible for future scalability

4. **Q**: Do we need separate routers for different API versions?
   **A**: Not initially, but structure should support versioning in the future

5. **Q**: Should we extract WebSocket handling?
   **A**: Out of scope for this refactoring, handle separately if needed

---

## Timeline Estimate

**Total Effort**: 12-16 hours

### Phase 1: Service Layer (4-5 hours)
- Create service contracts
- Extract AnalysisOrchestrationService
- Extract System2ClassificationService
- Write unit tests

### Phase 2: Route Modules (4-5 hours)
- Create confidence_routes.py
- Create transparency_routes.py
- Create stub_routes.py
- Write integration tests

### Phase 3: Main.py Refactoring (2-3 hours)
- Update router registration
- Remove business logic
- Clean up imports
- Update documentation

### Phase 4: Testing & Validation (2-3 hours)
- E2E test suite
- Regression testing
- Performance validation
- Documentation updates

---

## Dependencies

### Internal Dependencies
- Service container infrastructure (existing)
- UnifiedContextStream (existing, recently refactored)
- Authentication/authorization system (existing)
- Logging infrastructure (existing)

### External Dependencies
- FastAPI ≥0.104.0
- Pydantic ≥2.0
- Python ≥3.11

---

## Documentation Requirements

1. **Architecture Documentation**
   - Update system architecture diagrams
   - Document service layer pattern
   - Document router pattern

2. **API Documentation**
   - No changes to OpenAPI/Swagger docs
   - Same endpoint documentation

3. **Developer Guide**
   - How to add new routes
   - How to create new services
   - DI best practices

4. **Migration Guide**
   - What changed and why
   - Where to find specific logic
   - How to contribute

---

## Approval Checklist

Before starting implementation:
- [ ] PRD reviewed and approved
- [ ] Architecture design approved
- [ ] Service contracts defined
- [ ] Test strategy agreed
- [ ] Migration path validated
- [ ] Rollback plan documented
- [ ] Success criteria clear
- [ ] Team alignment achieved

---

**Status**: ✅ **READY FOR IMPLEMENTATION**

**Next Step**: Create detailed task breakdown in `2-tasks-main-py-refactoring.md`

---

*Created: 2025-10-19*
*Campaign: Operation Lean - Target #2*
*Follows ai-dev-tasks methodology for structured AI-assisted development*
