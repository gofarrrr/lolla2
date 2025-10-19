# Task Breakdown: main.py Refactoring

**Campaign**: Operation Lean - Target #2
**Parent PRD**: `2-prd-main-py-refactoring.md`
**Total Estimated Effort**: 12-16 hours
**Methodology**: ai-dev-tasks sequential task execution
**Date**: 2025-10-19

---

## Task Execution Guidelines

1. **Complete tasks sequentially** - Do not skip ahead
2. **Test after each major task** - Validate before proceeding
3. **Commit after each task group** - Maintain rollback points
4. **Mark tasks complete** - Track progress explicitly
5. **Ask for approval** - Before making breaking changes

---

## Task Groups

### 1.0 Setup & Analysis (1-2 hours)

#### 1.1 Create Project Structure
- [ ] Create `src/services/application/` directory
- [ ] Create `src/api/routes/` directory
- [ ] Create `tests/services/` directory
- [ ] Create `tests/api/` directory
- [ ] Add `__init__.py` files to all new directories

**Validation**:
```bash
ls -la src/services/application/
ls -la src/api/routes/
```

#### 1.2 Analyze main.py Current State
- [ ] Read `src/main.py` completely
- [ ] Identify all functions and their LOC
- [ ] Categorize functions: infrastructure vs business logic vs routes
- [ ] Create inventory of all stub endpoints (count and list)
- [ ] Document current dependencies and imports

**Validation**: Create checklist of all functions to migrate

#### 1.3 Create Service Contracts
- [ ] Create `src/services/application/contracts.py`
- [ ] Define `IAnalysisOrchestrationService` protocol
- [ ] Define `ISystem2ClassificationService` protocol
- [ ] Define data models (`AnalysisResult`, `AnalysisContext`, `Tier`)
- [ ] Add type hints and docstrings

**Validation**:
```bash
python3 -c "from src.services.application.contracts import IAnalysisOrchestrationService; print('✅ Contracts imported')"
```

**Acceptance Criteria**:
- All contract protocols defined with proper type hints
- All data models using Pydantic or dataclasses
- Comprehensive docstrings for all interfaces

---

### 2.0 Service Layer Extraction (4-5 hours)

#### 2.1 Extract AnalysisOrchestrationService
- [ ] Create `src/services/application/analysis_orchestration_service.py`
- [ ] Create `AnalysisOrchestrationService` class
- [ ] Extract `analyze_query()` logic from main.py
  - Current location: Find in main.py
  - Make async-compatible
  - Add error handling
  - Add logging
  - Preserve exact behavior
- [ ] Extract `_generate_analysis_with_memory()` logic
  - Make async-compatible
  - Add proper error handling
  - Maintain same logic
- [ ] Add dependency injection for required services
- [ ] Add proper initialization

**Code Structure**:
```python
class AnalysisOrchestrationService:
    def __init__(
        self,
        context_stream: UnifiedContextStream,
        # other dependencies
    ):
        self.context_stream = context_stream
        # initialization

    async def analyze_query(
        self,
        query: str,
        config: Dict[str, Any]
    ) -> AnalysisResult:
        """Execute analysis on query"""
        # Implementation from main.py
        ...

    async def generate_analysis_with_memory(
        self,
        context: AnalysisContext
    ) -> str:
        """Generate analysis with memory"""
        # Implementation from main.py
        ...
```

**Validation**:
```bash
python3 -c "from src.services.application.analysis_orchestration_service import AnalysisOrchestrationService; print('✅ Service imported')"
```

**Acceptance Criteria**:
- Service class created with proper initialization
- All methods are async
- Same behavior as original main.py functions
- Comprehensive error handling
- Logging added

#### 2.2 Extract System2ClassificationService
- [ ] Create `src/services/application/system2_classification_service.py`
- [ ] Create `System2ClassificationService` class
- [ ] Extract `classify_system2_tier()` logic from main.py
  - Find current implementation
  - Extract logic exactly
  - Add proper type hints
  - Add validation
- [ ] Add tier classification logic
- [ ] Add complexity assessment
- [ ] Support tier 1, 2, 3 classification

**Code Structure**:
```python
class System2ClassificationService:
    def __init__(self):
        # initialization if needed
        pass

    def classify_tier(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tier:
        """Classify query into System-2 tier (1, 2, or 3)"""
        # Implementation from main.py
        ...
```

**Validation**:
```bash
python3 -c "from src.services.application.system2_classification_service import System2ClassificationService; s = System2ClassificationService(); print('✅ Service instantiated')"
```

**Acceptance Criteria**:
- Service created and instantiable
- Logic matches main.py implementation
- Returns proper Tier enum
- Handles edge cases

#### 2.3 Write Service Unit Tests
- [ ] Create `tests/services/test_analysis_orchestration_service.py`
  - Test `analyze_query()` with various inputs
  - Test `generate_analysis_with_memory()` with memory context
  - Test error handling
  - Test async behavior
  - Mock all dependencies
  - Target: ≥90% coverage

- [ ] Create `tests/services/test_system2_classification_service.py`
  - Test tier 1 classification (simple queries)
  - Test tier 2 classification (moderate complexity)
  - Test tier 3 classification (complex reasoning)
  - Test edge cases
  - Target: ≥90% coverage

**Validation**:
```bash
python3 -m pytest tests/services/test_analysis_orchestration_service.py -v
python3 -m pytest tests/services/test_system2_classification_service.py -v
```

**Acceptance Criteria**:
- All service tests passing
- ≥90% code coverage
- Comprehensive test scenarios
- Mocks used appropriately

---

### 3.0 Route Module Extraction (4-5 hours)

#### 3.1 Create Confidence Routes
- [ ] Create `src/api/routes/confidence_routes.py`
- [ ] Create APIRouter with prefix="/api" and tag="confidence"
- [ ] Extract `_get_confidence_data()` from main.py
  - Find function in main.py
  - Copy to confidence_routes.py
  - Make it a route helper (not endpoint)
  - Preserve logic exactly
- [ ] Extract `_compute_weighted_score()` from main.py
  - Find function in main.py
  - Copy to confidence_routes.py
  - Make it a route helper
  - Preserve logic exactly
- [ ] Create `GET /api/confidence-trace` endpoint
  - Extract from main.py if exists
  - Use helper functions
  - Return same response format
  - Add proper error handling

**Code Structure**:
```python
from fastapi import APIRouter, HTTPException, Depends

router = APIRouter(prefix="/api", tags=["confidence"])

def _get_confidence_data(trace_id: str):
    """Helper to get confidence data"""
    # From main.py
    ...

def _compute_weighted_score(data: Dict[str, Any]):
    """Helper to compute weighted score"""
    # From main.py
    ...

@router.get("/confidence-trace")
async def get_confidence_trace(trace_id: str):
    """Get confidence trace for given trace ID"""
    data = _get_confidence_data(trace_id)
    score = _compute_weighted_score(data)
    return {"trace_id": trace_id, "data": data, "score": score}
```

**Validation**:
```bash
python3 -c "from src.api.routes.confidence_routes import router; print(f'✅ Router created with {len(router.routes)} routes')"
```

**Acceptance Criteria**:
- Router created successfully
- All helper functions extracted
- Endpoints return same format as before
- Proper error handling

#### 3.2 Create Transparency Routes
- [ ] Create `src/api/routes/transparency_routes.py`
- [ ] Create APIRouter with prefix="/api" and tag="transparency"
- [ ] Extract `get_transparency_dossier()` from main.py
  - Find function in main.py
  - Copy to transparency_routes.py as endpoint
  - Preserve logic exactly
  - Add proper error handling
- [ ] Create `GET /api/transparency-dossier` endpoint
  - Use extracted dossier logic
  - Return formatted dossier
  - Add evidence assembly
  - Match original response format

**Code Structure**:
```python
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/api", tags=["transparency"])

@router.get("/transparency-dossier")
async def get_transparency_dossier(
    trace_id: Optional[str] = None
):
    """Get transparency dossier with evidence"""
    # Implementation from main.py
    dossier = _assemble_dossier(trace_id)
    return dossier
```

**Validation**:
```bash
python3 -c "from src.api.routes.transparency_routes import router; print('✅ Transparency router created')"
```

**Acceptance Criteria**:
- Router created successfully
- Dossier endpoint functional
- Same response format
- Evidence properly assembled

#### 3.3 Create Stub Routes
- [ ] Create `src/api/routes/stub_routes.py`
- [ ] Create APIRouter with prefix="/api" and tag="stubs"
- [ ] Find all `_stub_*` functions in main.py
- [ ] Count total stub endpoints (should be ~40+)
- [ ] Extract ALL stub functions to stub_routes.py
  - Move function definitions
  - Convert to route endpoints
  - Add TODO comments marking them as stubs
  - Add consistent stub response format
- [ ] Group stubs by feature area (comments)
- [ ] Add helper to generate stub responses

**Code Structure**:
```python
from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["stubs"])

def _stub_response(endpoint_name: str):
    """Generate consistent stub response"""
    return {
        "status": "stub",
        "endpoint": endpoint_name,
        "message": "This endpoint is not yet implemented",
        "todo": "Implement actual logic"
    }

# TODO: Implement actual semantic search
@router.post("/semantic-search")
async def stub_semantic_search():
    """Stub: Semantic search endpoint"""
    return _stub_response("semantic_search")

# TODO: Implement actual recommendation engine
@router.get("/recommendations")
async def stub_recommendations():
    """Stub: Recommendations endpoint"""
    return _stub_response("recommendations")

# ... (all other stubs)
```

**Validation**:
```bash
python3 -c "from src.api.routes.stub_routes import router; print(f'✅ Stub router with {len(router.routes)} endpoints')"
```

**Acceptance Criteria**:
- All stub functions extracted
- ≥40 stub endpoints consolidated
- TODO comments added
- Consistent response format
- Easy to identify for future implementation

#### 3.4 Write Route Integration Tests
- [ ] Create `tests/api/test_confidence_routes.py`
  - Test GET /api/confidence-trace with valid trace_id
  - Test with invalid trace_id
  - Test error handling
  - Mock service dependencies
  - Validate response format

- [ ] Create `tests/api/test_transparency_routes.py`
  - Test GET /api/transparency-dossier
  - Test with various trace_ids
  - Validate dossier structure
  - Test error handling

- [ ] Create `tests/api/test_stub_routes.py`
  - Test all stub endpoints return consistent format
  - Validate stub response structure
  - Ensure all stubs accessible

**Validation**:
```bash
python3 -m pytest tests/api/ -v
```

**Acceptance Criteria**:
- All route tests passing
- Integration with FastAPI validated
- Response formats validated
- Error handling tested

---

### 4.0 Main.py Refactoring (2-3 hours)

#### 4.1 Update Router Registration
- [ ] Import all new routers in main.py
  ```python
  from src.api.routes.confidence_routes import router as confidence_router
  from src.api.routes.transparency_routes import router as transparency_router
  from src.api.routes.stub_routes import router as stub_router
  ```
- [ ] Register routers with app
  ```python
  app.include_router(confidence_router)
  app.include_router(transparency_router)
  app.include_router(stub_router)
  ```
- [ ] Remove old endpoint definitions from main.py
- [ ] Verify router order (health checks first, stubs last)

**Validation**:
```bash
python3 -c "from src.main import app; print(f'✅ App has {len(app.routes)} routes')"
```

**Acceptance Criteria**:
- All routers registered
- No duplicate endpoint definitions
- Proper router ordering

#### 4.2 Update Service Container
- [ ] Register AnalysisOrchestrationService in container
- [ ] Register System2ClassificationService in container
- [ ] Ensure proper dependency injection
- [ ] Add service initialization logging
- [ ] Verify services accessible via container

**Code Updates**:
```python
# In main.py or service_container.py
from src.services.application.analysis_orchestration_service import AnalysisOrchestrationService
from src.services.application.system2_classification_service import System2ClassificationService

# Register services
container.register(AnalysisOrchestrationService)
container.register(System2ClassificationService)
```

**Validation**:
```bash
python3 -c "from src.services.container import global_container; print('✅ Services registered')"
```

**Acceptance Criteria**:
- Services registered in container
- Dependency injection working
- Services instantiable

#### 4.3 Remove Business Logic from main.py
- [ ] Remove `analyze_query()` function (now in service)
- [ ] Remove `_generate_analysis_with_memory()` (now in service)
- [ ] Remove `classify_system2_tier()` (now in service)
- [ ] Remove `_get_confidence_data()` (now in confidence_routes)
- [ ] Remove `_compute_weighted_score()` (now in confidence_routes)
- [ ] Remove `get_transparency_dossier()` (now in transparency_routes)
- [ ] Remove ALL `_stub_*` functions (now in stub_routes)

**Validation**:
```bash
# Check main.py line count
wc -l src/main.py
# Should be ~300 LOC or less
```

**Acceptance Criteria**:
- main.py reduced to ~300 LOC
- Zero business logic functions remain
- All logic moved to services or routes
- No orphaned imports

#### 4.4 Clean Up Imports
- [ ] Remove unused imports from main.py
- [ ] Add imports for new routers
- [ ] Add imports for new services
- [ ] Organize imports (stdlib, third-party, local)
- [ ] Run `ruff` to validate imports

**Validation**:
```bash
ruff check src/main.py
```

**Acceptance Criteria**:
- No unused imports
- All imports organized
- Linter passes
- No import errors

#### 4.5 Update Documentation
- [ ] Update docstrings in main.py
- [ ] Add module-level documentation
- [ ] Document router registration pattern
- [ ] Document service registration pattern
- [ ] Update inline comments

**Acceptance Criteria**:
- Clear module docstring
- All public functions documented
- Clear comments for complex sections

---

### 5.0 Testing & Validation (2-3 hours)

#### 5.1 Run Unit Tests
- [ ] Run all service unit tests
  ```bash
  python3 -m pytest tests/services/ -v --cov=src/services/application
  ```
- [ ] Verify ≥90% coverage for services
- [ ] Fix any failing tests
- [ ] Add missing test cases

**Acceptance Criteria**:
- All unit tests passing
- ≥90% service coverage
- No skipped tests

#### 5.2 Run Integration Tests
- [ ] Run all route integration tests
  ```bash
  python3 -m pytest tests/api/ -v
  ```
- [ ] Verify all routes accessible
- [ ] Verify response formats correct
- [ ] Fix any failing tests

**Acceptance Criteria**:
- All integration tests passing
- All routes accessible
- Response formats validated

#### 5.3 Run E2E Tests
- [ ] Start the FastAPI server
  ```bash
  python3 src/main.py
  ```
- [ ] Test all endpoints manually or with E2E suite
- [ ] Verify confidence trace endpoint works
- [ ] Verify transparency dossier endpoint works
- [ ] Verify all stub endpoints return stub responses
- [ ] Test error handling

**Validation**:
```bash
# Health check
curl http://localhost:8000/api/v53/health

# Confidence trace
curl http://localhost:8000/api/confidence-trace?trace_id=test

# Transparency dossier
curl http://localhost:8000/api/transparency-dossier
```

**Acceptance Criteria**:
- All endpoints accessible
- Same response formats as before
- Error handling works
- No 500 errors

#### 5.4 Performance Validation
- [ ] Run performance benchmarks on key endpoints
- [ ] Compare to baseline (before refactoring)
- [ ] Ensure no performance regression (within ±5%)
- [ ] Check memory usage
- [ ] Check startup time

**Validation**:
```bash
# Use Apache Bench or similar
ab -n 1000 -c 10 http://localhost:8000/api/v53/health
```

**Acceptance Criteria**:
- Response times within ±5% of baseline
- No memory leaks
- Startup time acceptable

#### 5.5 Regression Testing
- [ ] Run full test suite
  ```bash
  make test
  ```
- [ ] Verify all existing tests still pass
- [ ] Check for any unexpected failures
- [ ] Validate backward compatibility

**Acceptance Criteria**:
- All existing tests passing
- No new test failures
- Backward compatibility maintained

---

### 6.0 Documentation & Cleanup (1 hour)

#### 6.1 Update Architecture Documentation
- [ ] Update system architecture diagram
- [ ] Document new service layer
- [ ] Document router pattern
- [ ] Add migration notes

**Location**: `docs/ARCHITECTURE.md` or similar

**Acceptance Criteria**:
- Architecture docs updated
- New patterns documented
- Clear for new developers

#### 6.2 Update Developer Guide
- [ ] Add "How to add new routes" section
- [ ] Add "How to create services" section
- [ ] Document dependency injection pattern
- [ ] Add examples

**Location**: `docs/DEVELOPER_GUIDE.md` or README

**Acceptance Criteria**:
- Clear guides for common tasks
- Code examples included
- Best practices documented

#### 6.3 Create Migration Summary
- [ ] Document what changed
- [ ] Document where logic moved
- [ ] Create before/after comparison
- [ ] List all new files created

**Location**: `docs/OPERATION_LEAN_MAIN_PY_SUMMARY.md`

**Acceptance Criteria**:
- Complete change log
- File mapping documented
- Summary statistics included

#### 6.4 Code Quality Checks
- [ ] Run linting on all new files
  ```bash
  ruff check src/services/application/
  ruff check src/api/routes/
  ```
- [ ] Run type checking
  ```bash
  mypy src/services/application/
  mypy src/api/routes/
  ```
- [ ] Format all code
  ```bash
  black src/services/application/
  black src/api/routes/
  black src/main.py
  ```
- [ ] Fix any issues

**Acceptance Criteria**:
- All files pass linting
- All files pass type checking
- Consistent formatting

---

## Completion Checklist

Before marking refactoring complete:

### Code Quality
- [ ] main.py is ≤300 LOC
- [ ] All services created with ≥90% test coverage
- [ ] All routes created with integration tests
- [ ] All tests passing
- [ ] Linting passes
- [ ] Type checking passes

### Functionality
- [ ] All endpoints accessible
- [ ] Response formats unchanged
- [ ] Error handling preserved
- [ ] No performance regression
- [ ] Backward compatibility 100%

### Documentation
- [ ] Architecture docs updated
- [ ] Developer guide updated
- [ ] Migration summary created
- [ ] Code comments added

### Deployment Readiness
- [ ] No breaking changes
- [ ] Rollback plan documented
- [ ] Monitoring in place
- [ ] Ready for production

---

## Post-Refactoring Tasks

After successful deployment:

### 7.1 Monitor Production (1 week)
- [ ] Monitor error rates
- [ ] Monitor response times
- [ ] Monitor memory usage
- [ ] Check logs for issues

### 7.2 Gather Feedback
- [ ] Developer feedback on new structure
- [ ] Identify pain points
- [ ] Document learnings

### 7.3 Remove Legacy Code
- [ ] Archive old main_legacy.py (if created)
- [ ] Remove feature flags (if used)
- [ ] Clean up temporary compatibility code

### 7.4 Update Roadmap
- [ ] Mark main.py as COMPLETE in LEAN_ROADMAP.md
- [ ] Update progress metrics
- [ ] Plan next refactoring target

---

## Task Status Tracking

Use this checklist to track overall progress:

- [ ] 1.0 Setup & Analysis (1-2 hours)
- [ ] 2.0 Service Layer Extraction (4-5 hours)
- [ ] 3.0 Route Module Extraction (4-5 hours)
- [ ] 4.0 Main.py Refactoring (2-3 hours)
- [ ] 5.0 Testing & Validation (2-3 hours)
- [ ] 6.0 Documentation & Cleanup (1 hour)

**Total Progress**: 0/6 task groups complete

---

## Emergency Rollback Procedure

If critical issues arise:

1. **Stop deployment immediately**
2. **Revert to main_legacy.py** (if backup created)
3. **Restart services**
4. **Investigate issues**
5. **Fix and redeploy**

---

**Status**: ✅ **READY FOR EXECUTION**

**Next Step**: Begin with Task 1.0 - Setup & Analysis

---

*Created: 2025-10-19*
*Campaign: Operation Lean - Target #2*
*Parent PRD: 2-prd-main-py-refactoring.md*
*Methodology: ai-dev-tasks sequential execution*
