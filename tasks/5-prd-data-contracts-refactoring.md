# PRD: Data Contracts Package Refactoring

**Priority**: HIGH (Score: 4.02)
**Campaign**: Operation Lean - Target #5
**Target File**: `src/engine/models/data_contracts.py`
**Date**: 2025-10-19

---

## 1. Introduction/Overview

The data_contracts.py file has grown from **636 LOC** (original roadmap) to **1523 LOC** - a 139% increase. It currently contains 47 classes and functions handling multiple unrelated concerns. This refactoring will apply **Domain-Driven Design** principles to separate data models, validators, transformers, and factories into focused modules.

**Problem Being Solved**:
- Single mega-file with 1523 LOC containing unrelated data structures
- Mixed responsibilities (models + validation + transformation + factories)
- High fan-in (85 imports) makes changes risky
- No clear domain boundaries
- Difficult to navigate and maintain

**Overall Goal**:
Break data_contracts.py into a well-organized package structure with clear domain boundaries while maintaining 100% backward compatibility via `__init__.py` re-exports for all 85 import sites.

---

## 2. Goals

1. **Reduce File Complexity**: Break 1523 LOC mega-file into 8-10 focused modules (~100-200 LOC each)
2. **Establish Domain Boundaries**: Separate models, validators, transformers, and factories
3. **Improve Discoverability**: Easy to find specific models or functions
4. **Maintain Backward Compatibility**: Zero breaking changes to 85 import sites
5. **Improve Testability**: Test each domain independently
6. **Reduce Blast Radius**: Changes to validators don't affect models

---

## 3. User Stories

**As a developer adding a new data model**,
I want models organized by domain (engagement, consultant, analysis),
So that I can find related models quickly and add new ones in the right place.

**As a developer writing validators**,
I want validation logic separated from data models,
So that I can modify validation rules without touching model definitions.

**As a developer using data contracts**,
I want my existing imports to continue working,
So that I don't have to update 85 import sites across the codebase.

**As a developer maintaining the codebase**,
I want clear separation between models, validators, transformers, and factories,
So that I understand where each concern lives.

**As a QA engineer writing tests**,
I want to test validators independently from models,
So that I can verify validation logic in isolation.

---

## 4. Current State Analysis

### File Statistics
- **Total LOC**: 1523 (grown from 636 in roadmap)
- **Classes**: ~40 (Pydantic models + Enums)
- **Functions**: ~7 (validators, transformers, factories)
- **Fan-In**: 85 imports across codebase
- **Max CC**: 13 (`get` method)

### Content Breakdown

#### Enums (~13 classes, ~130 LOC)
- `EngagementPhase`, `MentalModelCategory`, `ConfidenceLevel`
- `VulnerabilityDetectionLevel`, `ExplorationDecision`
- `ClarificationQuestionType`, `ClarificationComplexity`
- `ContextType`, `ContextRelevanceLevel`, `CognitiveCacheLevel`
- `StrategicLayer`, `CognitiveFunction`, `ExtendedConsultantRole`

#### Engagement Models (~8 classes, ~400 LOC)
- `EngagementContext`, `ClarificationQuestion`, `ClarificationResponse`
- `ClarificationSession`, `ExplorationContext`, `WorkflowState`
- `DeliverableArtifact`, `FailureModeResponse`

#### Consultant Models (~5 classes, ~200 LOC)
- `ConsultantSpecialization`, `ScoringWeights`, `ConsultantMatrix`
- `ConsultantBlueprint`, (related models)

#### Analysis Models (~8 classes, ~300 LOC)
- `MentalModelDefinition`, `ReasoningStep`, `ResearchIntelligence`
- `CognitiveState`, `ContextElement`, `ContextRelevanceScore`
- `HallucinationCheck`, (related models)

#### Event Models (~6 classes, ~200 LOC)
- Context events, evidence events, pipeline events

#### Factory Functions (~7 functions, ~150 LOC)
- `create_engagement_initiated_event()`
- `create_model_selection_event()`
- `create_vulnerability_assessment_event()`
- `create_exploration_strategy_event()`
- `create_hallucination_detection_event()`
- (others)

#### Validation Functions (~3 functions, ~100 LOC)
- `validate_data_contract_compliance()`
- `get_schema_version()`
- (others)

#### Transformation Functions (~40 LOC)
- Legacy conversion, CloudEvents mapping

---

## 5. Functional Requirements

### 5.1 Package Structure

**5.1.1** Create package: `src/engine/models/data_contracts/`
- Convert data_contracts.py to package
- Maintain backward compatibility via `__init__.py`

**5.1.2** Create subpackage: `src/engine/models/data_contracts/models/`
- `engagement_models.py` - Engagement-related Pydantic models
- `consultant_models.py` - Consultant-related Pydantic models
- `analysis_models.py` - Analysis and reasoning models
- `event_models.py` - Event and context event models
- `enums.py` - All enum definitions

**5.1.3** Create subpackage: `src/engine/models/data_contracts/validators/`
- `contract_validators.py` - Data contract compliance validation
- (future: add more validators as needed)

**5.1.4** Create subpackage: `src/engine/models/data_contracts/factories/`
- `engagement_factory.py` - Engagement event creation
- `event_factory.py` - General event creation

**5.1.5** Create subpackage: `src/engine/models/data_contracts/transformers/`
- `legacy_transformer.py` - Backward compatibility transformations
- (future: CloudEvents transformers if needed)

### 5.2 Backward Compatibility

**5.2.1** `__init__.py` re-exports
- Re-export ALL classes and functions from the original file
- Maintain exact same import paths
- Example: `from src.engine.models.data_contracts import EngagementContext` still works

**5.2.2** Import site preservation
- Zero changes required to 85 import sites
- All existing code continues working without modification

---

## 6. Technical Requirements

### TR-1: Package Structure ✅ REQUIRED

```
src/engine/models/data_contracts/
├── __init__.py                    # Re-exports everything for backward compat
├── models/
│   ├── __init__.py
│   ├── enums.py                   # All enum definitions (~130 LOC)
│   ├── engagement_models.py       # Engagement domain (~400 LOC)
│   ├── consultant_models.py       # Consultant domain (~200 LOC)
│   ├── analysis_models.py         # Analysis & reasoning (~300 LOC)
│   └── event_models.py            # Events and context (~200 LOC)
├── validators/
│   ├── __init__.py
│   └── contract_validators.py     # Validation functions (~100 LOC)
├── factories/
│   ├── __init__.py
│   ├── engagement_factory.py      # Engagement events (~80 LOC)
│   └── event_factory.py           # General events (~70 LOC)
└── transformers/
    ├── __init__.py
    └── legacy_transformer.py      # Legacy compat (~40 LOC)
```

### TR-2: Backward Compatibility ✅ REQUIRED

Root `__init__.py` must re-export:
```python
# All enums
from .models.enums import (
    EngagementPhase,
    MentalModelCategory,
    ConfidenceLevel,
    # ... all enums
)

# All models
from .models.engagement_models import (
    EngagementContext,
    ClarificationQuestion,
    # ... all engagement models
)

from .models.consultant_models import (
    ConsultantSpecialization,
    # ... all consultant models
)

# All factories
from .factories.engagement_factory import (
    create_engagement_initiated_event,
    # ... all factories
)

# All validators
from .validators.contract_validators import (
    validate_data_contract_compliance,
    # ... all validators
)
```

### TR-3: Import Validation ✅ REQUIRED

After refactoring, ALL of these must still work:
```python
from src.engine.models.data_contracts import EngagementContext
from src.engine.models.data_contracts import ConsultantBlueprint
from src.engine.models.data_contracts import create_engagement_initiated_event
from src.engine.models.data_contracts import validate_data_contract_compliance
```

### TR-4: Test Coverage ✅ REQUIRED

- Existing tests must pass without modification
- New tests for each submodule (optional but recommended)
- Integration test validating all imports still work

---

## 7. Refactoring Strategy

### Phase 1: Setup (15 min)
1. Backup original file
2. Create package directory structure
3. Create empty `__init__.py` files

### Phase 2: Extract Enums (15 min)
1. Create `models/enums.py`
2. Move all enum definitions
3. Update root `__init__.py` to re-export

### Phase 3: Extract Models (60 min)
1. Create `models/engagement_models.py` - move engagement models
2. Create `models/consultant_models.py` - move consultant models
3. Create `models/analysis_models.py` - move analysis models
4. Create `models/event_models.py` - move event models
5. Update root `__init__.py` to re-export all models

### Phase 4: Extract Factories (20 min)
1. Create `factories/engagement_factory.py`
2. Create `factories/event_factory.py`
3. Update root `__init__.py` to re-export

### Phase 5: Extract Validators (15 min)
1. Create `validators/contract_validators.py`
2. Update root `__init__.py` to re-export

### Phase 6: Extract Transformers (10 min)
1. Create `transformers/legacy_transformer.py` (if needed)
2. Update root `__init__.py` to re-export

### Phase 7: Testing & Validation (30 min)
1. Test all imports work
2. Run existing test suite
3. Validate 85 import sites (grep check)
4. Delete original data_contracts.py

---

## 8. Success Metrics

### Code Quality Metrics
- ✅ Original file (1523 LOC) broken into 8-10 modules (~100-200 LOC each)
- ✅ Clear domain boundaries (models/validators/factories/transformers)
- ✅ Zero breaking changes to imports
- ✅ All existing tests pass

### Functional Metrics
- ✅ 100% backward compatibility (85 import sites work)
- ✅ All Pydantic models preserved
- ✅ All enums preserved
- ✅ All functions preserved

### Developer Experience Metrics
- ✅ Easy to find specific model (clear naming)
- ✅ Easy to add new model (clear location)
- ✅ Easy to test validators (isolated modules)

---

## 9. Non-Goals (Explicit Exclusions)

1. ❌ **Changing model definitions**: Keep all Pydantic models unchanged
2. ❌ **Modifying validation logic**: Move, don't modify
3. ❌ **Adding new features**: Pure refactoring only
4. ❌ **Updating import sites**: Use re-exports for compatibility
5. ❌ **Performance optimization**: Not the goal of this refactoring
6. ❌ **Changing APIs**: All functions keep same signatures
7. ❌ **Removing deprecated code**: Keep everything
8. ❌ **Adding type hints**: Keep existing type annotations
9. ❌ **Reformatting code**: Minimal changes, preserve structure
10. ❌ **Renaming models**: Keep exact same class names

---

## 10. Risks & Mitigation

### Risk 1: Breaking 85 Import Sites ⚠️ CRITICAL
**Impact**: Entire codebase may fail to import
**Mitigation**:
- Use comprehensive `__init__.py` re-exports
- Test ALL imports before deleting original file
- Keep backup for emergency rollback
- Grep validation: `grep -r "from src.engine.models.data_contracts import" src/`

### Risk 2: Circular Import Dependencies ⚠️ MEDIUM
**Impact**: Models may have circular dependencies
**Mitigation**:
- Use TYPE_CHECKING for type hints
- Import at runtime in factory functions if needed
- Keep models in separate files to avoid cross-dependencies

### Risk 3: Missing Re-exports ⚠️ MEDIUM
**Impact**: Some imports may break
**Mitigation**:
- Automated script to generate `__init__.py` from original file
- Comprehensive test of all exported symbols
- Use `__all__` to explicitly list exports

### Risk 4: Test Failures ⚠️ LOW
**Impact**: Existing tests may break
**Mitigation**:
- Run tests before and after refactoring
- Only move code, don't modify
- Fix import paths in tests if needed

---

## 11. Timeline & Effort Estimate

**Total Estimated Time**: 2.5-3 hours

### Breakdown
- Phase 1: Setup (15 min)
- Phase 2: Extract Enums (15 min)
- Phase 3: Extract Models (60 min)
- Phase 4: Extract Factories (20 min)
- Phase 5: Extract Validators (15 min)
- Phase 6: Extract Transformers (10 min)
- Phase 7: Testing & Validation (30 min)
- Documentation (15 min)

---

## 12. Acceptance Criteria

### Must Have ✅
1. ✅ Original data_contracts.py converted to package
2. ✅ 8-10 focused module files created
3. ✅ All 47 classes and functions preserved
4. ✅ Root `__init__.py` re-exports everything
5. ✅ All existing tests pass without modification
6. ✅ All 85 import sites work without changes
7. ✅ Zero breaking changes

### Should Have 🎯
1. 🎯 Clear README in package documenting structure
2. 🎯 Each module 100-200 LOC (focused)
3. 🎯 Comprehensive docstrings preserved
4. 🎯 Validation that all imports work (automated test)

### Nice to Have ⭐
1. ⭐ New tests for each submodule
2. ⭐ Improved organization within modules
3. ⭐ Migration guide for future developers

---

## 13. Dependencies

- All files importing from `src.engine.models.data_contracts` (85 import sites)
- Existing test suite
- Pydantic library (models use BaseModel)

---

## 14. Related Documents

- LEAN_ROADMAP.md - Operation Lean Phase 1 roadmap
- tasks/5-tasks-data-contracts-refactoring.md - Detailed task breakdown (to be created)

---

**PRD Status**: ✅ READY FOR IMPLEMENTATION
**Campaign**: Operation Lean - Target #5
**Priority**: HIGH (4.02 score)
**Date**: 2025-10-19
