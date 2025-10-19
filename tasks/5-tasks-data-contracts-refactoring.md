# Tasks: Data Contracts Package Refactoring

**PRD**: `5-prd-data-contracts-refactoring.md`
**Priority**: HIGH (Score: 4.02)
**Estimated Duration**: 2.5-3 hours
**Current State**: 1523 LOC → Target: 8-10 modules (~100-200 LOC each)

---

## Quick Reference

**Main File**: `src/engine/models/data_contracts.py` (1523 LOC, 47 classes/functions)
**Target Architecture**: Package with models/, validators/, factories/, transformers/
**Risk Level**: HIGH (85 import sites - backward compatibility critical)
**Backward Compatibility**: MANDATORY - all imports must work

---

## Task Checklist

### 1.0 Setup & Analysis ✅
- [ ] 1.1 Backup original file: `cp data_contracts.py data_contracts_backup.py`
- [ ] 1.2 Analyze file structure: count classes, functions, enums
- [ ] 1.3 Verify import sites: `grep -r "from src.engine.models.data_contracts import" src/ | wc -l`
- [ ] 1.4 Create baseline test: verify existing imports work
- [ ] 1.5 Create package directory structure

### 2.0 Create Package Structure ✅
- [ ] 2.1 Create `src/engine/models/data_contracts/` directory
- [ ] 2.2 Create `src/engine/models/data_contracts/__init__.py`
- [ ] 2.3 Create `src/engine/models/data_contracts/models/` directory
- [ ] 2.4 Create `src/engine/models/data_contracts/validators/` directory
- [ ] 2.5 Create `src/engine/models/data_contracts/factories/` directory
- [ ] 2.6 Create `src/engine/models/data_contracts/transformers/` directory
- [ ] 2.7 Create all `__init__.py` files for subpackages

### 3.0 Extract Enums ✅
- [ ] 3.1 Create `models/enums.py`
- [ ] 3.2 Copy all enum definitions from original file
  - EngagementPhase, MentalModelCategory, ConfidenceLevel
  - VulnerabilityDetectionLevel, ExplorationDecision
  - ClarificationQuestionType, ClarificationComplexity
  - ContextType, ContextRelevanceLevel, CognitiveCacheLevel
  - StrategicLayer, CognitiveFunction, ExtendedConsultantRole
- [ ] 3.3 Add imports (Enum, str from typing)
- [ ] 3.4 Update root `__init__.py` to re-export all enums
- [ ] 3.5 Test: `python3 -c "from src.engine.models.data_contracts import EngagementPhase"`

### 4.0 Extract Engagement Models ✅
- [ ] 4.1 Create `models/engagement_models.py`
- [ ] 4.2 Add imports (BaseModel, Field, etc.)
- [ ] 4.3 Copy engagement-related models:
  - [ ] EngagementContext
  - [ ] ClarificationQuestion, ClarificationResponse, ClarificationSession
  - [ ] ExplorationContext
  - [ ] WorkflowState
  - [ ] DeliverableArtifact
  - [ ] FailureModeResponse
- [ ] 4.4 Import enums from `.enums` (relative import)
- [ ] 4.5 Update root `__init__.py` to re-export engagement models
- [ ] 4.6 Test imports work

### 5.0 Extract Consultant Models ✅
- [ ] 5.1 Create `models/consultant_models.py`
- [ ] 5.2 Add imports
- [ ] 5.3 Copy consultant-related models:
  - [ ] ConsultantSpecialization
  - [ ] ScoringWeights
  - [ ] ConsultantMatrix
  - [ ] ConsultantBlueprint
  - [ ] (other consultant models)
- [ ] 5.4 Import enums from `.enums`
- [ ] 5.5 Update root `__init__.py` to re-export consultant models
- [ ] 5.6 Test imports work

### 6.0 Extract Analysis Models ✅
- [ ] 6.1 Create `models/analysis_models.py`
- [ ] 6.2 Add imports
- [ ] 6.3 Copy analysis-related models:
  - [ ] MentalModelDefinition
  - [ ] ReasoningStep
  - [ ] ResearchIntelligence
  - [ ] CognitiveState
  - [ ] ContextElement
  - [ ] ContextRelevanceScore
  - [ ] HallucinationCheck
  - [ ] (other analysis models)
- [ ] 6.4 Import enums from `.enums`
- [ ] 6.5 Update root `__init__.py` to re-export analysis models
- [ ] 6.6 Test imports work

### 7.0 Extract Event Models ✅
- [ ] 7.1 Create `models/event_models.py`
- [ ] 7.2 Add imports
- [ ] 7.3 Copy event-related models (if any separate event models exist)
- [ ] 7.4 Update root `__init__.py` to re-export
- [ ] 7.5 Test imports work

### 8.0 Extract Factory Functions ✅
- [ ] 8.1 Create `factories/engagement_factory.py`
- [ ] 8.2 Copy engagement event factories:
  - [ ] create_engagement_initiated_event()
  - [ ] create_exploration_strategy_event()
- [ ] 8.3 Add imports (import models from ..models.*)
- [ ] 8.4 Create `factories/event_factory.py`
- [ ] 8.5 Copy general event factories:
  - [ ] create_model_selection_event()
  - [ ] create_vulnerability_assessment_event()
  - [ ] create_hallucination_detection_event()
- [ ] 8.6 Update root `__init__.py` to re-export all factories
- [ ] 8.7 Test: `python3 -c "from src.engine.models.data_contracts import create_engagement_initiated_event"`

### 9.0 Extract Validators ✅
- [ ] 9.1 Create `validators/contract_validators.py`
- [ ] 9.2 Copy validation functions:
  - [ ] validate_data_contract_compliance()
  - [ ] get_schema_version()
- [ ] 9.3 Add imports
- [ ] 9.4 Update root `__init__.py` to re-export validators
- [ ] 9.5 Test validators import

### 10.0 Extract Transformers ✅
- [ ] 10.1 Create `transformers/legacy_transformer.py` (if legacy transform functions exist)
- [ ] 10.2 Copy transformation functions
- [ ] 10.3 Update root `__init__.py` to re-export
- [ ] 10.4 Test transformers import

### 11.0 Finalize Root __init__.py ✅
- [ ] 11.1 Verify ALL exports are present
- [ ] 11.2 Add `__all__` list for explicit exports
- [ ] 11.3 Add module docstring
- [ ] 11.4 Test comprehensive import:
  ```python
  from src.engine.models.data_contracts import (
      # Enums
      EngagementPhase, ConfidenceLevel,
      # Models
      EngagementContext, ConsultantBlueprint,
      # Factories
      create_engagement_initiated_event,
      # Validators
      validate_data_contract_compliance,
  )
  ```

### 12.0 Validation & Testing ✅
- [ ] 12.1 Run import validation script:
  ```python
  # Test all 47 exports work
  ```
- [ ] 12.2 Run existing test suite: `pytest tests/engine/models/` -v
- [ ] 12.3 Check import sites: `grep -r "from src.engine.models.data_contracts import" src/`
- [ ] 12.4 Verify count matches (should still be 85)
- [ ] 12.5 Spot check 10 random import sites manually
- [ ] 12.6 Performance check: import time not significantly increased

### 13.0 Cleanup & Documentation ✅
- [ ] 13.1 Delete original `data_contracts.py` file
- [ ] 13.2 Keep backup: `data_contracts_backup.py`
- [ ] 13.3 Create `data_contracts/README.md` documenting structure
- [ ] 13.4 Update any relevant documentation
- [ ] 13.5 Create refactoring summary: `tasks/5-REFACTORING-SUMMARY.md`

---

## Validation Commands

### Quick Import Tests
```bash
# Test each category
python3 -c "from src.engine.models.data_contracts import EngagementPhase, ConfidenceLevel"
python3 -c "from src.engine.models.data_contracts import EngagementContext, ClarificationQuestion"
python3 -c "from src.engine.models.data_contracts import ConsultantBlueprint, ScoringWeights"
python3 -c "from src.engine.models.data_contracts import MentalModelDefinition, ReasoningStep"
python3 -c "from src.engine.models.data_contracts import create_engagement_initiated_event"
python3 -c "from src.engine.models.data_contracts import validate_data_contract_compliance"
```

### Comprehensive Test
```python
# test_data_contracts_imports.py
import sys

try:
    from src.engine.models.data_contracts import (
        # Test a few from each category
        EngagementPhase,
        EngagementContext,
        ConsultantBlueprint,
        create_engagement_initiated_event,
        validate_data_contract_compliance,
    )
    print("✅ All imports successful")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
```

### Import Site Validation
```bash
# Count import sites before
grep -r "from src.engine.models.data_contracts import" src/ | wc -l

# Count import sites after (should be same)
grep -r "from src.engine.models.data_contracts import" src/ | wc -l

# Spot check
grep -r "from src.engine.models.data_contracts import" src/ | head -10
```

---

## Emergency Rollback

If critical issues arise:

```bash
# 1. Remove package directory
rm -rf src/engine/models/data_contracts/

# 2. Restore original file
cp src/engine/models/data_contracts_backup.py src/engine/models/data_contracts.py

# 3. Test
python3 -c "from src.engine.models.data_contracts import EngagementContext; print('✅ Rollback successful')"
```

---

## Progress Tracking

**Current Phase**: Setup

Mark tasks complete as you go:
- [ ] = Not started
- [x] = Complete
- [~] = In progress
- [!] = Blocked

---

**Task File Status**: ✅ READY FOR EXECUTION
**Estimated Completion**: 2.5-3 hours
**Risk Level**: HIGH (85 import sites)
**Backward Compatibility**: MANDATORY
**Date**: 2025-10-19
