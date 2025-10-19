# Tasks: Method Actor Devils Advocate Service Layer Extraction

**PRD**: `3-prd-method-actor-devils-advocate-refactoring.md`
**Priority**: HIGH (Score: 4.28)
**Estimated Duration**: 8-12 hours
**Current State**: 1160 LOC → Target: ~300 LOC (74% reduction)

---

## Quick Reference

**Main File**: `src/core/method_actor_devils_advocate.py` (1160 LOC, 10 classes, 14 methods)
**Target Architecture**: 6 service files + 1 orchestrator file
**Risk Level**: MEDIUM (no high fan-in, but complex logic)
**Backward Compatibility**: CRITICAL - all public APIs must preserve

---

## Task Checklist

### 1.0 Setup & Planning ✅
- [ ] 1.1 Create `src/core/services/` directory (if not exists)
- [ ] 1.2 Create `tests/core/services/` directory (if not exists)
- [ ] 1.3 Analyze current code structure and create method categorization
- [ ] 1.4 Identify all public APIs that must be preserved
- [ ] 1.5 Create baseline test: verify existing system works
- [ ] 1.6 Backup original file: `cp method_actor_devils_advocate.py method_actor_devils_advocate_backup.py`

### 2.0 Create Service Interfaces (Protocols) ✅
- [ ] 2.1 Create `src/core/services/persona_engine.py`
- [ ] 2.2 Define `PersonaEngine` abstract base class with methods:
  - [ ] 2.2.1 `generate_dialogue(algorithmic_result, recommendation, context) -> MethodActorDialogue`
  - [ ] 2.2.2 `get_persona_config() -> MethodActorPersona`
  - [ ] 2.2.3 `get_signature_methods() -> List[str]`
- [ ] 2.3 Add abstract properties:
  - [ ] 2.3.1 `persona_id: str`
  - [ ] 2.3.2 `persona_type: PersonaType`
- [ ] 2.4 Add comprehensive docstrings with examples
- [ ] 2.5 Verify imports work: `python3 -c "from src.core.services.persona_engine import PersonaEngine"`

### 3.0 Extract ConfigurationLoader ✅
- [ ] 3.1 Create `src/core/services/configuration_loader.py`
- [ ] 3.2 Extract `_load_yaml_config()` method (lines 164-189)
- [ ] 3.3 Extract `_load_thin_variables()` method (lines 191-210)
- [ ] 3.4 Extract `_load_personas_from_yaml()` method (lines 287-321)
- [ ] 3.5 Create `ConfigurationLoader` class with methods:
  - [ ] 3.5.1 `load_yaml_config(yaml_path: str) -> Dict[str, Any]`
  - [ ] 3.5.2 `load_thin_variables(yaml_config: Optional[Dict]) -> Dict[str, Any]`
  - [ ] 3.5.3 `load_personas_from_yaml(yaml_config: Dict) -> Dict[str, MethodActorPersona]`
- [ ] 3.6 Keep `ConfigurationError` exception in original file (shared)
- [ ] 3.7 Add comprehensive docstrings
- [ ] 3.8 Write unit tests: `tests/core/services/test_configuration_loader.py`
  - [ ] 3.8.1 Test YAML loading with valid config
  - [ ] 3.8.2 Test YAML loading with missing file
  - [ ] 3.8.3 Test YAML loading with invalid YAML
  - [ ] 3.8.4 Test thin variables loading from YAML
  - [ ] 3.8.5 Test thin variables fallback to defaults
  - [ ] 3.8.6 Test persona loading from YAML
- [ ] 3.9 Verify test coverage ≥85%: `pytest --cov=src.core.services.configuration_loader`

### 4.0 Extract ForwardMotionConverter ✅
- [ ] 4.1 Create `src/core/services/forward_motion_converter.py`
- [ ] 4.2 Move `ForwardMotionConverter` class (lines 976-1053)
- [ ] 4.3 Keep existing methods:
  - [ ] 4.3.1 `convert_munger_challenges_to_actions(challenges, recommendation)`
  - [ ] 4.3.2 `convert_ackoff_challenges_to_actions(challenges, recommendation)`
- [ ] 4.4 Add generic method:
  - [ ] 4.4.1 `convert_challenges_to_actions(challenges, recommendation, persona_type)`
- [ ] 4.5 Import dependencies: `ForwardMotionAction`, `ForwardMotionType`, `DevilsAdvocateChallenge`
- [ ] 4.6 Add comprehensive docstrings with examples
- [ ] 4.7 Write unit tests: `tests/core/services/test_forward_motion_converter.py`
  - [ ] 4.7.1 Test Munger challenge conversion
  - [ ] 4.7.2 Test Ackoff challenge conversion
  - [ ] 4.7.3 Test empty challenges handling
  - [ ] 4.7.4 Test experiment action generation
  - [ ] 4.7.5 Test guardrail action generation
  - [ ] 4.7.6 Test reversible step generation
- [ ] 4.8 Verify test coverage ≥85%

### 5.0 Extract ToneSafeguards ✅
- [ ] 5.1 Create `src/core/services/tone_safeguards.py`
- [ ] 5.2 Move `ToneSafeguards` class (lines 1056-1099)
- [ ] 5.3 Keep existing method:
  - [ ] 5.3.1 `assess_dialogue_safety(dialogue_text, persona)`
- [ ] 5.4 Add enhanced methods:
  - [ ] 5.4.1 `validate_enabling_challenger_patterns(dialogue_text) -> bool`
  - [ ] 5.4.2 `check_psychological_safety(dialogue_text) -> float`
  - [ ] 5.4.3 `detect_gotcha_patterns(dialogue_text) -> List[str]`
- [ ] 5.5 Import dependencies: `MethodActorPersona`
- [ ] 5.6 Add comprehensive docstrings
- [ ] 5.7 Write unit tests: `tests/core/services/test_tone_safeguards.py`
  - [ ] 5.7.1 Test dialogue safety assessment
  - [ ] 5.7.2 Test gotcha-ism detection
  - [ ] 5.7.3 Test vulnerability opening detection
  - [ ] 5.7.4 Test forward motion pattern detection
  - [ ] 5.7.5 Test psychological safety scoring
  - [ ] 5.7.6 Test edge cases (empty dialogue, all caps, etc.)
- [ ] 5.8 Verify test coverage ≥85%

### 6.0 Extract MungerPersonaEngine ✅
- [ ] 6.1 Create `src/core/services/munger_persona_engine.py`
- [ ] 6.2 Create `MungerPersonaEngine` class implementing `PersonaEngine`
- [ ] 6.3 Move Munger persona initialization (lines 223-252)
- [ ] 6.4 Move `_generate_munger_dialogue()` method (lines 455-532)
- [ ] 6.5 Move Munger helper methods:
  - [ ] 6.5.1 `_get_historical_business_analogy()` (lines 687-697)
  - [ ] 6.5.2 `_transform_biases_to_munger_stories()` (lines 699-722)
  - [ ] 6.5.3 `_generate_munger_inversion_questions()` (lines 724-743)
- [ ] 6.6 Implement `PersonaEngine` interface:
  - [ ] 6.6.1 `generate_dialogue(algorithmic_result, recommendation, context, thin_variables, forward_motion_converter, tone_safeguards)`
  - [ ] 6.6.2 `get_persona_config() -> MethodActorPersona`
  - [ ] 6.6.3 `get_signature_methods() -> List[str]`
- [ ] 6.7 Add properties:
  - [ ] 6.7.1 `persona_id = "charlie_munger"`
  - [ ] 6.7.2 `persona_type = PersonaType.CHARLIE_MUNGER`
- [ ] 6.8 Import dependencies: `PersonaEngine`, `MethodActorPersona`, `MethodActorDialogue`, `ComprehensiveChallengeResult`, `ForwardMotionConverter`, `ToneSafeguards`
- [ ] 6.9 Add comprehensive docstrings with Munger persona description
- [ ] 6.10 Write unit tests: `tests/core/services/test_munger_persona_engine.py`
  - [ ] 6.10.1 Test dialogue generation with bias challenges
  - [ ] 6.10.2 Test historical analogy generation
  - [ ] 6.10.3 Test bias-to-story transformation
  - [ ] 6.10.4 Test inversion question generation
  - [ ] 6.10.5 Test vulnerability opening inclusion
  - [ ] 6.10.6 Test persona config retrieval
  - [ ] 6.10.7 Test edge cases (empty challenges, missing context)
- [ ] 6.11 Verify test coverage ≥85%

### 7.0 Extract AckoffPersonaEngine ✅
- [ ] 7.1 Create `src/core/services/ackoff_persona_engine.py`
- [ ] 7.2 Create `AckoffPersonaEngine` class implementing `PersonaEngine`
- [ ] 7.3 Move Ackoff persona initialization (lines 254-283)
- [ ] 7.4 Move `_generate_ackoff_dialogue()` method (lines 534-609)
- [ ] 7.5 Move Ackoff helper methods:
  - [ ] 7.5.1 `_transform_assumptions_to_ackoff_questions()` (lines 745-761)
  - [ ] 7.5.2 `_generate_ackoff_idealized_design()` (lines 763-774)
- [ ] 7.6 Implement `PersonaEngine` interface:
  - [ ] 7.6.1 `generate_dialogue(algorithmic_result, recommendation, context, thin_variables, forward_motion_converter, tone_safeguards)`
  - [ ] 7.6.2 `get_persona_config() -> MethodActorPersona`
  - [ ] 7.6.3 `get_signature_methods() -> List[str]`
- [ ] 7.7 Add properties:
  - [ ] 7.7.1 `persona_id = "russell_ackoff"`
  - [ ] 7.7.2 `persona_type = PersonaType.RUSSELL_ACKOFF`
- [ ] 7.8 Import dependencies: same as Munger
- [ ] 7.9 Add comprehensive docstrings with Ackoff persona description
- [ ] 7.10 Write unit tests: `tests/core/services/test_ackoff_persona_engine.py`
  - [ ] 7.10.1 Test dialogue generation with assumption challenges
  - [ ] 7.10.2 Test assumption-to-question transformation
  - [ ] 7.10.3 Test idealized design generation
  - [ ] 7.10.4 Test curiosity-driven opening
  - [ ] 7.10.5 Test persona config retrieval
  - [ ] 7.10.6 Test edge cases (empty challenges, missing context)
- [ ] 7.11 Verify test coverage ≥85%

### 8.0 Refactor MethodActorDevilsAdvocate Orchestrator ✅
- [ ] 8.1 Update imports:
  - [ ] 8.1.1 Import `ConfigurationLoader`
  - [ ] 8.1.2 Import `PersonaEngine`
  - [ ] 8.1.3 Import `MungerPersonaEngine`
  - [ ] 8.1.4 Import `AckoffPersonaEngine`
  - [ ] 8.1.5 Import `ForwardMotionConverter`
  - [ ] 8.1.6 Import `ToneSafeguards`
- [ ] 8.2 Refactor `__init__` method:
  - [ ] 8.2.1 Initialize `ConfigurationLoader`
  - [ ] 8.2.2 Use `ConfigurationLoader` to load YAML config
  - [ ] 8.2.3 Use `ConfigurationLoader` to load thin variables
  - [ ] 8.2.4 Initialize persona engine registry: `Dict[PersonaType, PersonaEngine]`
  - [ ] 8.2.5 Register `MungerPersonaEngine` in registry
  - [ ] 8.2.6 Register `AckoffPersonaEngine` in registry
  - [ ] 8.2.7 Initialize `ForwardMotionConverter` instance
  - [ ] 8.2.8 Initialize `ToneSafeguards` instance
  - [ ] 8.2.9 Remove old persona initialization logic
  - [ ] 8.2.10 Remove old `_load_yaml_config`, `_load_thin_variables`, `_load_personas_from_yaml` methods
- [ ] 8.3 Refactor `_transform_to_method_actor_dialogues`:
  - [ ] 8.3.1 Get persona engine from registry
  - [ ] 8.3.2 Delegate dialogue generation to persona engine
  - [ ] 8.3.3 Pass `forward_motion_converter` and `tone_safeguards` to engines
  - [ ] 8.3.4 Remove old `_generate_munger_dialogue` method
  - [ ] 8.3.5 Remove old `_generate_ackoff_dialogue` method
- [ ] 8.4 Remove all Munger-specific helper methods:
  - [ ] 8.4.1 Remove `_get_historical_business_analogy`
  - [ ] 8.4.2 Remove `_transform_biases_to_munger_stories`
  - [ ] 8.4.3 Remove `_generate_munger_inversion_questions`
- [ ] 8.5 Remove all Ackoff-specific helper methods:
  - [ ] 8.5.1 Remove `_transform_assumptions_to_ackoff_questions`
  - [ ] 8.5.2 Remove `_generate_ackoff_idealized_design`
- [ ] 8.6 Update `_convert_to_forward_motion`:
  - [ ] 8.6.1 Already delegates to `ForwardMotionConverter` - verify working
- [ ] 8.7 Update `_assess_anti_failure_measures`:
  - [ ] 8.7.1 Delegate safety checks to `ToneSafeguards`
- [ ] 8.8 Verify public APIs unchanged:
  - [ ] 8.8.1 `method_actor_comprehensive_challenge()` signature identical
  - [ ] 8.8.2 `run_method_actor_critique()` signature identical
  - [ ] 8.8.3 `get_method_actor_devils_advocate()` factory unchanged
- [ ] 8.9 Update docstrings to reflect new architecture
- [ ] 8.10 Add plugin architecture helper method:
  - [ ] 8.10.1 `register_persona_engine(persona_type: PersonaType, engine: PersonaEngine)`

### 9.0 Update Service Exports ✅
- [ ] 9.1 Update `src/core/services/__init__.py`:
  - [ ] 9.1.1 Export `PersonaEngine`
  - [ ] 9.1.2 Export `MungerPersonaEngine`
  - [ ] 9.1.3 Export `AckoffPersonaEngine`
  - [ ] 9.1.4 Export `ForwardMotionConverter`
  - [ ] 9.1.5 Export `ToneSafeguards`
  - [ ] 9.1.6 Export `ConfigurationLoader`
- [ ] 9.2 Verify imports work:
  - [ ] 9.2.1 `from src.core.services import PersonaEngine`
  - [ ] 9.2.2 `from src.core.services import MungerPersonaEngine`
  - [ ] 9.2.3 `from src.core.services import AckoffPersonaEngine`

### 10.0 Integration Testing ✅
- [ ] 10.1 Create `tests/core/test_method_actor_devils_advocate_integration.py`
- [ ] 10.2 Test end-to-end flow:
  - [ ] 10.2.1 Initialize MethodActorDevilsAdvocate
  - [ ] 10.2.2 Run `method_actor_comprehensive_challenge()`
  - [ ] 10.2.3 Verify all personas activated
  - [ ] 10.2.4 Verify forward motion actions generated
  - [ ] 10.2.5 Verify enabling challenger score calculated
  - [ ] 10.2.6 Verify anti-failure measures assessed
- [ ] 10.3 Test public API compatibility:
  - [ ] 10.3.1 Test `run_method_actor_critique()` with existing format
  - [ ] 10.3.2 Verify response format unchanged
  - [ ] 10.3.3 Verify backward compatibility with existing callers
- [ ] 10.4 Test YAML configuration:
  - [ ] 10.4.1 Test with valid YAML config file
  - [ ] 10.4.2 Test fallback to hardcoded personas
  - [ ] 10.4.3 Test thin variables loading
- [ ] 10.5 Test plugin architecture:
  - [ ] 10.5.1 Create mock persona engine
  - [ ] 10.5.2 Register via `register_persona_engine()`
  - [ ] 10.5.3 Verify activation in dialogue generation
- [ ] 10.6 Run all existing tests: `pytest tests/core/test_method_actor*`
- [ ] 10.7 Verify test coverage: `pytest --cov=src.core.method_actor_devils_advocate --cov-report=html`

### 11.0 Performance Validation ✅
- [ ] 11.1 Benchmark original system:
  - [ ] 11.1.1 Time Tier 1 analysis (target: <3s)
  - [ ] 11.1.2 Time Tier 2 analysis (target: <5s)
  - [ ] 11.1.3 Time Tier 3 analysis (target: <10s)
- [ ] 11.2 Benchmark refactored system:
  - [ ] 11.2.1 Time Tier 1 analysis
  - [ ] 11.2.2 Time Tier 2 analysis
  - [ ] 11.2.3 Time Tier 3 analysis
- [ ] 11.3 Compare results:
  - [ ] 11.3.1 Verify no regression (within ±10%)
  - [ ] 11.3.2 Document any improvements
- [ ] 11.4 Memory profiling:
  - [ ] 11.4.1 Profile original system memory usage
  - [ ] 11.4.2 Profile refactored system memory usage
  - [ ] 11.4.3 Verify no significant increase (within ±15%)

### 12.0 Documentation ✅
- [ ] 12.1 Update `src/core/method_actor_devils_advocate.py` docstring
- [ ] 12.2 Create `docs/PLUGIN_ARCHITECTURE_GUIDE.md`:
  - [ ] 12.2.1 How to add new personas
  - [ ] 12.2.2 Code examples (Ray Dalio example)
  - [ ] 12.2.3 Testing new personas
  - [ ] 12.2.4 Best practices
- [ ] 12.3 Create refactoring summary: `tasks/3-REFACTORING-SUMMARY.md`
  - [ ] 12.3.1 LOC reduction summary
  - [ ] 12.3.2 Files created vs removed
  - [ ] 12.3.3 Architecture before/after diagrams
  - [ ] 12.3.4 Plugin architecture examples
  - [ ] 12.3.5 Testing summary
  - [ ] 12.3.6 Performance comparison
- [ ] 12.4 Update `CLAUDE.md` with new architecture notes
- [ ] 12.5 Create test results document: `tasks/3-TEST-RESULTS.md`

### 13.0 Cleanup & Rollout ✅
- [ ] 13.1 Remove backup file (after confirming all tests pass)
- [ ] 13.2 Verify line count reduction:
  - [ ] 13.2.1 Count lines: `wc -l src/core/method_actor_devils_advocate.py`
  - [ ] 13.2.2 Verify ~300 LOC target achieved
  - [ ] 13.2.3 Count new service files
- [ ] 13.3 Run final validation:
  - [ ] 13.3.1 All unit tests pass: `pytest tests/core/services/`
  - [ ] 13.3.2 All integration tests pass: `pytest tests/core/test_method_actor*`
  - [ ] 13.3.3 Test coverage ≥85%: `pytest --cov=src.core.services --cov=src.core.method_actor_devils_advocate`
- [ ] 13.4 Create git backup (optional):
  - [ ] 13.4.1 `git add .`
  - [ ] 13.4.2 `git commit -m "Operation Lean Target #3: Method Actor DA refactored"`
- [ ] 13.5 Update LEAN_ROADMAP.md:
  - [ ] 13.5.1 Mark Target #3 as COMPLETE
  - [ ] 13.5.2 Add completion date
  - [ ] 13.5.3 Document LOC reduction achieved

---

## Validation Commands

### Quick Smoke Tests
```bash
# Test imports work
python3 -c "from src.core.method_actor_devils_advocate import get_method_actor_devils_advocate"
python3 -c "from src.core.services import PersonaEngine, MungerPersonaEngine, AckoffPersonaEngine"

# Test instantiation
python3 -c "from src.core.method_actor_devils_advocate import get_method_actor_devils_advocate; da = get_method_actor_devils_advocate(); print('✅ Instantiation works')"

# Line count check
wc -l src/core/method_actor_devils_advocate.py
```

### Unit Test Runs
```bash
# Test each service independently
pytest tests/core/services/test_configuration_loader.py -v
pytest tests/core/services/test_forward_motion_converter.py -v
pytest tests/core/services/test_tone_safeguards.py -v
pytest tests/core/services/test_munger_persona_engine.py -v
pytest tests/core/services/test_ackoff_persona_engine.py -v

# Test all services
pytest tests/core/services/ -v

# Test integration
pytest tests/core/test_method_actor_devils_advocate_integration.py -v

# Coverage report
pytest --cov=src.core.services --cov=src.core.method_actor_devils_advocate --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Integration Validation
```bash
# Run demo (if exists)
python3 src/core/method_actor_devils_advocate.py

# Test with real scenario
python3 -c "
import asyncio
from src.core.method_actor_devils_advocate import get_method_actor_devils_advocate

async def test():
    da = get_method_actor_devils_advocate()
    result = await da.method_actor_comprehensive_challenge(
        'Strategic acquisition recommendation',
        {'industry': 'tech', 'company': 'TestCo', 's2_tier': 2},
        tier_level=2
    )
    print(f'✅ Result: {len(result.method_actor_dialogues)} dialogues generated')

asyncio.run(test())
"
```

---

## Emergency Rollback

If critical issues arise:

```bash
# 1. Restore backup
cp src/core/method_actor_devils_advocate_backup.py src/core/method_actor_devils_advocate.py

# 2. Run tests to confirm
pytest tests/core/test_method_actor* -v

# 3. Remove new service files (optional)
rm -rf src/core/services/{persona_engine,munger_persona_engine,ackoff_persona_engine,forward_motion_converter,tone_safeguards,configuration_loader}.py
rm -rf tests/core/services/test_{munger,ackoff,forward_motion,tone,configuration}*.py
```

---

## Progress Tracking

**Current Phase**: Task 1.0 - Setup & Planning

Mark tasks complete as you go:
- [ ] = Not started
- [x] = Complete
- [~] = In progress
- [!] = Blocked

---

**Task File Status**: ✅ READY FOR EXECUTION
**Estimated Completion**: 8-12 hours
**Risk Level**: MEDIUM
**Backward Compatibility**: CRITICAL
**Date**: 2025-10-19
