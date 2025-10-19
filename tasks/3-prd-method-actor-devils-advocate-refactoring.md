# PRD: Method Actor Devils Advocate Service Layer Extraction

**Priority**: HIGH (Score: 4.28)
**Campaign**: Operation Lean - Target #3
**Target File**: `src/core/method_actor_devils_advocate.py`
**Date**: 2025-10-19

---

## 1. Introduction/Overview

The Method Actor Devils Advocate system has grown from **431 LOC** (original roadmap) to **1160 LOC** - a 169% increase. It currently contains 10 classes and 14 methods handling multiple unrelated concerns. This refactoring will extract specialized services using the Strategy Pattern and Plugin Architecture.

**Problem Being Solved**:
- Single file with 1160 LOC containing multiple responsibilities
- Tight coupling between persona engines (Munger, Ackoff) and orchestrator
- No clear plugin architecture for adding new personas
- Configuration logic mixed with business logic
- Forward motion conversion and tone safeguards embedded in main class

**Overall Goal**:
Extract persona engines, forward motion converter, and safety assessors into independent, pluggable services while maintaining backward compatibility.

---

## 2. Goals

1. **Reduce Core File Complexity**: Reduce `method_actor_devils_advocate.py` from 1160 LOC to ~300 LOC (74% reduction)
2. **Enable Plugin Architecture**: Create extensible persona engine system for easy addition of new personas
3. **Improve Testability**: Create 5 independent test suites instead of 1 massive suite
4. **Separate Concerns**: Extract configuration, persona logic, forward motion, and safety into dedicated services
5. **Maintain Backward Compatibility**: Zero breaking changes to existing API (`run_method_actor_critique`, `method_actor_comprehensive_challenge`)
6. **Enable Future Personas**: Make it trivial to add Ray Dalio, Jeff Bezos, or other thought leader personas

---

## 3. User Stories

**As a cognitive platform developer**,
I want to add a new persona (e.g., Ray Dalio) without modifying the core orchestrator,
So that I can extend the system without risking regressions.

**As a developer implementing persona engines**,
I want Munger and Ackoff logic in separate classes,
So that I can modify one persona without affecting the other.

**As a QA engineer writing tests**,
I want to test ForwardMotionConverter independently,
So that I can verify action generation without initializing the entire Method Actor system.

**As a platform maintainer**,
I want YAML configuration loading separated from business logic,
So that I can modify config handling without touching persona implementations.

**As a developer using the Method Actor DA system**,
I want my existing code to continue working without changes,
So that I don't have to update imports or API calls.

---

## 4. Functional Requirements

### 4.1 Service Extraction

**4.1.1** Extract `PersonaEngine` abstract base class
- Define interface for all persona engines
- Methods: `generate_dialogue()`, `get_persona_config()`, `get_signature_methods()`
- Expose abstract properties: `persona_id`, `persona_type`

**4.1.2** Extract `MungerPersonaEngine` from MethodActorDevilsAdvocate
- Move Munger-specific dialogue generation (lines ~455-532)
- Move Munger-specific helper methods (lines ~687-743)
- Move Munger persona initialization (lines ~223-252)
- Implement `PersonaEngine` interface
- Expose methods: `generate_munger_dialogue()`, `get_historical_analogy()`, `generate_inversion_questions()`

**4.1.3** Extract `AckoffPersonaEngine` from MethodActorDevilsAdvocate
- Move Ackoff-specific dialogue generation (lines ~534-609)
- Move Ackoff-specific helper methods (lines ~745-774)
- Move Ackoff persona initialization (lines ~254-283)
- Implement `PersonaEngine` interface
- Expose methods: `generate_ackoff_dialogue()`, `generate_idealized_design()`, `transform_assumptions_to_questions()`

**4.1.4** Extract `ForwardMotionConverter` into standalone service
- Already exists as inner class (lines ~976-1053)
- Extract to `src/core/services/forward_motion_converter.py`
- Keep interface: `convert_munger_challenges_to_actions()`, `convert_ackoff_challenges_to_actions()`
- Add generic: `convert_challenges_to_actions()` for extensibility

**4.1.5** Extract `ToneSafeguards` into standalone service
- Already exists as inner class (lines ~1056-1099)
- Extract to `src/core/services/tone_safeguards.py`
- Keep interface: `assess_dialogue_safety()`
- Add: `validate_enabling_challenger_patterns()`, `check_psychological_safety()`

**4.1.6** Extract `ConfigurationLoader` service
- Move YAML loading logic (lines ~164-189)
- Move thin variables loading (lines ~191-210)
- Move persona loading from YAML (lines ~287-321)
- Expose methods: `load_yaml_config()`, `load_thin_variables()`, `load_personas_from_yaml()`

### 4.2 Core MethodActorDevilsAdvocate Responsibilities

After extraction, the core orchestrator should ONLY handle:
- Persona engine registry and selection
- High-level workflow orchestration
- Integration with EnhancedDevilsAdvocateSystem
- Evidence recording via UnifiedContextStream
- Public API compatibility (`run_method_actor_critique`, `method_actor_comprehensive_challenge`)

---

## 5. Technical Requirements

### TR-1: Service Layer Architecture ✅ REQUIRED

```python
src/core/services/
├── __init__.py                      # Service exports
├── persona_engine.py                # Abstract PersonaEngine base class
├── munger_persona_engine.py         # Charlie Munger implementation
├── ackoff_persona_engine.py         # Russell Ackoff implementation
├── forward_motion_converter.py      # Challenge → Action converter
├── tone_safeguards.py               # Safety and tone assessment
└── configuration_loader.py          # YAML and config loading
```

### TR-2: Plugin Architecture ✅ REQUIRED

- PersonaEngine must be an abstract interface/Protocol
- New personas should be addable by:
  1. Creating new class implementing PersonaEngine
  2. Adding to persona registry in config
  3. Zero changes to core orchestrator
- Example future personas: Ray Dalio, Jeff Bezos, Peter Thiel

### TR-3: Backward Compatibility ✅ REQUIRED

- All existing imports must continue working
- Public API methods unchanged:
  - `method_actor_comprehensive_challenge()`
  - `run_method_actor_critique()`
  - `get_method_actor_devils_advocate()`
- All existing response formats preserved
- All existing UnifiedContextStream events preserved

### TR-4: Test Coverage ✅ REQUIRED

- Minimum 85% coverage for each extracted service
- Test suites:
  - `test_munger_persona_engine.py`
  - `test_ackoff_persona_engine.py`
  - `test_forward_motion_converter.py`
  - `test_tone_safeguards.py`
  - `test_configuration_loader.py`
  - `test_method_actor_devils_advocate_integration.py`

### TR-5: Configuration Compatibility ✅ REQUIRED

- Existing YAML configs must continue working
- Support fallback to hardcoded personas if no YAML
- Thin variables must load from YAML or use defaults
- ConfigurationError handling preserved

---

## 6. Non-Functional Requirements

### NFR-1: Performance
- No degradation in analysis time (maintain <5s for tier 2, <10s for tier 3)
- Lazy initialization of persona engines (only load when needed)

### NFR-2: Maintainability
- Each service file: 100-250 LOC (focused, cohesive)
- Clear separation of concerns
- Self-documenting code with comprehensive docstrings

### NFR-3: Extensibility
- Adding new persona: <100 LOC implementation, zero core changes
- Forward motion types extensible via enum
- Tone safeguard patterns configurable

---

## 7. Success Metrics

### Code Quality Metrics
- ✅ Reduce main file from 1160 LOC → ~300 LOC (74% reduction)
- ✅ Create 6 focused service files (~100-250 LOC each)
- ✅ Reduce max cyclomatic complexity from 9 → <5
- ✅ Maintain 100% backward compatibility

### Functional Metrics
- ✅ All existing tests pass without modification
- ✅ Zero breaking changes to public API
- ✅ All UnifiedContextStream events still generated

### Developer Experience Metrics
- ✅ Add new persona in <2 hours (vs ~1 day currently)
- ✅ Test persona logic in isolation (vs full system test currently)
- ✅ Understand persona implementation in <10 minutes (vs ~1 hour currently)

---

## 8. Design

### 8.1 Before Refactoring

```
method_actor_devils_advocate.py (1160 LOC)
├── ConfigurationError
├── MethodActorPersona dataclass
├── ForwardMotionAction dataclass
├── MethodActorDialogue dataclass
├── MethodActorDAResult dataclass
├── PersonaType enum
├── ForwardMotionType enum
├── MethodActorDevilsAdvocate class (23+ methods)
│   ├── __init__
│   ├── _load_yaml_config
│   ├── _load_thin_variables
│   ├── _initialize_personas
│   ├── _load_personas_from_yaml
│   ├── method_actor_comprehensive_challenge ← PUBLIC API
│   ├── _transform_to_method_actor_dialogues
│   ├── _generate_munger_dialogue ← MUNGER LOGIC
│   ├── _generate_ackoff_dialogue ← ACKOFF LOGIC
│   ├── _convert_to_forward_motion
│   ├── _calculate_enabling_challenger_score
│   ├── _calculate_forward_motion_conversion_rate
│   ├── _assess_anti_failure_measures
│   ├── _get_historical_business_analogy ← MUNGER HELPER
│   ├── _transform_biases_to_munger_stories ← MUNGER HELPER
│   ├── _generate_munger_inversion_questions ← MUNGER HELPER
│   ├── _transform_assumptions_to_ackoff_questions ← ACKOFF HELPER
│   ├── _generate_ackoff_idealized_design ← ACKOFF HELPER
│   ├── _record_method_actor_da_evidence
│   └── run_method_actor_critique ← PUBLIC API
├── ForwardMotionConverter class (2 methods)
├── ToneSafeguards class (1 method)
└── get_method_actor_devils_advocate() factory
```

### 8.2 After Refactoring

```
method_actor_devils_advocate.py (~300 LOC) - ORCHESTRATOR ONLY
├── ConfigurationError
├── [All dataclasses preserved]
├── [All enums preserved]
├── MethodActorDevilsAdvocate class (~200 LOC)
│   ├── __init__ (with persona engine registry)
│   ├── method_actor_comprehensive_challenge ← PUBLIC API
│   ├── _transform_to_method_actor_dialogues (delegates to engines)
│   ├── _convert_to_forward_motion (delegates to converter)
│   ├── _calculate_enabling_challenger_score
│   ├── _calculate_forward_motion_conversion_rate
│   ├── _assess_anti_failure_measures (delegates to safeguards)
│   ├── _record_method_actor_da_evidence
│   └── run_method_actor_critique ← PUBLIC API
└── get_method_actor_devils_advocate() factory

src/core/services/persona_engine.py (~80 LOC)
└── PersonaEngine abstract base class
    ├── generate_dialogue() ← abstract
    ├── get_persona_config() ← abstract
    └── get_signature_methods() ← abstract

src/core/services/munger_persona_engine.py (~250 LOC)
└── MungerPersonaEngine(PersonaEngine)
    ├── generate_dialogue()
    ├── get_persona_config()
    ├── get_historical_analogy()
    ├── transform_biases_to_stories()
    └── generate_inversion_questions()

src/core/services/ackoff_persona_engine.py (~220 LOC)
└── AckoffPersonaEngine(PersonaEngine)
    ├── generate_dialogue()
    ├── get_persona_config()
    ├── generate_idealized_design()
    └── transform_assumptions_to_questions()

src/core/services/forward_motion_converter.py (~120 LOC)
└── ForwardMotionConverter
    ├── convert_munger_challenges_to_actions()
    ├── convert_ackoff_challenges_to_actions()
    └── convert_challenges_to_actions() ← GENERIC

src/core/services/tone_safeguards.py (~80 LOC)
└── ToneSafeguards
    ├── assess_dialogue_safety()
    ├── validate_enabling_challenger_patterns()
    └── check_psychological_safety()

src/core/services/configuration_loader.py (~110 LOC)
└── ConfigurationLoader
    ├── load_yaml_config()
    ├── load_thin_variables()
    └── load_personas_from_yaml()
```

### 8.3 Plugin Architecture Example

Adding Ray Dalio persona:

```python
# src/core/services/dalio_persona_engine.py
from src.core.services.persona_engine import PersonaEngine

class DalioPersonaEngine(PersonaEngine):
    def generate_dialogue(self, algorithmic_result, recommendation, context):
        # Implement Principles-based radical transparency dialogue
        ...

    def get_persona_config(self):
        return MethodActorPersona(
            persona_id="ray_dalio",
            character_archetype="Bridgewater Associates Founder",
            cognitive_style="Radical Transparency, Principled Thinking",
            ...
        )

# Register in orchestrator __init__:
self.persona_registry.register("ray_dalio", DalioPersonaEngine())
```

---

## 9. Non-Goals (Explicit Exclusions)

1. ❌ **Changing algorithmic foundation**: Keep EnhancedDevilsAdvocateSystem integration as-is
2. ❌ **Modifying response formats**: Keep all existing data structures unchanged
3. ❌ **Removing YAML support**: Preserve YAML configuration capability
4. ❌ **Implementing new personas**: Focus on extraction, not addition (yet)
5. ❌ **Changing UnifiedContextStream events**: Keep evidence recording unchanged
6. ❌ **Performance optimization**: Maintain current performance, don't optimize further
7. ❌ **Removing hardcoded fallbacks**: Keep fallback personas if no YAML config
8. ❌ **Async/await refactoring**: Keep existing async patterns unchanged
9. ❌ **LLM integration changes**: Keep existing LLM service usage as-is
10. ❌ **Error handling changes**: Preserve existing exception handling patterns

---

## 10. Risks & Mitigation

### Risk 1: Breaking Existing Integrations ⚠️ HIGH
**Impact**: StatefulPipelineOrchestrator and other consumers may break
**Mitigation**:
- Maintain all public API methods with identical signatures
- Create integration tests before refactoring
- Use facade pattern to preserve external interface

### Risk 2: Configuration Loading Issues ⚠️ MEDIUM
**Impact**: YAML configs may fail to load after extraction
**Mitigation**:
- Extract ConfigurationLoader first, test independently
- Ensure fallback to hardcoded personas works
- Add comprehensive config loading tests

### Risk 3: Persona Logic Fragmentation ⚠️ MEDIUM
**Impact**: Split logic may lose coherence
**Mitigation**:
- Keep persona engines cohesive (all Munger logic together)
- Comprehensive integration tests
- Clear documentation of persona capabilities

### Risk 4: Testing Complexity ⚠️ LOW
**Impact**: More files = more test files needed
**Mitigation**:
- Each service gets dedicated test file
- Integration tests verify end-to-end flow
- Maintain >85% coverage requirement

---

## 11. Timeline & Effort Estimate

**Total Estimated Time**: 8-12 hours

### Phase 1: Setup & Interfaces (2 hours)
- Create service directory structure
- Define PersonaEngine abstract base class
- Create interface contracts

### Phase 2: Service Extraction (4-5 hours)
- Extract MungerPersonaEngine (1.5h)
- Extract AckoffPersonaEngine (1.5h)
- Extract ForwardMotionConverter (0.5h)
- Extract ToneSafeguards (0.5h)
- Extract ConfigurationLoader (1h)

### Phase 3: Orchestrator Refactoring (2 hours)
- Update MethodActorDevilsAdvocate to use services
- Implement persona engine registry
- Update initialization logic

### Phase 4: Testing & Validation (2-3 hours)
- Write unit tests for each service
- Create integration tests
- Validate backward compatibility
- Test YAML config loading

### Phase 5: Documentation (1-2 hours)
- Update docstrings
- Create plugin architecture guide
- Document adding new personas
- Create refactoring summary

---

## 12. Acceptance Criteria

### Must Have ✅
1. ✅ method_actor_devils_advocate.py reduced from 1160 LOC → ~300 LOC
2. ✅ 6 service files created (persona_engine, munger, ackoff, forward_motion, tone, config)
3. ✅ All existing tests pass without modification
4. ✅ All public API methods work identically
5. ✅ YAML configuration continues to work
6. ✅ Test coverage ≥85% for each service
7. ✅ Zero breaking changes to external APIs

### Should Have 🎯
1. 🎯 Plugin architecture demonstrated with example
2. 🎯 Documentation for adding new personas
3. 🎯 Integration tests for full flow
4. 🎯 Performance benchmarks (no regression)

### Nice to Have ⭐
1. ⭐ Example third persona (Ray Dalio) implemented
2. ⭐ Enhanced configuration validation
3. ⭐ Performance optimizations

---

## 13. Dependencies

- `src/core/enhanced_devils_advocate_system.py` - Algorithmic foundation (unchanged)
- `src/core/unified_context_stream.py` - Evidence recording (unchanged)
- `cognitive_architecture/NWAY_DEVILS_ADVOCATE_001.yaml` - Config file (unchanged)

---

## 14. Related Documents

- LEAN_ROADMAP.md - Operation Lean Phase 1 roadmap
- tasks/3-tasks-method-actor-devils-advocate-refactoring.md - Detailed task breakdown (to be created)

---

**PRD Status**: ✅ READY FOR IMPLEMENTATION
**Campaign**: Operation Lean - Target #3
**Priority**: HIGH (4.28 score)
**Date**: 2025-10-19
