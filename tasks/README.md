# Operation Lean: Refactoring Tasks & PRDs

**Campaign**: Operation Lean
**Objective**: Complete 4 high-priority refactorings identified by data-driven analysis
**Format**: Following ai-dev-tasks methodology (snarktank/ai-dev-tasks)
**Status**: IN PROGRESS

---

## Overview

This directory contains Product Requirements Documents (PRDs) and detailed task lists for the 4 remaining refactoring targets from the LEAN_ROADMAP.md.

Each refactoring follows the ai-dev-tasks workflow:
1. **PRD Creation**: Comprehensive requirements document
2. **Task Generation**: Detailed, hierarchical task breakdown
3. **Sequential Execution**: Step-by-step implementation with checkpoints
4. **Human Review**: Approval gates at each major milestone

---

## Refactoring Targets

### ✅ 1. unified_context_stream.py (PLANNED)
- **Priority Score**: 5.41 (HIGHEST)
- **Complexity**: God Object with 60+ methods, 184 import dependencies
- **Strategy**: Service Layer Extraction Pattern
- **Files**:
  - `1-prd-unified-context-stream-refactoring.md` ✅
  - `1-tasks-unified-context-stream-refactoring.md` ✅
- **Est. Duration**: 2-3 sprints (80-120 hours)
- **Risk**: HIGH (184 import sites)

### 🔄 2. main.py (IN PROGRESS)
- **Priority Score**: 5.08
- **Complexity**: 822 LOC entry point with business logic
- **Strategy**: Vertical Slice Extraction Pattern
- **Files**:
  - `2-prd-main-py-refactoring.md` (TODO)
  - `2-tasks-main-py-refactoring.md` (TODO)
- **Est. Duration**: 1-2 sprints (40-80 hours)
- **Risk**: MEDIUM (entry point changes)

### ⏳ 3. method_actor_devils_advocate.py (PENDING)
- **Priority Score**: 4.28
- **Complexity**: 431 LOC with 23 methods, tight coupling
- **Strategy**: Strategy Pattern + Plugin Architecture
- **Files**:
  - `3-prd-method-actor-devils-advocate-refactoring.md` (TODO)
  - `3-tasks-method-actor-devils-advocate-refactoring.md` (TODO)
- **Est. Duration**: 1-2 sprints (40-80 hours)
- **Risk**: MEDIUM (YAML config complexity)

### ⏳ 4. data_contracts.py (PENDING)
- **Priority Score**: 4.02
- **Complexity**: 636 LOC mega-model file, 85 import dependencies
- **Strategy**: Domain-Driven Design - Bounded Context Separation
- **Files**:
  - `4-prd-data-contracts-refactoring.md` (TODO)
  - `4-tasks-data-contracts-refactoring.md` (TODO)
- **Est. Duration**: 1-2 sprints (40-80 hours)
- **Risk**: HIGH (85 import sites)

---

## Workflow

### Phase 1: Planning (CURRENT)
- [x] Create PRD for target #1 (unified_context_stream.py)
- [x] Generate tasks for target #1
- [ ] Create PRD for target #2 (main.py)
- [ ] Generate tasks for target #2
- [ ] Create PRD for target #3 (method_actor_devils_advocate.py)
- [ ] Generate tasks for target #3
- [ ] Create PRD for target #4 (data_contracts.py)
- [ ] Generate tasks for target #4
- [ ] Review all PRDs with team
- [ ] Prioritize execution order

### Phase 2: Execution
For each target:
1. Review PRD with stakeholders
2. Approve task list
3. Execute tasks sequentially using ULTRATHINK
4. Human review at each checkpoint
5. Run comprehensive tests
6. Deploy with monitoring
7. Mark as complete in LEAN_ROADMAP.md

### Phase 3: Validation
- Monitor production for 2 weeks per refactoring
- Collect metrics (LOC reduction, CC reduction, test coverage)
- Document lessons learned
- Update architecture documentation

---

## PRD Template Structure

Each PRD follows this format:

1. **Introduction/Overview** - Problem statement and goals
2. **Goals** - Specific, measurable objectives
3. **User Stories** - Who benefits and how
4. **Functional Requirements** - Detailed specifications
5. **Non-Goals** - Explicit out-of-scope items
6. **Design Considerations** - Architecture and patterns
7. **Technical Considerations** - Constraints and dependencies
8. **Success Metrics** - Quantifiable outcomes
9. **Open Questions** - Unresolved decisions

## Task List Template Structure

Each task list follows this format:

```markdown
## Relevant Files
- `path/to/file.py` - Description

## Tasks
- [ ] 1.0 Parent Task
  - [ ] 1.1 Subtask
  - [ ] 1.2 Subtask
- [ ] 2.0 Parent Task
  - [ ] 2.1 Subtask
```

---

## Execution Guidelines

### For AI (Claude Code / ULTRATHINK)

1. **Read PRD First**: Understand complete context before starting
2. **Sequential Execution**: Complete tasks in order (1.1 → 1.2 → 2.0)
3. **Checkpoint Validation**: Run tests after each major section
4. **Ask Questions**: Flag open questions or blockers immediately
5. **Document Changes**: Update docs as you go, not at end

### For Human Reviewers

1. **Approve PRD**: Review and approve before task execution
2. **Monitor Progress**: Check task completion at end of each day
3. **Review Checkpoints**: Approve at validation checkpoints
4. **Test Locally**: Run tests on your machine at milestones
5. **Deploy Decision**: Final approval for production deployment

---

## Metrics & Success Criteria

### Code Quality Targets

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| unified_context_stream.py LOC | 384 | TBD | ≤150 |
| unified_context_stream.py CC | 18 | TBD | ≤8 |
| main.py LOC | 822 | TBD | ≤300 |
| method_actor_devils_advocate.py LOC | 431 | TBD | ≤150 |
| data_contracts.py LOC | 636 | TBD | ≤100 per file |

### Safety Targets

- **Zero Breaking Changes**: All import sites continue working
- **Test Coverage**: Maintain ≥85% coverage
- **Performance**: <1% regression tolerance
- **Deployment**: Zero rollbacks

---

## Risk Management

### High Risk Refactorings

1. **unified_context_stream.py**: 184 import sites
   - Mitigation: Facade pattern + comprehensive import testing

2. **data_contracts.py**: 85 import sites
   - Mitigation: Backward-compatible re-exports in `__init__.py`

### Medium Risk Refactorings

3. **main.py**: Entry point changes
   - Mitigation: Extensive integration tests + staged rollout

4. **method_actor_devils_advocate.py**: YAML config
   - Mitigation: Config validation + fallback handling

---

## Dependencies

### External
- ai-dev-tasks methodology (snarktank/ai-dev-tasks)
- LEAN_ROADMAP.md (Operation Lean master plan)
- docs/PIPELINE_MIGRATION_ANALYSIS.md (reference implementation)

### Internal
- Operation Lean quantitative analysis
- Importance Matrix scoring
- unified_client.py refactoring (completed reference)

---

## Next Steps

1. ✅ Complete PRD #1 (unified_context_stream.py)
2. ✅ Complete tasks #1
3. 🔄 Create PRD #2 (main.py) - IN PROGRESS
4. Create tasks #2
5. Create PRD #3 (method_actor_devils_advocate.py)
6. Create tasks #3
7. Create PRD #4 (data_contracts.py)
8. Create tasks #4
9. Review with team
10. Begin execution with ULTRATHINK

---

**Status**: Phase 1 (Planning) - 25% Complete (1/4 PRDs done)
**Next**: Complete remaining 3 PRDs + task lists
**Timeline**: Planning complete by end of week, execution starts next sprint

---

*Generated for Operation Lean - 2025-10-18*
*Following ai-dev-tasks methodology for systematic AI-assisted refactoring*
