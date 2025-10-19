# Engine ↔ Core Dependency Inversion Plan

**Last Updated**: 2025-10-20

## Phase Overview

| Phase | Target Violations | Status |
|-------|-------------------|--------|
| Baseline (2025-10-20) | 139 direct imports | ✅ Recorded (see `docs/ENGINE_CORE_DEPENDENCY_BASELINE.md`) |
| Phase 1 | ≤120 | 🔄 In progress (current: 139 direct imports; adapters being introduced) |
| Phase 2 | ≤80 | ⏳ Planned |
| Phase 3 | ≤40 | ⏳ Planned |
| Phase 4 | 0 | ⏳ Planned |

## Subsystem Distribution (imports routed through adapters)

```
adapters: 20
agents: 2
analytics: 1
api: 1
arbitration: 1
core: 35
engines: 24
enterprise: 1
flywheel: 4
integrations: 5
main: 1
metrics: 1
models: 2
monitoring: 4
optimization: 1
persistence: 4
quality: 2
rag: 1
schemas: 2
security: 1
services: 5
tools: 2
transparency: 1
utils: 1
```

## Next Actions

1. Replace adapter re-exports with dedicated interfaces and adapters per subsystem (ContextMetrics/Evidence done; next up: event bus, pipeline orchestrator facades).
2. Update guardrail targets (`ARCH_GUARD_PHASE`) as each phase completes. Current enforcement remains at baseline until first reduction.
3. Document migration guides for teams touching `src/engine` modules; circulate adapter usage examples in engineering sync.
