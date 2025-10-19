from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Sequence, Set

from .flow_contracts import StageId, StageSpec


def topological_sort(plan: Sequence[StageSpec]) -> List[StageSpec]:
    """Kahn's algorithm to order stages by dependencies.

    Raises ValueError on cycles or missing dependencies.
    """
    by_id: Dict[str, StageSpec] = {s.id.value: s for s in plan}
    indeg: Dict[str, int] = {s.id.value: 0 for s in plan}
    graph: Dict[str, Set[str]] = defaultdict(set)

    for s in plan:
        for dep in s.requires:
            dep_id = dep.value
            if dep_id not in by_id:
                raise ValueError(f"Unknown dependency: {dep_id} for stage {s.id.value}")
            graph[dep_id].add(s.id.value)
            indeg[s.id.value] += 1

    q: Deque[str] = deque([sid for sid, d in indeg.items() if d == 0])
    ordered: List[StageSpec] = []

    while q:
        sid = q.popleft()
        ordered.append(by_id[sid])
        for nxt in graph.get(sid, ()):  # type: ignore[arg-type]
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(ordered) != len(plan):
        raise ValueError("Cycle detected or unresolved dependencies in execution plan")

    return ordered
