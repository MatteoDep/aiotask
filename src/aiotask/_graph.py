from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiotask import TaskInfo, TaskStatus, _LoopState


class TaskGraph:
    def __init__(self, root_id: int | None = None) -> None:
        self._root_id = root_id

    def _state(self) -> _LoopState:
        from aiotask import _get_state

        return _get_state()

    def _all_ids(self) -> list[int]:
        """Return all task IDs in this graph (BFS through children tree from root)."""
        state = self._state()
        if self._root_id is None:
            return list(state.task_infos.keys())
        result: list[int] = []
        visited: set[int] = set()
        stack = [self._root_id]
        while stack:
            tid = stack.pop()
            if tid in visited or tid not in state.task_infos:
                continue
            visited.add(tid)
            result.append(tid)
            stack.extend(state.task_infos[tid].children)
        return result

    def nodes(self) -> list[TaskInfo]:
        """Return all nodes in topological order (depth ASC, then id ASC)."""
        state = self._state()
        ids = self._all_ids()
        infos = [state.task_infos[tid] for tid in ids if tid in state.task_infos]
        return sorted(infos, key=lambda x: (x.depth, x.id))

    def node(self, task_id: int) -> TaskInfo:
        from aiotask import get_node

        return get_node(task_id)

    def roots(self) -> list[TaskInfo]:
        """Nodes with no deps (no upstream edges)."""
        return [n for n in self.nodes() if not n.deps]

    def leaves(self) -> list[TaskInfo]:
        """Nodes with no dependents (no downstream edges)."""
        return [n for n in self.nodes() if not n.dependents]

    def upstream(self, task_id: int) -> list[TaskInfo]:
        """All transitive upstream deps of task_id."""
        state = self._state()
        if task_id not in state.task_infos:
            return []
        result: list[TaskInfo] = []
        visited: set[int] = set()
        stack = list(state.task_infos[task_id].deps)
        while stack:
            tid = stack.pop()
            if tid in visited:
                continue
            visited.add(tid)
            if tid in state.task_infos:
                info = state.task_infos[tid]
                result.append(info)
                stack.extend(info.deps)
        return result

    def downstream(self, task_id: int) -> list[TaskInfo]:
        """All transitive downstream dependents of task_id."""
        state = self._state()
        if task_id not in state.task_infos:
            return []
        result: list[TaskInfo] = []
        visited: set[int] = set()
        stack = list(state.task_infos[task_id].dependents)
        while stack:
            tid = stack.pop()
            if tid in visited:
                continue
            visited.add(tid)
            if tid in state.task_infos:
                info = state.task_infos[tid]
                result.append(info)
                stack.extend(info.dependents)
        return result

    def summary(self) -> dict[TaskStatus, int]:
        """Count of nodes per status."""
        counts: dict[TaskStatus, int] = {}
        for n in self.nodes():
            counts[n.status] = counts.get(n.status, 0) + 1
        return counts

    def critical_path(self) -> list[TaskInfo]:
        """Longest-duration path (by accumulated elapsed time) from a root to a leaf."""
        nodes = self.nodes()  # topological order
        if not nodes:
            return []

        # Accumulate max duration from any root to each node
        duration_acc: dict[int, float] = {}
        prev: dict[int, int | None] = {}

        for n in nodes:
            node_dur = n.duration()
            if not n.deps:
                duration_acc[n.id] = node_dur
                prev[n.id] = None
            else:
                best = -1.0
                best_prev: int | None = None
                for dep_id in n.deps:
                    if dep_id in duration_acc:
                        candidate = duration_acc[dep_id] + node_dur
                        if candidate > best:
                            best = candidate
                            best_prev = dep_id
                duration_acc[n.id] = best if best >= 0 else node_dur
                prev[n.id] = best_prev

        leaves = self.leaves()
        if not leaves:
            return nodes

        best_leaf = max(leaves, key=lambda n: duration_acc.get(n.id, 0.0))

        # Trace back from best leaf to root
        path: list[TaskInfo] = []
        state = self._state()
        current_id: int | None = best_leaf.id
        while current_id is not None:
            if current_id in state.task_infos:
                path.append(state.task_infos[current_id])
            current_id = prev.get(current_id)

        path.reverse()
        return path

    @classmethod
    def from_task(cls, task: asyncio.Task) -> TaskGraph:
        from aiotask import _get_state

        state = _get_state()
        task_id = state.task_ids.get(task)
        return cls(root_id=task_id)

    @classmethod
    def current(cls) -> TaskGraph:
        """Graph over all tasks in the current event loop."""
        return cls(root_id=None)
