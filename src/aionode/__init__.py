import asyncio
import functools
import importlib.metadata
import inspect
import threading
import time
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, cast, runtime_checkable

__version__ = importlib.metadata.version("aionode")


@dataclass(slots=True)
class _Resolved[T]:
    awaitable: Awaitable[T]


def resolve[T](awaitable: Awaitable[T]) -> T:
    return _Resolved(awaitable)  # type: ignore[return-value]


def node[**P, R](
    func: Callable[P, Coroutine[Any, Any, R]],
    /,
    wait_for: Sequence[Awaitable[Any]] | None = None,
    track: bool = True,
    auto_progress: bool = True,
) -> Callable[P, Coroutine[Any, Any, R]]:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if track:
            await _init_task_info(start=False, auto_progress=auto_progress)
            _start = _start_task
        else:

            async def _start() -> None:
                pass

        # Identify _Resolved positions
        resolved_arg_idxs = [(i, a) for i, a in enumerate(args) if isinstance(a, _Resolved)]
        resolved_kwarg_keys = [(k, v) for k, v in kwargs.items() if isinstance(v, _Resolved)]
        all_awaitables = (
            [a.awaitable for _, a in resolved_arg_idxs]
            + [v.awaitable for _, v in resolved_kwarg_keys]
            + list(wait_for or [])
        )

        if track:
            state = _get_state()
            our_id = _task_id.get()
            dep_tasks: list[asyncio.Task] = (
                [a.awaitable for _, a in resolved_arg_idxs if isinstance(a.awaitable, asyncio.Task)]
                + [v.awaitable for _, v in resolved_kwarg_keys if isinstance(v.awaitable, asyncio.Task)]
                + [d for d in (wait_for or []) if isinstance(d, asyncio.Task)]
            )
            for dep_task in dep_tasks:
                if dep_task in state.task_ids:
                    await _register_dep(our_id, state.task_ids[dep_task])

        try:
            if all_awaitables:
                results = await asyncio.gather(*all_awaitables)
                n_args = len(resolved_arg_idxs)
                n_kw = len(resolved_kwarg_keys)
                arg_results = list(results[:n_args])
                kwarg_results = list(results[n_args : n_args + n_kw])
                # results[n_args + n_kw:] are wait_for results — discarded
            else:
                arg_results, kwarg_results = [], []
        except Exception as e:
            msg = "Failed while waiting to start."
            raise RuntimeError(msg) from e

        # Rebuild args/kwargs with resolved values
        resolved_args = list(args)
        for (i, _), val in zip(resolved_arg_idxs, arg_results, strict=True):
            resolved_args[i] = val
        resolved_kwargs = dict(kwargs)
        for (k, _), val in zip(resolved_kwarg_keys, kwarg_results, strict=True):
            resolved_kwargs[k] = val

        await _start()

        try:
            result = func(*resolved_args, **resolved_kwargs)
            retval = await result if inspect.isawaitable(result) else result
        except BaseException as exc:
            if track:
                _mark_done(_task_id.get(), exc, _get_state())
            raise
        else:
            if track:
                _mark_done(_task_id.get(), None, _get_state())
        return retval

    return wrapper


class _Unset:
    """Sentinel for unset keyword arguments."""


_UNSET = _Unset()


class TaskStatus(StrEnum):
    """Status of a tracked asyncio task."""

    WAITING = "waiting to start"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class TaskInfo:
    """Metadata and state for a single tracked asyncio task."""

    id: int
    task: asyncio.Task
    name: str
    parent: int | None
    status: TaskStatus
    subtasks: tuple[int, ...] = ()
    running_subtasks: tuple[int, ...] = ()
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exception: BaseException | None = None
    logs: str = ""
    completed: float = 0
    total: float | None = None
    auto_progress: bool = True
    deps: tuple[int, ...] = ()
    dependents: tuple[int, ...] = ()
    tree_depth: int = 0
    dag_depth: int = 0
    _start_mono: float | None = field(default=None, repr=False, compare=False)
    _finish_mono: float | None = field(default=None, repr=False, compare=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    _edit_allowed: bool = field(default=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any, /) -> None:
        if name in ("_edit_allowed", "_lock", "_start_mono", "_finish_mono"):
            object.__setattr__(self, name, value)
            return

        if hasattr(self, "_edit_allowed") and not object.__getattribute__(self, "_edit_allowed"):
            msg = "Edit not allowed. Use the `allow_edit` context manager."
            raise RuntimeError(msg)

        object.__setattr__(self, name, value)

    @contextmanager
    def edit(self) -> Iterator[None]:
        self._edit_allowed = True
        try:
            yield
        finally:
            self._edit_allowed = False

    @asynccontextmanager
    async def allow_edit(self) -> AsyncGenerator[None]:
        async with self._lock:
            self._edit_allowed = True
            try:
                yield
            finally:
                self._edit_allowed = False

    async def update(
        self,
        *,
        completed: float | _Unset = _UNSET,
        total: float | None | _Unset = _UNSET,
    ) -> None:
        """Update user-facing fields atomically."""
        async with self.allow_edit():
            if not isinstance(completed, _Unset):
                self.completed = completed
            if not isinstance(total, _Unset):
                self.total = total

    def subtasks_info(
        self,
        fmt: Callable[["TaskInfo"], str] = "- {0.name}: {0.status.value}".format,
        sep: str = "\n",
        all_subtasks: bool = False,
    ) -> str:
        items: list[str] = []
        for child_id in self.subtasks if all_subtasks else self.running_subtasks:
            try:
                items.append(fmt(get_task_info(child_id)))
            except ValueError:
                continue
        return sep.join(items)

    def started(self) -> bool:
        return self.started_at is not None

    def done(self) -> bool:
        return self.finished_at is not None

    def duration(self) -> float:
        """Get task duration in seconds."""
        if self._start_mono is None:
            return 0.0
        end = self._finish_mono if self._finish_mono is not None else time.monotonic()
        return end - self._start_mono


_task_id: ContextVar[int] = ContextVar("task_id")


@dataclass
class _LoopState:
    task_infos: dict[int, "TaskInfo"] = field(default_factory=dict)
    task_ids: dict[asyncio.Task, int] = field(default_factory=dict)
    _next_id: int = field(default=0)
    _id_lock: threading.Lock = field(default_factory=threading.Lock)

    def allocate_id(self) -> int:
        with self._id_lock:
            task_id = self._next_id
            self._next_id += 1
            return task_id


_loop_states: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopState] = weakref.WeakKeyDictionary()


def _get_state() -> _LoopState:
    loop = asyncio.get_running_loop()
    try:
        return _loop_states[loop]
    except KeyError:
        state = _LoopState()
        _loop_states[loop] = state
        return state


def _mark_done(task_id: int, exc: BaseException | None, state: _LoopState) -> None:
    """Synchronously update TaskInfo when a tracked task finishes.

    Called from within the wrapper coroutine (via try/except), so no event-loop
    concurrency can interleave here — bypassing the async lock is safe.
    """
    task_info = state.task_infos[task_id]
    with task_info.edit():
        task_info._finish_mono = time.monotonic()
        task_info.finished_at = datetime.now()
        if isinstance(exc, asyncio.CancelledError):
            task_info.status = TaskStatus.CANCELLED
        elif exc is not None:
            task_info.status = TaskStatus.FAILED
            task_info.exception = exc
        else:
            task_info.status = TaskStatus.DONE
        if task_info.total is None:
            task_info.total = 1
            task_info.completed = 1

    if task_info.parent is not None:
        parent_info = state.task_infos[task_info.parent]
        with parent_info.edit():
            if task_info.auto_progress:
                parent_info.completed = (parent_info.completed or 0) + 1
            if task_id in parent_info.running_subtasks:
                parent_info.running_subtasks = tuple(
                    tid for tid in parent_info.running_subtasks if tid != task_id
                )


def _get_task() -> asyncio.Task:
    try:
        task = asyncio.current_task()
    except RuntimeError as e:
        msg = "This function can only be called from a coroutine."
        raise RuntimeError(msg) from e
    if task is None:
        msg = "No current task. This function must be called from within an asyncio task, not a callback."
        raise RuntimeError(msg)
    return task


async def _init_task_info(start: bool = True, auto_progress: bool = True) -> None:
    state = _get_state()
    task = _get_task()
    task_name = task.get_name()
    if task in state.task_ids:
        msg = f"Task {task_name} is already initialized"
        raise RuntimeError(msg)

    task_id = state.allocate_id()

    # get parent
    try:
        parent_id = _task_id.get()
    except LookupError:
        parent_id = None

    parent_tree_depth = state.task_infos[parent_id].tree_depth if parent_id is not None else -1

    task_info = TaskInfo(
        id=task_id,
        name=task_name,
        parent=parent_id,
        started_at=datetime.now() if start else None,
        _start_mono=time.monotonic() if start else None,
        status=TaskStatus.RUNNING if start else TaskStatus.WAITING,
        task=task,
        auto_progress=auto_progress,
        tree_depth=parent_tree_depth + 1,
        dag_depth=0,
    )

    async with task_info.allow_edit():
        state.task_infos[task_id] = task_info
        if parent_id is not None:
            parent_task_info = state.task_infos[parent_id]
            async with parent_task_info.allow_edit():
                parent_task_info.subtasks = (*parent_task_info.subtasks, task_id)
                if start:
                    parent_task_info.running_subtasks = (*parent_task_info.running_subtasks, task_id)
                if parent_task_info.auto_progress:
                    total = parent_task_info.total or 0
                    parent_task_info.total = total + 1
        state.task_ids[task] = task_id
        _task_id.set(task_id)


async def _start_task() -> None:
    state = _get_state()
    task = _get_task()
    if task not in state.task_ids:
        msg = f"Cannot start uninitialized task {task.get_name()}"
        raise RuntimeError(msg)
    task_id = _task_id.get()
    task_info = state.task_infos[task_id]
    async with task_info.allow_edit():
        task_info._start_mono = time.monotonic()
        task_info.started_at = datetime.now()
        task_info.status = TaskStatus.RUNNING
    if task_info.parent is not None:
        parent_info = state.task_infos[task_info.parent]
        async with parent_info.allow_edit():
            parent_info.running_subtasks = (*parent_info.running_subtasks, task_id)


def _would_cycle(state: _LoopState, from_id: int, to_id: int) -> bool:
    """Return True if adding edge from_id -> to_id would create a cycle."""
    visited: set[int] = set()
    stack = [from_id]
    while stack:
        tid = stack.pop()
        if tid == to_id:
            continue
        if tid in visited:
            continue
        visited.add(tid)
        info = state.task_infos.get(tid)
        if info is not None:
            for dep_id in info.dependents:
                if dep_id == to_id:
                    return True
                stack.append(dep_id)
    return False


async def _register_dep(from_id: int, to_id: int) -> None:
    """Register a dependency edge: from_id depends on to_id."""
    state = _get_state()
    from_info = state.task_infos.get(from_id)
    to_info = state.task_infos.get(to_id)
    if from_info is None or to_info is None:
        return
    if _would_cycle(state, from_id, to_id):
        from_desc = from_info.name
        to_desc = to_info.name
        msg = f"Circular dependency detected: {from_desc!r} -> {to_desc!r} would create a cycle."
        raise RuntimeError(msg)
    async with from_info.allow_edit():
        if to_id not in from_info.deps:
            from_info.deps = (*from_info.deps, to_id)
        new_dag_depth = to_info.dag_depth + 1
        if new_dag_depth > from_info.dag_depth:
            from_info.dag_depth = new_dag_depth
    async with to_info.allow_edit():
        if from_id not in to_info.dependents:
            to_info.dependents = (*to_info.dependents, from_id)


async def log(value: str = "", end: str = "\n") -> None:
    """Add log to task info."""
    try:
        task_id = _task_id.get()
    except LookupError:
        return
    state = _get_state()
    task_info = state.task_infos[task_id]
    async with task_info.allow_edit():
        task_info.logs += value + end


async def get_task_id(task: asyncio.Task, timeout: float = 1) -> int:
    """Get the task_id associated with an asyncio task."""
    state = _get_state()
    async with asyncio.timeout(timeout):
        while task not in state.task_ids:
            await asyncio.sleep(0)
        return state.task_ids[task]


def current_task_info() -> TaskInfo:
    """Return the TaskInfo for the currently executing tracked task.

    Raises RuntimeError if called outside a node()-wrapped coroutine.
    """
    try:
        task_id = _task_id.get()
    except LookupError:
        msg = "Not inside a tracked task. Call this from within a node()-wrapped coroutine."
        raise RuntimeError(msg) from None
    return get_task_info(task_id)


def get_task_info(task_id: int) -> TaskInfo:
    """Get the task info from a task_id."""
    loop = asyncio.get_running_loop()
    try:
        return _loop_states[loop].task_infos[task_id]
    except KeyError:
        msg = f"No task with id {task_id!r} found in the current event loop."
        raise ValueError(msg) from None



def remove_task(task_id: int) -> None:
    """Remove a task and all its descendants from tracking to free memory."""
    loop = asyncio.get_running_loop()
    state = _loop_states[loop]
    if task_id not in state.task_infos:
        msg = f"No task with id {task_id!r} found in the current event loop."
        raise ValueError(msg)
    stack = [task_id]
    while stack:
        tid = stack.pop()
        task_info = state.task_infos.pop(tid, None)
        if task_info is None:
            continue
        state.task_ids.pop(task_info.task, None)
        stack.extend(task_info.subtasks)


def make_async[**P, T](
    func: Callable[P, T],
) -> Callable[P, Coroutine[Any, Any, T]]:
    """Run function in a separate thread."""

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


@runtime_checkable
class SupportsNext[T](Protocol):
    def __next__(self) -> T: ...


async def make_async_generator[T](gen: SupportsNext[T]) -> AsyncGenerator[T]:
    """Run each `next` call in a separate thread."""
    sentinel = object()

    def step() -> T | object:
        return next(gen, sentinel)

    while True:
        obj = await asyncio.to_thread(step)
        if obj is sentinel:
            break
        yield cast("T", obj)


def walk_tree(root: asyncio.Task | int | None = None) -> Iterator[TaskInfo]:
    """DFS pre-order through call tree (parent -> subtasks).

    If *root* is an ``asyncio.Task``, its tracked id is looked up.
    If *root* is ``None``, the root task (parent=None) is used.
    """
    state = _get_state()
    if root is None:
        root_id = next((tid for tid, info in state.task_infos.items() if info.parent is None), None)
        if root_id is None:
            return
    elif isinstance(root, int):
        root_id = root
    else:
        root_id = state.task_ids.get(root)
        if root_id is None:
            return

    stack = [root_id]
    while stack:
        tid = stack.pop()
        info = state.task_infos.get(tid)
        if info is None:
            continue
        yield info
        # Push children in reverse so leftmost child is visited first
        stack.extend(reversed(info.subtasks))


def walk_dag(root: asyncio.Task | int | None = None) -> Iterator[TaskInfo]:
    """Topological order (Kahn's algorithm) over tasks.

    Uses both tree edges (parent->child) and DAG edges (dep->dependent).
    If *root* is ``None``, includes all tasks in the event loop.
    """
    state = _get_state()

    # Collect the set of task IDs to include
    if root is None:
        ids = list(state.task_infos.keys())
    else:
        if isinstance(root, int):
            start_id = root
        else:
            start_id = state.task_ids.get(root)
            if start_id is None:
                return
        ids = []
        visited: set[int] = set()
        bfs = [start_id]
        while bfs:
            tid = bfs.pop()
            if tid in visited or tid not in state.task_infos:
                continue
            visited.add(tid)
            ids.append(tid)
            bfs.extend(state.task_infos[tid].subtasks)

    infos = {tid: state.task_infos[tid] for tid in ids if tid in state.task_infos}
    if not infos:
        return

    # Kahn's algorithm — edges: parent->child and dep->dependent
    in_degree: dict[int, int] = dict.fromkeys(infos, 0)
    successors: dict[int, list[int]] = {tid: [] for tid in infos}

    for tid, info in infos.items():
        if info.parent is not None and info.parent in infos:
            in_degree[tid] += 1
            successors[info.parent].append(tid)
        for dep_id in info.deps:
            if dep_id in infos and dep_id != info.parent:
                in_degree[tid] += 1
                successors[dep_id].append(tid)

    queue = sorted(tid for tid, d in in_degree.items() if d == 0)
    while queue:
        tid = queue.pop(0)
        yield infos[tid]
        newly_free = sorted(s for s in successors[tid] if in_degree[s] - 1 == 0)
        for s in successors[tid]:
            in_degree[s] -= 1
        queue = sorted(set(queue) | set(newly_free))


__all__ = [
    "TaskInfo",
    "TaskStatus",
    "current_task_info",
    "get_task_id",
    "get_task_info",
    "log",
    "make_async",
    "make_async_generator",
    "node",
    "remove_task",
    "resolve",
    "walk_dag",
    "walk_tree",
]
