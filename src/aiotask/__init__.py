import asyncio
import functools
import threading
import time
import weakref
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from aiotask._graph import TaskGraph
    from aiotask._render import RenderConfig, get_render, watch

from ._awaitify import node


class TaskStatus(StrEnum):
    WAITING = "waiting to start"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "canceled"


@dataclass(slots=True)
class TaskInfo:
    id: int
    task: asyncio.Task
    description: str
    parent: int | None
    children: list[int]
    running_children: list[int]
    status: TaskStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exception: BaseException | None = None
    logs: str = ""
    completed: float = 0
    total: float | None = None
    auto_progress: bool = True
    deps: list[int] = field(default_factory=list)
    dependents: list[int] = field(default_factory=list)
    depth: int = 0
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

    @asynccontextmanager
    async def allow_edit(self) -> AsyncGenerator[None]:
        async with self._lock:
            self._edit_allowed = True
            try:
                yield
            finally:
                self._edit_allowed = False

    def children_info(
        self,
        fmt: Callable[[TaskInfo], str] = "- {0.description}: {0.status.value}".format,
        sep: str = "\n",
        all_children: bool = False,
    ) -> str:
        return sep.join(
            [fmt(get_task(child_id)) for child_id in (self.children if all_children else self.running_children)]
        )

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
    task_infos: dict[int, TaskInfo] = field(default_factory=dict)
    task_ids: dict[asyncio.Task, int] = field(default_factory=dict)
    background_tasks: set[asyncio.Task] = field(default_factory=set)
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


async def _set_done(task_id: int) -> None:
    state = _get_state()
    task_info = state.task_infos[task_id]
    async with task_info.allow_edit():
        task_info._finish_mono = time.monotonic()
        task_info.finished_at = datetime.now()
        if task_info.task.cancelled():
            task_info.status = TaskStatus.CANCELLED
        elif exc := task_info.task.exception():
            task_info.status = TaskStatus.FAILED
            task_info.exception = exc
        else:
            task_info.status = TaskStatus.DONE
        if task_info.total is None:
            task_info.total = 1
            task_info.completed = 1


async def _update_parent(task_id: int, parent_id: int, auto_progress: bool) -> None:
    state = _get_state()
    parent_task_info = state.task_infos[parent_id]
    async with parent_task_info.allow_edit():
        if auto_progress:
            parent_task_info.completed = (parent_task_info.completed or 0) + 1
        if task_id in parent_task_info.running_children:
            parent_task_info.running_children.remove(task_id)


def _add_done_callback(task: asyncio.Task, task_id: int, state: _LoopState) -> None:
    def callback(_: asyncio.Task) -> None:
        callback_task = asyncio.create_task(_set_done(task_id=task_id))
        state.background_tasks.add(callback_task)
        callback_task.add_done_callback(state.background_tasks.discard)

    task.add_done_callback(callback)


def _add_update_parent_callback(
    task: asyncio.Task,
    task_id: int,
    parent_id: int,
    auto_progress: bool,
    state: _LoopState,
) -> None:
    def callback(_: asyncio.Task) -> None:
        callback_task = asyncio.create_task(
            _update_parent(task_id=task_id, parent_id=parent_id, auto_progress=auto_progress)
        )
        state.background_tasks.add(callback_task)
        callback_task.add_done_callback(state.background_tasks.discard)

    task.add_done_callback(callback)


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

    task_info = TaskInfo(
        id=task_id,
        description=task_name,
        parent=parent_id,
        children=[],
        started_at=datetime.now() if start else None,
        _start_mono=time.monotonic() if start else None,
        status=TaskStatus.RUNNING if start else TaskStatus.WAITING,
        task=task,
        running_children=[],
        auto_progress=auto_progress,
    )

    async with task_info.allow_edit():
        state.task_infos[task_id] = task_info
        _add_done_callback(task, task_id=task_id, state=state)
        if parent_id is not None:
            parent_task_info = state.task_infos[parent_id]
            async with parent_task_info.allow_edit():
                parent_task_info.children.append(task_id)
                if start:
                    parent_task_info.running_children.append(task_id)
                _add_update_parent_callback(
                    task,
                    task_id=task_id,
                    parent_id=parent_id,
                    auto_progress=parent_task_info.auto_progress,
                    state=state,
                )
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
            parent_info.running_children.append(task_id)


async def _register_dep(from_id: int, to_id: int) -> None:
    """Register a dependency edge: from_id depends on to_id."""
    state = _get_state()
    from_info = state.task_infos.get(from_id)
    to_info = state.task_infos.get(to_id)
    if from_info is None or to_info is None:
        return
    async with from_info.allow_edit():
        if to_id not in from_info.deps:
            from_info.deps.append(to_id)
        new_depth = to_info.depth + 1
        if new_depth > from_info.depth:
            from_info.depth = new_depth
    async with to_info.allow_edit():
        if from_id not in to_info.dependents:
            to_info.dependents.append(from_id)


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


def get_task(task_id: int) -> TaskInfo:
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
        stack.extend(task_info.children)


def track[**P, R](
    func: Callable[P, Coroutine[Any, Any, R]],
    start: bool = True,
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Track a coroutine by recording task info."""

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        await _init_task_info(start=start)
        return await func(*args, **kwargs)

    return wrapper



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


__all__ = [
    "RenderConfig",
    "TaskGraph",
    "TaskInfo",
    "TaskStatus",
    "get_render",
    "get_task",
    "get_task_id",
    "log",
    "make_async",
    "make_async_generator",
    "node",
    "remove_task",
    "track",
    "watch",
]


def __getattr__(name: str) -> object:
    if name == "TaskGraph":
        from aiotask._graph import TaskGraph as _TaskGraph

        return _TaskGraph
    if name in ("get_render", "RenderConfig", "watch"):
        import aiotask._render as _render

        return getattr(_render, name)
    msg = f"module 'aiotask' has no attribute {name!r}"
    raise AttributeError(msg)
