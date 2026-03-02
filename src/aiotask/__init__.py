import asyncio
import functools
import inspect
import threading
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Concatenate, Protocol, cast, runtime_checkable


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
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    _edit_allowed: bool = field(default=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any, /) -> None:
        if name in ("_edit_allowed", "_lock"):
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
        all_children=False,
    ) -> str:
        return sep.join(
            [fmt(get_task_info(child_id)) for child_id in (self.children if all_children else self.running_children)]
        )

    def started(self) -> bool:
        return self.started_at is not None

    def done(self) -> bool:
        return self.finished_at is not None

    def duration(self) -> float:
        """Get task duration in seconds."""
        if self.started_at is None:
            return 0.0
        return ((self.finished_at or datetime.now()) - self.started_at).total_seconds()


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
        task = cast("asyncio.Task", asyncio.current_task())
    except RuntimeError as e:
        msg = "This function can only be called from a coroutine."
        raise RuntimeError(msg) from e
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
        task_info.started_at = datetime.now()
        task_info.status = TaskStatus.RUNNING
        if task_info.parent is not None:
            state.task_infos[task_info.parent].running_children.append(task_id)


async def log(value: str = "", end="\n") -> None:
    """Add log to task info."""
    try:
        task_id = _task_id.get()
    except TimeoutError:
        return
    state = _get_state()
    task_info = state.task_infos[task_id]
    async with task_info.allow_edit():
        task_info.logs += value + end


async def get_task_id(task: asyncio.Task, timeout: float = 1) -> int:
    """Get the task_id associated with a task."""
    state = _get_state()
    async with asyncio.timeout(timeout):
        while task not in state.task_ids:
            await asyncio.sleep(0)
        return state.task_ids[task]


def get_task_info(task_id: int) -> TaskInfo:
    """Get the task info from a task_id."""
    loop = asyncio.get_running_loop()
    return _loop_states[loop].task_infos[task_id]


def track_task[**P, R](
    func: Callable[P, Coroutine[Any, Any, R]],
    start: bool = True,
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Track a coroutine by recording task info."""

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        await _init_task_info(start=start)
        return await func(*args, **kwargs)

    return wrapper


def wait_for[**P, R](
    func: Callable[P, Coroutine[Any, Any, R]],
    *to_await: Awaitable,
    track: bool = False,
    start: bool = False,
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Wait for awaitables (e.g. other tasks) and then run the function.

    You can choose to track setting `track=True` and to start before running the wrapped
    function setting `start=True`. Notice that if you are chaining multiple `wait_for` or
    `inject` you should start only on the first wrap and track only on the last wrap.
    For example:
    `task = asyncio.create_task(
        wait_for(inject(my_func, dep, start=True), *awaitables, track=True)(*other_args)`
    """

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if track:
            await _init_task_info(start=False)
        if to_await:
            try:
                await asyncio.gather(*to_await)
            except Exception as e:
                msg = "Failed while waiting to start."
                raise RuntimeError(msg) from e
        if start:
            await _start_task()
        return await func(*args, **kwargs)

    return wrapper


def inject[**P, T, R](
    func: Callable[Concatenate[T, P], Coroutine[Any, Any, R]],
    dep: Awaitable[T] | T,
    track: bool = False,
    start: bool = False,
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Inject awaitables (e.g. other tasks) or simple variables and then run the function.

    You can choose to track setting `track=True` and to start before running the wrapped
    function setting `start=True`. Notice that if you are chaining multiple `wait_for` or
    `inject` you should start only on the first wrap and track only on the last wrap.
    For example:
    `task = asyncio.create_task(
        wait_for(inject(my_func, dep, start=True), *awaitables, track=True)(*other_args)`
    """

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if track:
            await _init_task_info(start=False)
        try:
            var = cast("T", await dep) if inspect.isawaitable(dep) else dep
        except Exception as e:
            first_param = next(iter(inspect.signature(func).parameters.values()))
            msg = f"Failed while waiting for injected variable '{first_param}'."
            raise RuntimeError(msg) from e
        if start:
            await _start_task()
        return await func(var, *args, **kwargs)

    return wrapper  # ty:ignore[invalid-return-type]


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
    "TaskInfo",
    "get_task_id",
    "get_task_info",
    "inject",
    "log",
    "make_async",
    "make_async_generator",
    "wait_for",
]
