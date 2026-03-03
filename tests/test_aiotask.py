"""Unit tests for aiotask - asyncio task tracking library."""

import asyncio
from typing import Any

import pytest

from aiotask import (
    TaskStatus,
    awaitify,
    get_task_id,
    get_task_info,
    inject,
    log,
    make_async,
    make_async_generator,
    remove_task,
    track_task,  # not in __all__ but part of the public surface
    wait_for,
)


def _current_task() -> asyncio.Task[Any]:
    """Return the running task, narrowing out the `None` branch."""
    task = asyncio.current_task()
    assert task is not None
    return task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _flush() -> None:
    """Yield control several times so background callback tasks can finish."""
    for _ in range(3):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# TaskInfo - immutability guard
# ---------------------------------------------------------------------------


class TestTaskInfoImmutability:
    async def test_direct_edit_raises(self) -> None:
        """Writing to TaskInfo fields without allow_edit must raise."""

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            with pytest.raises(RuntimeError, match="allow_edit"):
                info.description = "new description"

        await asyncio.create_task(track_task(coro)())

    async def test_allow_edit_permits_write(self) -> None:
        """allow_edit context manager must permit field updates."""

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            async with info.allow_edit():
                info.description = "updated"
            assert info.description == "updated"

        await asyncio.create_task(track_task(coro)())

    async def test_internal_fields_always_writable(self) -> None:
        """_edit_allowed and _lock can be set without the context manager."""

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            # Should not raise
            info._edit_allowed = info._edit_allowed

        await asyncio.create_task(track_task(coro)())


# ---------------------------------------------------------------------------
# TaskInfo - helper methods
# ---------------------------------------------------------------------------


class TestTaskInfoMethods:
    async def test_started_false_before_init(self) -> None:
        """started() is False when started_at is None."""
        from aiotask import TaskInfo, TaskStatus

        # Build a minimal TaskInfo to test the method in isolation
        task = asyncio.create_task(asyncio.sleep(0))
        info = TaskInfo(
            id=999,
            task=task,
            description="tmp",
            parent=None,
            children=[],
            running_children=[],
            status=TaskStatus.WAITING,
        )
        # Override protection so we can leave started_at as None
        assert info.started() is False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_started_true_after_setting_started_at(self) -> None:
        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            # track_task sets start=True by default
            assert info.started() is True

        await asyncio.create_task(track_task(coro)())

    async def test_done_false_while_running(self) -> None:
        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            assert info.done() is False

        await asyncio.create_task(track_task(coro)())

    async def test_done_true_after_completion(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track_task(coro)())
        await task
        await _flush()

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.done() is True

    async def test_duration_zero_before_start(self) -> None:
        from aiotask import TaskInfo, TaskStatus

        task = asyncio.create_task(asyncio.sleep(0))
        info = TaskInfo(
            id=998,
            task=task,
            description="tmp",
            parent=None,
            children=[],
            running_children=[],
            status=TaskStatus.WAITING,
        )
        assert info.duration() == 0.0
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_duration_positive_while_running(self) -> None:
        async def coro() -> None:
            await asyncio.sleep(0.01)
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            assert info.duration() > 0.0

        await asyncio.create_task(track_task(coro)())

    async def test_duration_frozen_after_completion(self) -> None:
        async def coro() -> None:
            await asyncio.sleep(0.01)

        task = asyncio.create_task(track_task(coro)())
        await task
        await _flush()

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        d1 = info.duration()
        await asyncio.sleep(0.01)
        d2 = info.duration()
        assert d1 == d2  # finished_at is fixed, so duration doesn't grow


# ---------------------------------------------------------------------------
# track_task
# ---------------------------------------------------------------------------


class TestTrackTask:
    async def test_returns_coroutine_result(self) -> None:
        async def coro() -> int:
            return 42

        result = await asyncio.create_task(track_task(coro)())
        assert result == 42

    async def test_status_running_while_executing(self) -> None:
        seen_status: list[TaskStatus] = []

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            seen_status.append(info.status)

        await asyncio.create_task(track_task(coro)())
        assert seen_status == [TaskStatus.RUNNING]

    async def test_status_done_after_completion(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track_task(coro)())
        await task
        await _flush()

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.status == TaskStatus.DONE

    async def test_status_failed_on_exception(self) -> None:
        async def failing_coro() -> None:
            raise ValueError("boom")

        task = asyncio.create_task(track_task(failing_coro)())
        with pytest.raises(ValueError, match="boom"):
            await task
        await _flush()

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.status == TaskStatus.FAILED
        assert isinstance(info.exception, ValueError)

    async def test_status_cancelled(self) -> None:
        async def slow_coro() -> None:
            await asyncio.sleep(10)

        task = asyncio.create_task(track_task(slow_coro)())
        await asyncio.sleep(0)  # let it start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _flush()

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.status == TaskStatus.CANCELLED

    async def test_task_info_has_correct_id(self) -> None:
        captured: list[int] = []

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            captured.append(task_id)

        task = asyncio.create_task(track_task(coro)())
        await task

        task_id = await get_task_id(task)
        assert captured == [task_id]

    async def test_double_init_raises(self) -> None:
        """Calling track_task on a task that already has info must raise."""

        async def coro() -> None:
            # _init_task_info is called once by the wrapper; calling it again
            # via a second track_task wrapper should raise.
            from aiotask import _init_task_info

            with pytest.raises(RuntimeError, match="already initialized"):
                await _init_task_info()

        await asyncio.create_task(track_task(coro)())

    async def test_start_false_yields_waiting_status(self) -> None:
        seen: list[TaskStatus] = []

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            seen.append(info.status)

        await asyncio.create_task(track_task(coro, start=False)())
        assert seen == [TaskStatus.WAITING]


# ---------------------------------------------------------------------------
# Parent-child relationships
# ---------------------------------------------------------------------------


class TestParentChild:
    async def test_child_gets_correct_parent(self) -> None:
        parent_id_holder: list[int] = []
        child_parent_holder: list[int | None] = []

        async def child_coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            child_parent_holder.append(info.parent)

        async def parent_coro() -> None:
            task_id = await get_task_id(_current_task())
            parent_id_holder.append(task_id)
            child_task = asyncio.create_task(track_task(child_coro)())
            await child_task

        await asyncio.create_task(track_task(parent_coro)())
        assert child_parent_holder == parent_id_holder

    async def test_parent_children_list_populated(self) -> None:
        child_id_holder: list[int] = []

        async def child_coro() -> None:
            child_id_holder.append(await get_task_id(_current_task()))

        async def parent_coro() -> None:
            child_task = asyncio.create_task(track_task(child_coro)())
            await child_task
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            assert child_id_holder[0] in info.children

        await asyncio.create_task(track_task(parent_coro)())

    async def test_auto_progress_updates_parent_total_and_completed(self) -> None:
        async def child_coro() -> None:
            pass

        async def parent_coro() -> None:
            task_id = await get_task_id(_current_task())

            child_task = asyncio.create_task(track_task(child_coro)())
            await child_task
            await _flush()

            info = get_task_info(task_id)
            assert info.total == 1
            assert info.completed == 1

        await asyncio.create_task(track_task(parent_coro)())

    async def test_children_info_returns_string(self) -> None:
        async def child_coro() -> None:
            await asyncio.sleep(0.01)

        async def parent_coro() -> None:
            child_task = asyncio.create_task(track_task(child_coro)())
            await asyncio.sleep(0)  # let child start so it's in running_children
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            text = info.children_info()
            assert isinstance(text, str)
            await child_task

        await asyncio.create_task(track_task(parent_coro)())


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------


class TestLog:
    async def test_log_appends_to_task_info(self) -> None:
        async def coro() -> None:
            await log("hello")
            await log("world")

        task = asyncio.create_task(track_task(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert "hello" in info.logs
        assert "world" in info.logs

    async def test_log_custom_end(self) -> None:
        async def coro() -> None:
            await log("no-newline", end="")

        task = asyncio.create_task(track_task(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.logs == "no-newline"


# ---------------------------------------------------------------------------
# wait_for
# ---------------------------------------------------------------------------


class TestWaitFor:
    async def test_waits_before_running(self) -> None:
        order: list[str] = []

        async def dependency() -> None:
            await asyncio.sleep(0)
            order.append("dep")

        async def coro() -> None:
            order.append("coro")

        dep_task = asyncio.create_task(dependency())
        wrapped = wait_for(coro, dep_task)
        await asyncio.create_task(wrapped())
        assert order == ["dep", "coro"]

    async def test_wait_for_with_track_and_start(self) -> None:
        seen: list[TaskStatus] = []

        async def dependency() -> None:
            await asyncio.sleep(0)

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            seen.append(info.status)

        dep_task = asyncio.create_task(dependency())
        wrapped = wait_for(coro, dep_task, track=True, start=True)
        await asyncio.create_task(wrapped())
        assert seen == [TaskStatus.RUNNING]

    async def test_wait_for_raises_on_failed_dependency(self) -> None:
        async def bad_dep() -> None:
            raise RuntimeError("dep failed")

        async def coro() -> None:
            pass

        dep_task = asyncio.create_task(bad_dep())
        wrapped = wait_for(coro, dep_task)
        with pytest.raises(RuntimeError, match="Failed while waiting to start"):
            await asyncio.create_task(wrapped())


# ---------------------------------------------------------------------------
# inject
# ---------------------------------------------------------------------------


class TestInject:
    async def test_inject_awaitable_passes_result_as_first_arg(self) -> None:
        received: list[str] = []

        async def coro(value: str) -> None:
            received.append(value)

        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        future.set_result("injected_value")

        wrapped = inject(coro, future)
        await asyncio.create_task(wrapped())  # ty: ignore[missing-argument]
        assert received == ["injected_value"]

    async def test_inject_plain_value_passes_it_directly(self) -> None:
        received: list[int] = []

        async def coro(value: int) -> None:
            received.append(value)

        wrapped = inject(coro, 99)
        await asyncio.create_task(wrapped())  # ty: ignore[missing-argument]
        assert received == [99]

    async def test_inject_with_extra_args(self) -> None:
        received: list[tuple] = []

        async def coro(first: str, second: int, *, kw: str) -> None:
            received.append((first, second, kw))

        wrapped = inject(coro, "hello")
        await asyncio.create_task(wrapped(42, kw="world"))
        assert received == [("hello", 42, "world")]

    async def test_inject_raises_on_failed_awaitable(self) -> None:
        async def bad_dep() -> str:
            raise ValueError("dep error")

        async def coro(value: str) -> None:
            pass

        wrapped = inject(coro, bad_dep())
        with pytest.raises(RuntimeError, match="Failed while waiting for injected variable"):
            await asyncio.create_task(wrapped())  # ty: ignore[missing-argument]

    async def test_inject_with_track_and_start(self) -> None:
        seen: list[TaskStatus] = []

        async def coro(value: int) -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            seen.append(info.status)

        wrapped = inject(coro, 1, track=True, start=True)
        await asyncio.create_task(wrapped())  # ty: ignore[missing-argument]
        assert seen == [TaskStatus.RUNNING]


# ---------------------------------------------------------------------------
# make_async
# ---------------------------------------------------------------------------


class TestMakeAsync:
    async def test_sync_function_runs_and_returns_value(self) -> None:
        def compute(x: int, y: int) -> int:
            return x + y

        async_compute = make_async(compute)
        result = await async_compute(3, 4)
        assert result == 7

    async def test_runs_in_thread(self) -> None:
        import threading

        caller_thread = threading.current_thread()
        thread_ids: list[int] = []

        def get_thread() -> None:
            ident = threading.current_thread().ident
            assert ident is not None
            thread_ids.append(ident)

        await make_async(get_thread)()
        assert thread_ids[0] != caller_thread.ident

    async def test_preserves_function_name(self) -> None:
        def my_func() -> None:
            pass

        assert make_async(my_func).__name__ == "my_func"  # ty: ignore[unresolved-attribute]

    async def test_sync_exception_propagates(self) -> None:
        def bad() -> None:
            raise ValueError("sync error")

        with pytest.raises(ValueError, match="sync error"):
            await make_async(bad)()


# ---------------------------------------------------------------------------
# make_async_generator
# ---------------------------------------------------------------------------


class TestMakeAsyncGenerator:
    async def test_yields_all_items(self) -> None:
        items = iter([1, 2, 3])
        result = [x async for x in make_async_generator(items)]
        assert result == [1, 2, 3]

    async def test_empty_iterator(self) -> None:
        result = [x async for x in make_async_generator(iter([]))]
        assert result == []

    async def test_accepts_any_iterator_with_next(self) -> None:
        class Counter:
            def __init__(self, n: int) -> None:
                self._n = n
                self._i = 0

            def __next__(self) -> int:
                if self._i >= self._n:
                    raise StopIteration
                self._i += 1
                return self._i

        result = [x async for x in make_async_generator(Counter(3))]
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# get_task_id / get_task_info
# ---------------------------------------------------------------------------


class TestGetTaskHelpers:
    async def test_get_task_id_resolves_for_running_task(self) -> None:
        task_id_holder: list[int] = []

        async def coro() -> None:
            task_id_holder.append(await get_task_id(_current_task()))

        task = asyncio.create_task(track_task(coro)())
        await task
        assert len(task_id_holder) == 1
        assert isinstance(task_id_holder[0], int)

    async def test_get_task_id_times_out_for_untracked_task(self) -> None:
        async def coro() -> None:
            await asyncio.sleep(1)

        task = asyncio.create_task(coro())
        with pytest.raises(TimeoutError):
            await get_task_id(task, timeout=0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_get_task_info_returns_correct_info(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track_task(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.id == task_id
        assert info.task is task

    async def test_ids_are_unique(self) -> None:
        ids: list[int] = []

        async def coro() -> None:
            ids.append(await get_task_id(_current_task()))

        tasks = [asyncio.create_task(track_task(coro)()) for _ in range(5)]
        await asyncio.gather(*tasks)
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# remove_task
# ---------------------------------------------------------------------------


class TestRemoveTask:
    async def test_remove_task_raises_for_unknown_id(self) -> None:
        # Run a tracked task first so the loop state is initialised
        async def coro() -> None:
            pass

        task = asyncio.create_task(track_task(coro)())
        await task

        with pytest.raises(ValueError, match="No task with id"):
            remove_task(999_999)

    async def test_remove_task_makes_get_task_info_raise(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track_task(coro)())
        await task
        await _flush()

        task_id = await get_task_id(task)
        remove_task(task_id)

        with pytest.raises(ValueError, match="No task with id"):
            get_task_info(task_id)

    async def test_remove_task_clears_task_id_mapping(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track_task(coro)())
        await task
        await _flush()

        task_id = await get_task_id(task)
        remove_task(task_id)

        # get_task_id polls until task appears in task_ids; after removal it must time out
        with pytest.raises(TimeoutError):
            await get_task_id(task, timeout=0.05)

    async def test_remove_task_removes_descendants(self) -> None:
        child_id_holder: list[int] = []
        grandchild_id_holder: list[int] = []

        async def grandchild_coro() -> None:
            grandchild_id_holder.append(await get_task_id(_current_task()))

        async def child_coro() -> None:
            child_id_holder.append(await get_task_id(_current_task()))
            await asyncio.create_task(track_task(grandchild_coro)())

        async def parent_coro() -> None:
            await asyncio.create_task(track_task(child_coro)())

        task = asyncio.create_task(track_task(parent_coro)())
        await task
        await _flush()

        parent_id = await get_task_id(task)
        child_id = child_id_holder[0]
        grandchild_id = grandchild_id_holder[0]

        remove_task(parent_id)

        for tid in (parent_id, child_id, grandchild_id):
            with pytest.raises(ValueError):
                get_task_info(tid)

    async def test_remove_child_does_not_affect_parent(self) -> None:
        child_id_holder: list[int] = []

        async def child_coro() -> None:
            child_id_holder.append(await get_task_id(_current_task()))

        async def parent_coro() -> None:
            await asyncio.create_task(track_task(child_coro)())

        task = asyncio.create_task(track_task(parent_coro)())
        await task
        await _flush()

        parent_id = await get_task_id(task)
        child_id = child_id_holder[0]

        remove_task(child_id)

        # Parent info must still be accessible
        info = get_task_info(parent_id)
        assert info.id == parent_id


# ---------------------------------------------------------------------------
# awaitify
# ---------------------------------------------------------------------------


class TestAwaitify:
    async def test_awaitify_plain_args(self) -> None:
        """All plain values passed to a sync function."""

        def add(x: int, y: int) -> int:
            return x + y

        result = await awaitify(add)(1, 2)
        assert result == 3

    async def test_awaitify_awaitable_args(self) -> None:
        """All args are Futures/coroutines — should be resolved before calling."""

        def add(x: int, y: int) -> int:
            return x + y

        loop = asyncio.get_event_loop()
        fx: asyncio.Future[int] = loop.create_future()
        fy: asyncio.Future[int] = loop.create_future()
        fx.set_result(10)
        fy.set_result(20)

        result = await awaitify(add)(fx, fy)
        assert result == 30

    async def test_awaitify_mixed_args(self) -> None:
        """Mix of plain and awaitable positional args."""

        def multiply(x: int, y: int) -> int:
            return x * y

        loop = asyncio.get_event_loop()
        fx: asyncio.Future[int] = loop.create_future()
        fx.set_result(5)

        result = await awaitify(multiply)(fx, 4)
        assert result == 20

    async def test_awaitify_async_func(self) -> None:
        """Wrapping an async function — return value is awaited transparently."""

        async def fetch(value: int) -> int:
            await asyncio.sleep(0)
            return value * 2

        result = await awaitify(fetch)(7)
        assert result == 14

    async def test_awaitify_kwargs(self) -> None:
        """Keyword arguments — both plain and awaitable — are resolved."""

        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        loop = asyncio.get_event_loop()
        name_fut: asyncio.Future[str] = loop.create_future()
        name_fut.set_result("world")

        result = await awaitify(greet)(name=name_fut, greeting="Hi")
        assert result == "Hi, world!"
