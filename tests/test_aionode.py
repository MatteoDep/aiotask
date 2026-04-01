"""Unit tests for aionode - asyncio task tracking library."""

import asyncio
from typing import Any

import pytest

from aionode import (
    TaskInfo,
    TaskStatus,
    current_task_info,
    get_task_id,
    get_task_info,
    log,
    make_async,
    make_async_generator,
    node,
    remove_task,
    resolve,
    walk_dag,
    walk_tree,
)


def _current_task() -> asyncio.Task[Any]:
    """Return the running task, narrowing out the `None` branch."""
    task = asyncio.current_task()
    assert task is not None
    return task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
                info.name = "new name"

        await asyncio.create_task(node(coro)())

    async def test_allow_edit_permits_write(self) -> None:
        """allow_edit context manager must permit field updates."""

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            async with info.allow_edit():
                info.name = "updated"
            assert info.name == "updated"

        await asyncio.create_task(node(coro)())

    async def test_internal_fields_always_writable(self) -> None:
        """_edit_allowed and _lock can be set without the context manager."""

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            # Should not raise
            info._edit_allowed = info._edit_allowed

        await asyncio.create_task(node(coro)())


# ---------------------------------------------------------------------------
# TaskInfo - helper methods
# ---------------------------------------------------------------------------


class TestTaskInfoMethods:
    async def test_started_false_before_init(self) -> None:
        """started() is False when started_at is None."""

        # Build a minimal TaskInfo to test the method in isolation
        task = asyncio.create_task(asyncio.sleep(0))
        info = TaskInfo(
            id=999,
            task=task,
            name="tmp",
            parent=None,
            subtasks=(),
            running_subtasks=(),
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
            # node sets start=True after resolving deps
            assert info.started() is True

        await asyncio.create_task(node(coro)())

    async def test_done_false_while_running(self) -> None:
        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            assert info.done() is False

        await asyncio.create_task(node(coro)())

    async def test_done_true_after_completion(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(node(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.done() is True

    async def test_duration_zero_before_start(self) -> None:

        task = asyncio.create_task(asyncio.sleep(0))
        info = TaskInfo(
            id=998,
            task=task,
            name="tmp",
            parent=None,
            subtasks=(),
            running_subtasks=(),
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

        await asyncio.create_task(node(coro)())

    async def test_duration_frozen_after_completion(self) -> None:
        async def coro() -> None:
            await asyncio.sleep(0.01)

        task = asyncio.create_task(node(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        d1 = info.duration()
        await asyncio.sleep(0.01)
        d2 = info.duration()
        assert d1 == d2  # finished_at is fixed, so duration doesn't grow


# ---------------------------------------------------------------------------
# TaskInfo.update()
# ---------------------------------------------------------------------------


class TestTaskInfoUpdate:
    async def test_update_completed(self) -> None:
        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            await info.update(completed=5)
            assert info.completed == 5

        await asyncio.create_task(node(coro)())

    async def test_update_total(self) -> None:
        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            await info.update(total=10)
            assert info.total == 10

        await asyncio.create_task(node(coro)())

    async def test_update_multiple_fields(self) -> None:
        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            await info.update(completed=3, total=10)
            assert info.completed == 3
            assert info.total == 10

        await asyncio.create_task(node(coro)())

    async def test_update_total_none(self) -> None:
        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            await info.update(total=5)
            await info.update(total=None)
            assert info.total is None

        await asyncio.create_task(node(coro)())

# ---------------------------------------------------------------------------
# node — core tracking behaviour (replaces old TestTrack)
# ---------------------------------------------------------------------------


class TestNodeCore:
    async def test_returns_coroutine_result(self) -> None:
        async def coro() -> int:
            return 42

        result = await asyncio.create_task(node(coro)())
        assert result == 42

    async def test_status_running_while_executing(self) -> None:
        seen_status: list[TaskStatus] = []

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            seen_status.append(info.status)

        await asyncio.create_task(node(coro)())
        assert seen_status == [TaskStatus.RUNNING]

    async def test_status_done_after_completion(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(node(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.status == TaskStatus.DONE

    async def test_status_failed_on_exception(self) -> None:
        async def failing_coro() -> None:
            raise ValueError("boom")

        task = asyncio.create_task(node(failing_coro)())
        with pytest.raises(ValueError, match="boom"):
            await task

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.status == TaskStatus.FAILED
        assert isinstance(info.exception, ValueError)

    async def test_status_cancelled(self) -> None:
        async def slow_coro() -> None:
            await asyncio.sleep(10)

        task = asyncio.create_task(node(slow_coro)())
        await asyncio.sleep(0)  # let it start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.status == TaskStatus.CANCELLED

    async def test_task_info_has_correct_id(self) -> None:
        captured: list[int] = []

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            captured.append(task_id)

        task = asyncio.create_task(node(coro)())
        await task

        task_id = await get_task_id(task)
        assert captured == [task_id]

    async def test_double_init_raises(self) -> None:
        """Calling track on a task that already has info must raise."""

        async def coro() -> None:
            # _init_task_info is called once by the wrapper; calling it again
            # via a second node wrapper should raise.
            from aionode import _init_task_info

            with pytest.raises(RuntimeError, match="already initialized"):
                await _init_task_info()

        await asyncio.create_task(node(coro)())

    async def test_waiting_state_with_pending_wait_for(self) -> None:
        """node() task is in WAITING while wait_for deps are unresolved."""
        future: asyncio.Future[None] = asyncio.get_event_loop().create_future()

        async def coro() -> None:
            pass

        task = asyncio.create_task(node(coro, wait_for=[future])())
        await asyncio.sleep(0)  # let task initialise
        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.status == TaskStatus.WAITING

        future.set_result(None)
        await task


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
            child_task = asyncio.create_task(node(child_coro)())
            await child_task

        await asyncio.create_task(node(parent_coro)())
        assert child_parent_holder == parent_id_holder

    async def test_parent_subtasks_list_populated(self) -> None:
        child_id_holder: list[int] = []

        async def child_coro() -> None:
            child_id_holder.append(await get_task_id(_current_task()))

        async def parent_coro() -> None:
            child_task = asyncio.create_task(node(child_coro)())
            await child_task
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            assert child_id_holder[0] in info.subtasks

        await asyncio.create_task(node(parent_coro)())

    async def test_auto_progress_updates_parent_total_and_completed(self) -> None:
        async def child_coro() -> None:
            pass

        async def parent_coro() -> None:
            task_id = await get_task_id(_current_task())

            child_task = asyncio.create_task(node(child_coro)())
            await child_task

            info = get_task_info(task_id)
            assert info.total == 1
            assert info.completed == 1

        await asyncio.create_task(node(parent_coro)())

    async def test_subtasks_info_returns_string(self) -> None:
        async def child_coro() -> None:
            await asyncio.sleep(0.01)

        async def parent_coro() -> None:
            child_task = asyncio.create_task(node(child_coro)())
            await asyncio.sleep(0)  # let child start so it's in running_subtasks
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            text = info.subtasks_info()
            assert isinstance(text, str)
            await child_task

        await asyncio.create_task(node(parent_coro)())

    async def test_subtask_does_not_inherit_parent_deps(self) -> None:
        """Subtasks created inside a node body inherit things correctly."""
        child_ids: list[int] = []
        upstream_ids: list[int] = []

        async def child_fn() -> None:
            child_ids.append(current_task_info().id)

        async def parent_fn(x: int) -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(child_fn)(), name="child")

        async def upstream_fn() -> int:
            upstream_ids.append(current_task_info().id)
            return 1

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                up = tg.create_task(node(upstream_fn)(), name="upstream")
                tg.create_task(node(parent_fn)(resolve(up)), name="parent")

        await asyncio.create_task(node(run)())

        child_info = get_task_info(child_ids[0])
        assert child_info.tree_depth == 2  # child is a subtask of parent (tree depth 1), so tree_depth=2
        assert child_info.dag_depth == 0   # child has no dep edges
        assert child_info.deps == ()


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------


class TestLog:
    async def test_log_appends_to_task_info(self) -> None:
        async def coro() -> None:
            await log("hello")
            await log("world")

        task = asyncio.create_task(node(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert "hello" in info.logs
        assert "world" in info.logs

    async def test_log_custom_end(self) -> None:
        async def coro() -> None:
            await log("no-newline", end="")

        task = asyncio.create_task(node(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.logs == "no-newline"


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
# get_task_id / get_task
# ---------------------------------------------------------------------------


class TestGetTaskHelpers:
    async def test_get_task_id_resolves_for_running_task(self) -> None:
        task_id_holder: list[int] = []

        async def coro() -> None:
            task_id_holder.append(await get_task_id(_current_task()))

        task = asyncio.create_task(node(coro)())
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

    async def test_get_task_returns_correct_info(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(node(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task_info(task_id)
        assert info.id == task_id
        assert info.task is task

    async def test_ids_are_unique(self) -> None:
        ids: list[int] = []

        async def coro() -> None:
            ids.append(await get_task_id(_current_task()))

        tasks = [asyncio.create_task(node(coro)()) for _ in range(5)]
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

        task = asyncio.create_task(node(coro)())
        await task

        with pytest.raises(ValueError, match="No task with id"):
            remove_task(999_999)

    async def test_remove_task_makes_get_task_raise(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(node(coro)())
        await task

        task_id = await get_task_id(task)
        remove_task(task_id)

        with pytest.raises(ValueError, match="No task with id"):
            get_task_info(task_id)

    async def test_remove_task_clears_task_id_mapping(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(node(coro)())
        await task

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
            await asyncio.create_task(node(grandchild_coro)())

        async def parent_coro() -> None:
            await asyncio.create_task(node(child_coro)())

        task = asyncio.create_task(node(parent_coro)())
        await task

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
            await asyncio.create_task(node(child_coro)())

        task = asyncio.create_task(node(parent_coro)())
        await task

        parent_id = await get_task_id(task)
        child_id = child_id_holder[0]

        remove_task(child_id)

        # Parent info must still be accessible
        info = get_task_info(parent_id)
        assert info.id == parent_id


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------


class TestNode:
    async def test_node_plain_args(self) -> None:
        """All plain values passed to a sync function."""

        async def add(x: int, y: int) -> int:
            return x + y

        result = await node(add)(1, 2)
        assert result == 3

    async def test_node_awaitable_args(self) -> None:
        """All args are Futures/coroutines — should be resolved before calling."""

        async def add(x: int, y: int) -> int:
            return x + y

        loop = asyncio.get_event_loop()
        fx: asyncio.Future[int] = loop.create_future()
        fy: asyncio.Future[int] = loop.create_future()
        fx.set_result(10)
        fy.set_result(20)

        result = await node(add)(resolve(fx), resolve(fy))
        assert result == 30

    async def test_node_mixed_args(self) -> None:
        """Mix of plain and awaitable positional args."""

        async def multiply(x: int, y: int) -> int:
            return x * y

        loop = asyncio.get_event_loop()
        fx: asyncio.Future[int] = loop.create_future()
        fx.set_result(5)

        result = await node(multiply)(resolve(fx), 4)
        assert result == 20

    async def test_node_async_func(self) -> None:
        """Wrapping an async function — return value is awaited transparently."""

        async def fetch(value: int) -> int:
            await asyncio.sleep(0)
            return value * 2

        result = await node(fetch)(7)
        assert result == 14

    async def test_node_kwargs(self) -> None:
        """Keyword arguments — both plain and awaitable — are resolved."""

        async def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        loop = asyncio.get_event_loop()
        name_fut: asyncio.Future[str] = loop.create_future()
        name_fut.set_result("world")

        result = await node(greet)(name=resolve(name_fut), greeting="Hi")
        assert result == "Hi, world!"

    async def test_node_awaitable_args_gathered_concurrently(self) -> None:
        """All awaitable args are gathered concurrently, not sequentially."""
        order: list[int] = []

        async def slow(n: int) -> int:
            await asyncio.sleep(0)
            order.append(n)
            return n

        async def add(x: int, y: int) -> int:
            return x + y

        t1 = asyncio.create_task(slow(1))
        t2 = asyncio.create_task(slow(2))
        result = await node(add)(resolve(t1), resolve(t2))
        assert result == 3

    async def test_node_task_arg_without_wait_for(self) -> None:
        """Passing a Task as arg registers the dep edge without explicit wait_for."""

        async def upstream() -> int:
            return 10

        async def downstream(x: int) -> int:
            return x + 1

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                up = tg.create_task(node(upstream)(), name="up")
                tg.create_task(node(downstream)(resolve(up)), name="down")


        await asyncio.create_task(node(run)())


# ---------------------------------------------------------------------------
# wait_for / dependents / depth
# ---------------------------------------------------------------------------


class TestDepEdges:
    async def test_deps_populated_via_task_arg(self) -> None:
        """Passing Task as arg registers deps/dependents/depth edges."""

        async def upstream_fn() -> int:
            return 1

        async def downstream_fn(x: int) -> int:
            return x + 1

        upstream_ids: list[int] = []
        downstream_ids: list[int] = []

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                up = tg.create_task(node(upstream_fn)(), name="upstream")
                down = tg.create_task(node(downstream_fn)(resolve(up)), name="downstream")

                async def capture() -> None:
                    up_id = await get_task_id(up)
                    down_id = await get_task_id(down)
                    upstream_ids.append(up_id)
                    downstream_ids.append(down_id)

                tg.create_task(capture(), name="capture")


        await asyncio.create_task(node(run)())

        up_id = upstream_ids[0]
        down_id = downstream_ids[0]

        up_info = get_task_info(up_id)
        down_info = get_task_info(down_id)

        assert down_id in up_info.dependents
        assert up_id in down_info.deps
        assert up_info.dag_depth == 0
        assert down_info.dag_depth == 1

    async def test_deps_populated_via_wait_for(self) -> None:
        """node(fn, wait_for=[upstream]) populates deps/dependents/depth."""

        async def upstream_fn() -> int:
            return 1

        async def downstream_fn() -> int:
            return 2

        upstream_ids: list[int] = []
        downstream_ids: list[int] = []

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                up = tg.create_task(node(upstream_fn)(), name="upstream")
                down = tg.create_task(node(downstream_fn, wait_for=[up])(), name="downstream")

                async def capture() -> None:
                    up_id = await get_task_id(up)
                    down_id = await get_task_id(down)
                    upstream_ids.append(up_id)
                    downstream_ids.append(down_id)

                tg.create_task(capture(), name="capture")

        await asyncio.create_task(node(run)())

        up_id = upstream_ids[0]
        down_id = downstream_ids[0]

        up_info = get_task_info(up_id)
        down_info = get_task_info(down_id)

        assert down_id in up_info.dependents
        assert up_id in down_info.deps
        assert up_info.dag_depth == 0
        assert down_info.dag_depth == 1

    async def test_depth_accumulates_transitively(self) -> None:
        """A -> B -> C should give C depth=2."""

        async def fn() -> None:
            await asyncio.sleep(0)

        a_ids: list[int] = []
        b_ids: list[int] = []
        c_ids: list[int] = []

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                a = tg.create_task(node(fn)(), name="a")
                b = tg.create_task(node(fn, wait_for=[a])(), name="b")
                c = tg.create_task(node(fn, wait_for=[b])(), name="c")

                async def capture() -> None:
                    a_ids.append(await get_task_id(a))
                    b_ids.append(await get_task_id(b))
                    c_ids.append(await get_task_id(c))

                tg.create_task(capture(), name="capture")

        await asyncio.create_task(node(run)())

        a_info = get_task_info(a_ids[0])
        b_info = get_task_info(b_ids[0])
        c_info = get_task_info(c_ids[0])

        assert a_info.dag_depth == 0
        assert b_info.dag_depth == 1
        assert c_info.dag_depth == 2


# ---------------------------------------------------------------------------
# node — track=False, auto_progress=False
# ---------------------------------------------------------------------------


class TestNodeOptions:
    async def test_node_track_false(self) -> None:
        """node(fn, track=False) should not create a TaskInfo entry."""

        async def fn() -> int:
            return 99

        result = await node(fn, track=False)()
        assert result == 99

    async def test_node_auto_progress_false(self) -> None:
        """node(fn, auto_progress=False) should not auto-count its own sub-children."""

        async def grandchild_fn() -> None:
            pass

        async def child_fn() -> None:
            gc_task = asyncio.create_task(node(grandchild_fn)(), name="grandchild")
            await gc_task
            task_id = await get_task_id(_current_task())
            info = get_task_info(task_id)
            # auto_progress=False means this node doesn't auto-count grandchild
            assert info.total is None

        async def run() -> None:
            child_task = asyncio.create_task(
                node(child_fn, auto_progress=False)(), name="no-progress-child"
            )
            await child_task

        await asyncio.create_task(node(run)())

    async def test_node_preserves_function_name(self) -> None:
        """node() wrapper should preserve __name__ via functools.wraps."""

        async def my_named_func() -> int:
            return 1

        wrapped = node(my_named_func)
        assert wrapped.__name__ == "my_named_func"  # ty: ignore[unresolved-attribute]


# ---------------------------------------------------------------------------
# Diamond dependency patterns
# ---------------------------------------------------------------------------


class TestDiamondDeps:
    async def test_diamond_dependency(self) -> None:
        """A -> B, A -> C, B -> D, C -> D — diamond pattern."""

        async def fn() -> None:
            await asyncio.sleep(0)

        ids: dict[str, int] = {}

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                a = tg.create_task(node(fn)(), name="a")
                b = tg.create_task(node(fn, wait_for=[a])(), name="b")
                c = tg.create_task(node(fn, wait_for=[a])(), name="c")
                d = tg.create_task(node(fn, wait_for=[b, c])(), name="d")

                async def capture() -> None:
                    ids["a"] = await get_task_id(a)
                    ids["b"] = await get_task_id(b)
                    ids["c"] = await get_task_id(c)
                    ids["d"] = await get_task_id(d)

                tg.create_task(capture(), name="capture")

        await asyncio.create_task(node(run)())

        a_info = get_task_info(ids["a"])
        b_info = get_task_info(ids["b"])
        c_info = get_task_info(ids["c"])
        d_info = get_task_info(ids["d"])

        # a has no dep edges
        assert a_info.dag_depth == 0
        # b and c depend on a
        assert b_info.dag_depth == 1
        assert c_info.dag_depth == 1
        # d depends on b and c (dag_depth = max(1,1) + 1 = 2)
        assert d_info.dag_depth == 2
        # d has both b and c as deps
        assert ids["b"] in d_info.deps
        assert ids["c"] in d_info.deps
        # a has b and c as dependents
        assert ids["b"] in a_info.dependents
        assert ids["c"] in a_info.dependents


# ---------------------------------------------------------------------------
# Error propagation through wait_for
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    async def test_upstream_failure_propagates(self) -> None:
        """When an upstream dep fails, the downstream node should raise."""

        async def failing_fn() -> int:
            raise ValueError("upstream boom")

        async def downstream_fn(x: int) -> int:
            return x + 1

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                up = tg.create_task(node(failing_fn)(), name="failing")
                tg.create_task(node(downstream_fn)(resolve(up)), name="downstream")

        with pytest.raises(ExceptionGroup):
            await asyncio.create_task(node(run)())


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------


class TestCircularDeps:
    async def test_circular_dep_raises(self) -> None:
        """Creating a circular dependency should raise RuntimeError."""
        from aionode import _register_dep

        async def fn() -> None:
            await asyncio.sleep(10)

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                a = tg.create_task(node(fn)(), name="a")
                b = tg.create_task(node(fn, wait_for=[a])(), name="b")

                a_id = await get_task_id(a)
                b_id = await get_task_id(b)

                # b already depends on a; adding a depends on b should cycle
                with pytest.raises(RuntimeError, match="Circular dependency"):
                    await _register_dep(a_id, b_id)

                a.cancel()
                b.cancel()

        try:
            await asyncio.create_task(node(run)())
        except BaseException:
            pass


# ---------------------------------------------------------------------------
# walk_tree / walk_dag
# ---------------------------------------------------------------------------


class TestWalkTree:
    async def test_walk_tree_dfs_order(self) -> None:
        """Parent appears before its children in DFS pre-order."""

        async def child_fn() -> None:
            await asyncio.sleep(0)

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(child_fn)(), name="c1")
                tg.create_task(node(child_fn)(), name="c2")

        root_task = asyncio.create_task(node(run)(), name="root")
        root_id = await get_task_id(root_task)
        await root_task

        names = [info.name for info in walk_tree(root_id)]
        assert names[0] == "root"
        assert "c1" in names
        assert "c2" in names
        assert names.index("root") < names.index("c1")
        assert names.index("root") < names.index("c2")

    async def test_walk_tree_respects_subtasks(self) -> None:
        """Enrich chunks appear as children of enrich."""

        async def chunk_fn() -> None:
            await asyncio.sleep(0)

        async def enrich_fn() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(chunk_fn)(), name="chunk-0")
                tg.create_task(node(chunk_fn)(), name="chunk-1")

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(enrich_fn)(), name="enrich")

        root_task = asyncio.create_task(node(run)(), name="root")
        root_id = await get_task_id(root_task)
        await root_task

        names = [info.name for info in walk_tree(root_id)]
        assert names.index("enrich") < names.index("chunk-0")
        assert names.index("enrich") < names.index("chunk-1")

    async def test_walk_tree_with_root(self) -> None:
        """Scoped traversal only yields subtree."""

        async def fn() -> None:
            await asyncio.sleep(0)

        subtree_id: int | None = None

        async def branch() -> None:
            nonlocal subtree_id
            subtree_id = await get_task_id(_current_task())
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(fn)(), name="leaf")

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(fn)(), name="sibling")
                tg.create_task(node(branch)(), name="branch")

        root_task = asyncio.create_task(node(run)(), name="root")
        await root_task

        assert subtree_id is not None
        names = {info.name for info in walk_tree(subtree_id)}
        assert "branch" in names
        assert "leaf" in names
        assert "sibling" not in names
        assert "root" not in names


class TestWalkDag:
    async def test_walk_dag_topological_order(self) -> None:
        """Dependencies appear before dependents."""

        async def fn() -> None:
            await asyncio.sleep(0)

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                a = tg.create_task(node(fn)(), name="a")
                b = tg.create_task(node(fn, wait_for=[a])(), name="b")
                tg.create_task(node(fn, wait_for=[b])(), name="c")

        root_task = asyncio.create_task(node(run)(), name="root")
        root_id = await get_task_id(root_task)
        await root_task

        names = [info.name for info in walk_dag(root_id)]
        assert names.index("a") < names.index("b")
        assert names.index("b") < names.index("c")

    async def test_walk_dag_isolated_nodes_after_parent(self) -> None:
        """Enrich chunks (no DAG deps) appear after their parent in topological order."""

        async def chunk_fn() -> None:
            await asyncio.sleep(0)

        async def enrich_fn() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(chunk_fn)(), name="chunk-0")
                tg.create_task(node(chunk_fn)(), name="chunk-1")

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(enrich_fn)(), name="enrich")

        root_task = asyncio.create_task(node(run)(), name="root")
        root_id = await get_task_id(root_task)
        await root_task

        names = [info.name for info in walk_dag(root_id)]
        assert names.index("enrich") < names.index("chunk-0")
        assert names.index("enrich") < names.index("chunk-1")

    async def test_walk_dag_with_root(self) -> None:
        """Scoped traversal only yields subtree."""

        async def fn() -> None:
            await asyncio.sleep(0)

        subtree_id: int | None = None

        async def branch() -> None:
            nonlocal subtree_id
            subtree_id = await get_task_id(_current_task())
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(fn)(), name="leaf")

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(fn)(), name="sibling")
                tg.create_task(node(branch)(), name="branch")

        root_task = asyncio.create_task(node(run)(), name="root")
        await root_task

        assert subtree_id is not None
        names = {info.name for info in walk_dag(subtree_id)}
        assert "branch" in names
        assert "leaf" in names
        assert "sibling" not in names

    async def test_walk_tree_and_dag_same_set(self) -> None:
        """Both iterators yield the same tasks, possibly in different order."""

        async def fn() -> None:
            await asyncio.sleep(0)

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                a = tg.create_task(node(fn)(), name="a")
                tg.create_task(node(fn, wait_for=[a])(), name="b")

        root_task = asyncio.create_task(node(run)(), name="root")
        root_id = await get_task_id(root_task)
        await root_task

        tree_ids = {info.id for info in walk_tree(root_id)}
        dag_ids = {info.id for info in walk_dag(root_id)}
        assert tree_ids == dag_ids
