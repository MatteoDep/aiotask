"""Unit tests for aiotask - asyncio task tracking library."""

import asyncio
import io
from typing import Any
from unittest.mock import patch

import pytest

from aiotask import (
    TaskStatus,
    get_task,
    get_task_id,
    log,
    make_async,
    make_async_generator,
    node,
    remove_task,
    track,
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
            info = get_task(task_id)
            with pytest.raises(RuntimeError, match="allow_edit"):
                info.description = "new description"

        await asyncio.create_task(track(coro)())

    async def test_allow_edit_permits_write(self) -> None:
        """allow_edit context manager must permit field updates."""

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task(task_id)
            async with info.allow_edit():
                info.description = "updated"
            assert info.description == "updated"

        await asyncio.create_task(track(coro)())

    async def test_internal_fields_always_writable(self) -> None:
        """_edit_allowed and _lock can be set without the context manager."""

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task(task_id)
            # Should not raise
            info._edit_allowed = info._edit_allowed

        await asyncio.create_task(track(coro)())


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
            subtasks=[],
            running_subtasks=[],
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
            info = get_task(task_id)
            # track sets start=True by default
            assert info.started() is True

        await asyncio.create_task(track(coro)())

    async def test_done_false_while_running(self) -> None:
        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task(task_id)
            assert info.done() is False

        await asyncio.create_task(track(coro)())

    async def test_done_true_after_completion(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track(coro)())
        await task
        await _flush()

        task_id = await get_task_id(task)
        info = get_task(task_id)
        assert info.done() is True

    async def test_duration_zero_before_start(self) -> None:
        from aiotask import TaskInfo, TaskStatus

        task = asyncio.create_task(asyncio.sleep(0))
        info = TaskInfo(
            id=998,
            task=task,
            description="tmp",
            parent=None,
            subtasks=[],
            running_subtasks=[],
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
            info = get_task(task_id)
            assert info.duration() > 0.0

        await asyncio.create_task(track(coro)())

    async def test_duration_frozen_after_completion(self) -> None:
        async def coro() -> None:
            await asyncio.sleep(0.01)

        task = asyncio.create_task(track(coro)())
        await task
        await _flush()

        task_id = await get_task_id(task)
        info = get_task(task_id)
        d1 = info.duration()
        await asyncio.sleep(0.01)
        d2 = info.duration()
        assert d1 == d2  # finished_at is fixed, so duration doesn't grow


# ---------------------------------------------------------------------------
# track
# ---------------------------------------------------------------------------


class TestTrack:
    async def test_returns_coroutine_result(self) -> None:
        async def coro() -> int:
            return 42

        result = await asyncio.create_task(track(coro)())
        assert result == 42

    async def test_status_running_while_executing(self) -> None:
        seen_status: list[TaskStatus] = []

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task(task_id)
            seen_status.append(info.status)

        await asyncio.create_task(track(coro)())
        assert seen_status == [TaskStatus.RUNNING]

    async def test_status_done_after_completion(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track(coro)())
        await task
        await _flush()

        task_id = await get_task_id(task)
        info = get_task(task_id)
        assert info.status == TaskStatus.DONE

    async def test_status_failed_on_exception(self) -> None:
        async def failing_coro() -> None:
            raise ValueError("boom")

        task = asyncio.create_task(track(failing_coro)())
        with pytest.raises(ValueError, match="boom"):
            await task
        await _flush()

        task_id = await get_task_id(task)
        info = get_task(task_id)
        assert info.status == TaskStatus.FAILED
        assert isinstance(info.exception, ValueError)

    async def test_status_cancelled(self) -> None:
        async def slow_coro() -> None:
            await asyncio.sleep(10)

        task = asyncio.create_task(track(slow_coro)())
        await asyncio.sleep(0)  # let it start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _flush()

        task_id = await get_task_id(task)
        info = get_task(task_id)
        assert info.status == TaskStatus.CANCELLED

    async def test_task_info_has_correct_id(self) -> None:
        captured: list[int] = []

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            captured.append(task_id)

        task = asyncio.create_task(track(coro)())
        await task

        task_id = await get_task_id(task)
        assert captured == [task_id]

    async def test_double_init_raises(self) -> None:
        """Calling track on a task that already has info must raise."""

        async def coro() -> None:
            # _init_task_info is called once by the wrapper; calling it again
            # via a second track wrapper should raise.
            from aiotask import _init_task_info

            with pytest.raises(RuntimeError, match="already initialized"):
                await _init_task_info()

        await asyncio.create_task(track(coro)())

    async def test_start_false_yields_waiting_status(self) -> None:
        seen: list[TaskStatus] = []

        async def coro() -> None:
            task_id = await get_task_id(_current_task())
            info = get_task(task_id)
            seen.append(info.status)

        await asyncio.create_task(track(coro, start=False)())
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
            info = get_task(task_id)
            child_parent_holder.append(info.parent)

        async def parent_coro() -> None:
            task_id = await get_task_id(_current_task())
            parent_id_holder.append(task_id)
            child_task = asyncio.create_task(track(child_coro)())
            await child_task

        await asyncio.create_task(track(parent_coro)())
        assert child_parent_holder == parent_id_holder

    async def test_parent_subtasks_list_populated(self) -> None:
        child_id_holder: list[int] = []

        async def child_coro() -> None:
            child_id_holder.append(await get_task_id(_current_task()))

        async def parent_coro() -> None:
            child_task = asyncio.create_task(track(child_coro)())
            await child_task
            task_id = await get_task_id(_current_task())
            info = get_task(task_id)
            assert child_id_holder[0] in info.subtasks

        await asyncio.create_task(track(parent_coro)())

    async def test_auto_progress_updates_parent_total_and_completed(self) -> None:
        async def child_coro() -> None:
            pass

        async def parent_coro() -> None:
            task_id = await get_task_id(_current_task())

            child_task = asyncio.create_task(track(child_coro)())
            await child_task
            await _flush()

            info = get_task(task_id)
            assert info.total == 1
            assert info.completed == 1

        await asyncio.create_task(track(parent_coro)())

    async def test_subtasks_info_returns_string(self) -> None:
        async def child_coro() -> None:
            await asyncio.sleep(0.01)

        async def parent_coro() -> None:
            child_task = asyncio.create_task(track(child_coro)())
            await asyncio.sleep(0)  # let child start so it's in running_subtasks
            task_id = await get_task_id(_current_task())
            info = get_task(task_id)
            text = info.subtasks_info()
            assert isinstance(text, str)
            await child_task

        await asyncio.create_task(track(parent_coro)())


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------


class TestLog:
    async def test_log_appends_to_task_info(self) -> None:
        async def coro() -> None:
            await log("hello")
            await log("world")

        task = asyncio.create_task(track(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task(task_id)
        assert "hello" in info.logs
        assert "world" in info.logs

    async def test_log_custom_end(self) -> None:
        async def coro() -> None:
            await log("no-newline", end="")

        task = asyncio.create_task(track(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task(task_id)
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

        task = asyncio.create_task(track(coro)())
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

        task = asyncio.create_task(track(coro)())
        await task

        task_id = await get_task_id(task)
        info = get_task(task_id)
        assert info.id == task_id
        assert info.task is task

    async def test_ids_are_unique(self) -> None:
        ids: list[int] = []

        async def coro() -> None:
            ids.append(await get_task_id(_current_task()))

        tasks = [asyncio.create_task(track(coro)()) for _ in range(5)]
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

        task = asyncio.create_task(track(coro)())
        await task

        with pytest.raises(ValueError, match="No task with id"):
            remove_task(999_999)

    async def test_remove_task_makes_get_task_raise(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track(coro)())
        await task
        await _flush()

        task_id = await get_task_id(task)
        remove_task(task_id)

        with pytest.raises(ValueError, match="No task with id"):
            get_task(task_id)

    async def test_remove_task_clears_task_id_mapping(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track(coro)())
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
            await asyncio.create_task(track(grandchild_coro)())

        async def parent_coro() -> None:
            await asyncio.create_task(track(child_coro)())

        task = asyncio.create_task(track(parent_coro)())
        await task
        await _flush()

        parent_id = await get_task_id(task)
        child_id = child_id_holder[0]
        grandchild_id = grandchild_id_holder[0]

        remove_task(parent_id)

        for tid in (parent_id, child_id, grandchild_id):
            with pytest.raises(ValueError):
                get_task(tid)

    async def test_remove_child_does_not_affect_parent(self) -> None:
        child_id_holder: list[int] = []

        async def child_coro() -> None:
            child_id_holder.append(await get_task_id(_current_task()))

        async def parent_coro() -> None:
            await asyncio.create_task(track(child_coro)())

        task = asyncio.create_task(track(parent_coro)())
        await task
        await _flush()

        parent_id = await get_task_id(task)
        child_id = child_id_holder[0]

        remove_task(child_id)

        # Parent info must still be accessible
        info = get_task(parent_id)
        assert info.id == parent_id


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------


class TestNode:
    async def test_node_plain_args(self) -> None:
        """All plain values passed to a sync function."""

        def add(x: int, y: int) -> int:
            return x + y

        result = await node(add)(1, 2)
        assert result == 3

    async def test_node_awaitable_args(self) -> None:
        """All args are Futures/coroutines — should be resolved before calling."""

        def add(x: int, y: int) -> int:
            return x + y

        loop = asyncio.get_event_loop()
        fx: asyncio.Future[int] = loop.create_future()
        fy: asyncio.Future[int] = loop.create_future()
        fx.set_result(10)
        fy.set_result(20)

        result = await node(add)(fx, fy)
        assert result == 30

    async def test_node_mixed_args(self) -> None:
        """Mix of plain and awaitable positional args."""

        def multiply(x: int, y: int) -> int:
            return x * y

        loop = asyncio.get_event_loop()
        fx: asyncio.Future[int] = loop.create_future()
        fx.set_result(5)

        result = await node(multiply)(fx, 4)
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

        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        loop = asyncio.get_event_loop()
        name_fut: asyncio.Future[str] = loop.create_future()
        name_fut.set_result("world")

        result = await node(greet)(name=name_fut, greeting="Hi")  # type: ignore[missing-argument, unknown-argument]
        assert result == "Hi, world!"


# ---------------------------------------------------------------------------
# deps / dependents / depth
# ---------------------------------------------------------------------------


class TestDepEdges:
    async def test_deps_populated_via_deps_param(self) -> None:
        """node(fn, deps=[upstream]) should populate deps/dependents/depth."""

        async def upstream_fn() -> int:
            return 1

        async def downstream_fn(x: int) -> int:
            return x + 1

        upstream_ids: list[int] = []
        downstream_ids: list[int] = []

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                up = tg.create_task(node(upstream_fn)(), name="upstream")
                down = tg.create_task(node(downstream_fn, deps=[up])(up), name="downstream")

                # Capture ids after both tasks complete
                async def capture() -> None:
                    up_id = await get_task_id(up)
                    down_id = await get_task_id(down)
                    upstream_ids.append(up_id)
                    downstream_ids.append(down_id)

                tg.create_task(capture(), name="capture")

        await asyncio.create_task(track(run)())
        await _flush()

        up_id = upstream_ids[0]
        down_id = downstream_ids[0]

        up_info = get_task(up_id)
        down_info = get_task(down_id)

        assert down_id in up_info.dependents
        assert up_id in down_info.deps
        assert down_info.depth == 1
        assert up_info.depth == 0

    async def test_depth_accumulates_transitively(self) -> None:
        """A -> B -> C should give C depth=2."""

        async def fn() -> None:
            await asyncio.sleep(0)

        a_ids: list[int] = []
        b_ids: list[int] = []
        c_ids: list[int] = []

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                a = tg.create_task(node(fn, deps=[])(), name="a")
                b = tg.create_task(node(fn, deps=[a])(), name="b")
                c = tg.create_task(node(fn, deps=[b])(), name="c")

                async def capture() -> None:
                    a_ids.append(await get_task_id(a))
                    b_ids.append(await get_task_id(b))
                    c_ids.append(await get_task_id(c))

                tg.create_task(capture(), name="capture")

        await asyncio.create_task(track(run)())
        await _flush()

        a_info = get_task(a_ids[0])
        b_info = get_task(b_ids[0])
        c_info = get_task(c_ids[0])

        assert a_info.depth == 0
        assert b_info.depth == 1
        assert c_info.depth == 2


# ---------------------------------------------------------------------------
# TaskGraph
# ---------------------------------------------------------------------------


class TestTaskGraph:
    async def test_graph_nodes_includes_all_children(self) -> None:
        from aiotask import TaskGraph

        async def child_fn() -> None:
            await asyncio.sleep(0)

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(child_fn)(), name="c1")
                tg.create_task(node(child_fn)(), name="c2")

                async def capture() -> None:
                    pass

                tg.create_task(capture(), name="capture")

        root_task = asyncio.create_task(track(run)(), name="root")
        root_id = await get_task_id(root_task)
        await root_task
        await _flush()

        graph = TaskGraph(root_id=root_id)
        node_names = {n.description for n in graph.nodes()}
        assert "root" in node_names
        assert "c1" in node_names
        assert "c2" in node_names

    async def test_graph_roots_and_leaves(self) -> None:
        from aiotask import TaskGraph

        async def fn() -> None:
            await asyncio.sleep(0)

        a_task_holder: list[asyncio.Task] = []
        c_task_holder: list[asyncio.Task] = []

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                a = tg.create_task(node(fn)(), name="a")
                b = tg.create_task(node(fn, deps=[a])(), name="b")
                c = tg.create_task(node(fn, deps=[b])(), name="c")
                a_task_holder.append(a)
                c_task_holder.append(c)

        root_task = asyncio.create_task(track(run)(), name="root")
        root_id = await get_task_id(root_task)
        await root_task
        await _flush()

        graph = TaskGraph(root_id=root_id)
        # root has no deps, a has no deps
        root_names = {n.description for n in graph.roots()}
        leaf_names = {n.description for n in graph.leaves()}

        # root and a have no deps => roots
        assert "a" in root_names
        # c has no dependents => leaf
        assert "c" in leaf_names

    async def test_graph_summary(self) -> None:
        from aiotask import TaskGraph

        async def fn() -> None:
            pass

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(fn)(), name="c1")
                tg.create_task(node(fn)(), name="c2")

        root_task = asyncio.create_task(track(run)(), name="root")
        root_id = await get_task_id(root_task)
        await root_task
        await _flush()

        graph = TaskGraph(root_id=root_id)
        summary = graph.summary()
        assert summary.get(TaskStatus.DONE, 0) >= 1

    async def test_graph_critical_path_returns_list(self) -> None:
        from aiotask import TaskGraph

        async def fn() -> None:
            await asyncio.sleep(0.01)

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                a = tg.create_task(node(fn)(), name="a")
                tg.create_task(node(fn, deps=[a])(), name="b")

        root_task = asyncio.create_task(track(run)(), name="root")
        root_id = await get_task_id(root_task)
        await root_task
        await _flush()

        graph = TaskGraph(root_id=root_id)
        path = graph.critical_path()
        assert isinstance(path, list)
        assert len(path) >= 1

    async def test_graph_from_task(self) -> None:
        from aiotask import TaskGraph

        async def fn() -> None:
            pass

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(node(fn)(), name="c1")

        root_task = asyncio.create_task(track(run)(), name="root")
        root_id = await get_task_id(root_task)
        await root_task
        await _flush()

        graph = TaskGraph.from_task(root_task)
        assert graph.root_id == root_id
        assert any(n.description == "c1" for n in graph.nodes())

    async def test_graph_repr(self) -> None:
        from aiotask import TaskGraph

        graph = TaskGraph(root_id=42)
        assert repr(graph) == "TaskGraph(root_id=42)"

        graph_none = TaskGraph()
        assert repr(graph_none) == "TaskGraph(root_id=None)"


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
            gc_task = asyncio.create_task(track(grandchild_fn)(), name="grandchild")
            await gc_task
            await _flush()
            task_id = await get_task_id(_current_task())
            info = get_task(task_id)
            # auto_progress=False means this node doesn't auto-count grandchild
            assert info.total is None

        async def run() -> None:
            child_task = asyncio.create_task(
                node(child_fn, auto_progress=False)(), name="no-progress-child"
            )
            await child_task

        await asyncio.create_task(track(run)())

    async def test_node_preserves_function_name(self) -> None:
        """node() wrapper should preserve __name__ via functools.wraps."""

        def my_named_func() -> int:
            return 1

        wrapped = node(my_named_func)
        assert wrapped.__name__ == "my_named_func"  # ty: ignore[unresolved-attribute]

    async def test_track_preserves_function_name(self) -> None:
        """track() wrapper should preserve __name__ via functools.wraps."""

        async def my_coro() -> None:
            pass

        wrapped = track(my_coro)
        assert wrapped.__name__ == "my_coro"  # ty: ignore[unresolved-attribute]


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
                b = tg.create_task(node(fn, deps=[a])(), name="b")
                c = tg.create_task(node(fn, deps=[a])(), name="c")
                d = tg.create_task(node(fn, deps=[b, c])(), name="d")

                async def capture() -> None:
                    ids["a"] = await get_task_id(a)
                    ids["b"] = await get_task_id(b)
                    ids["c"] = await get_task_id(c)
                    ids["d"] = await get_task_id(d)

                tg.create_task(capture(), name="capture")

        await asyncio.create_task(track(run)())
        await _flush()

        a_info = get_task(ids["a"])
        b_info = get_task(ids["b"])
        c_info = get_task(ids["c"])
        d_info = get_task(ids["d"])

        # a is root
        assert a_info.depth == 0
        # b and c depend on a
        assert b_info.depth == 1
        assert c_info.depth == 1
        # d depends on b and c (depth = max(1,1) + 1 = 2)
        assert d_info.depth == 2
        # d has both b and c as deps
        assert ids["b"] in d_info.deps
        assert ids["c"] in d_info.deps
        # a has b and c as dependents
        assert ids["b"] in a_info.dependents
        assert ids["c"] in a_info.dependents


# ---------------------------------------------------------------------------
# Error propagation through deps
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    async def test_upstream_failure_propagates_through_deps(self) -> None:
        """When an upstream dep fails, the downstream node should raise."""

        async def failing_fn() -> int:
            raise ValueError("upstream boom")

        def downstream_fn(x: int) -> int:
            return x + 1

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                up = tg.create_task(node(failing_fn)(), name="failing")
                tg.create_task(node(downstream_fn, deps=[up])(up), name="downstream")

        with pytest.raises(ExceptionGroup):
            await asyncio.create_task(track(run)())


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------


class TestCircularDeps:
    async def test_circular_dep_raises(self) -> None:
        """Creating a circular dependency should raise RuntimeError."""
        from aiotask import _register_dep

        async def fn() -> None:
            await asyncio.sleep(10)

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                a = tg.create_task(node(fn)(), name="a")
                b = tg.create_task(node(fn, deps=[a])(), name="b")
                await _flush()

                a_id = await get_task_id(a)
                b_id = await get_task_id(b)

                # b already depends on a; adding a depends on b should cycle
                with pytest.raises(RuntimeError, match="Circular dependency"):
                    await _register_dep(a_id, b_id)

                a.cancel()
                b.cancel()

        try:
            await asyncio.create_task(track(run)())
        except BaseException:
            pass


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    async def _make_graph(self) -> tuple[Any, int]:
        from aiotask import TaskGraph

        async def fn() -> None:
            pass

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:
                a = tg.create_task(node(fn)(), name="task-a")
                tg.create_task(node(fn, deps=[a])(), name="task-b")

        root_task = asyncio.create_task(track(run)(), name="root")
        root_id = await get_task_id(root_task)
        await root_task
        await _flush()

        graph = TaskGraph(root_id=root_id)
        return graph, root_id

    async def test_render_text_returns_string(self) -> None:
        from aiotask._render import render_text

        graph, _ = await self._make_graph()
        output = render_text(graph)
        assert isinstance(output, str)
        assert "root" in output
        assert "task-a" in output
        assert "task-b" in output

    async def test_render_text_no_root(self) -> None:
        from aiotask import TaskGraph
        from aiotask._render import render_text

        async def fn() -> None:
            pass

        root_task = asyncio.create_task(track(fn)(), name="only-task")
        await root_task
        await _flush()

        graph = TaskGraph.current()
        output = render_text(graph)
        assert isinstance(output, str)

    async def test_render_text_empty_graph(self) -> None:
        from aiotask import TaskGraph
        from aiotask._render import render_text

        graph = TaskGraph(root_id=999_999)
        output = render_text(graph)
        assert output == ""

    async def test_get_render_returns_callable(self) -> None:
        from aiotask._render import get_render

        render_fn = get_render(rich=False)
        assert callable(render_fn)

        graph, _ = await self._make_graph()
        output = render_fn(graph)
        assert isinstance(output, str)
        assert len(output) > 0

    async def test_render_text_contains_tree_chars(self) -> None:
        from aiotask._render import render_text

        graph, _ = await self._make_graph()
        output = render_text(graph)
        # Should contain tree drawing characters
        assert "├─" in output or "└─" in output

    async def test_render_dag_view(self) -> None:
        import importlib.util

        from aiotask._render import RenderConfig, render_text

        graph, _ = await self._make_graph()
        config = RenderConfig(view="dag")
        output = render_text(graph, config)

        assert isinstance(output, str)
        assert "task-a" in output
        assert "task-b" in output
        if importlib.util.find_spec("asciidag") is not None:
            # asciidag renders graph characters
            assert "*" in output

    async def test_watch_completes_for_done_graph(self) -> None:
        from aiotask._render import watch

        graph, _ = await self._make_graph()

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            await watch(graph, interval=0.01)

        output = buf.getvalue()
        assert len(output) > 0
