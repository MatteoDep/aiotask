"""Unit tests for aiotask - asyncio task tracking library."""

import asyncio
from typing import Any

import pytest

from aiotask import (
    TaskStatus,
    get_node,
    get_node_id,
    log,
    make_async,
    make_async_generator,
    node,
    remove_node,
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
            task_id = await get_node_id(_current_task())
            info = get_node(task_id)
            with pytest.raises(RuntimeError, match="allow_edit"):
                info.description = "new description"

        await asyncio.create_task(track(coro)())

    async def test_allow_edit_permits_write(self) -> None:
        """allow_edit context manager must permit field updates."""

        async def coro() -> None:
            task_id = await get_node_id(_current_task())
            info = get_node(task_id)
            async with info.allow_edit():
                info.description = "updated"
            assert info.description == "updated"

        await asyncio.create_task(track(coro)())

    async def test_internal_fields_always_writable(self) -> None:
        """_edit_allowed and _lock can be set without the context manager."""

        async def coro() -> None:
            task_id = await get_node_id(_current_task())
            info = get_node(task_id)
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
            task_id = await get_node_id(_current_task())
            info = get_node(task_id)
            # track sets start=True by default
            assert info.started() is True

        await asyncio.create_task(track(coro)())

    async def test_done_false_while_running(self) -> None:
        async def coro() -> None:
            task_id = await get_node_id(_current_task())
            info = get_node(task_id)
            assert info.done() is False

        await asyncio.create_task(track(coro)())

    async def test_done_true_after_completion(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track(coro)())
        await task
        await _flush()

        task_id = await get_node_id(task)
        info = get_node(task_id)
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
            task_id = await get_node_id(_current_task())
            info = get_node(task_id)
            assert info.duration() > 0.0

        await asyncio.create_task(track(coro)())

    async def test_duration_frozen_after_completion(self) -> None:
        async def coro() -> None:
            await asyncio.sleep(0.01)

        task = asyncio.create_task(track(coro)())
        await task
        await _flush()

        task_id = await get_node_id(task)
        info = get_node(task_id)
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
            task_id = await get_node_id(_current_task())
            info = get_node(task_id)
            seen_status.append(info.status)

        await asyncio.create_task(track(coro)())
        assert seen_status == [TaskStatus.RUNNING]

    async def test_status_done_after_completion(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track(coro)())
        await task
        await _flush()

        task_id = await get_node_id(task)
        info = get_node(task_id)
        assert info.status == TaskStatus.DONE

    async def test_status_failed_on_exception(self) -> None:
        async def failing_coro() -> None:
            raise ValueError("boom")

        task = asyncio.create_task(track(failing_coro)())
        with pytest.raises(ValueError, match="boom"):
            await task
        await _flush()

        task_id = await get_node_id(task)
        info = get_node(task_id)
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

        task_id = await get_node_id(task)
        info = get_node(task_id)
        assert info.status == TaskStatus.CANCELLED

    async def test_task_info_has_correct_id(self) -> None:
        captured: list[int] = []

        async def coro() -> None:
            task_id = await get_node_id(_current_task())
            captured.append(task_id)

        task = asyncio.create_task(track(coro)())
        await task

        task_id = await get_node_id(task)
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
            task_id = await get_node_id(_current_task())
            info = get_node(task_id)
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
            task_id = await get_node_id(_current_task())
            info = get_node(task_id)
            child_parent_holder.append(info.parent)

        async def parent_coro() -> None:
            task_id = await get_node_id(_current_task())
            parent_id_holder.append(task_id)
            child_task = asyncio.create_task(track(child_coro)())
            await child_task

        await asyncio.create_task(track(parent_coro)())
        assert child_parent_holder == parent_id_holder

    async def test_parent_children_list_populated(self) -> None:
        child_id_holder: list[int] = []

        async def child_coro() -> None:
            child_id_holder.append(await get_node_id(_current_task()))

        async def parent_coro() -> None:
            child_task = asyncio.create_task(track(child_coro)())
            await child_task
            task_id = await get_node_id(_current_task())
            info = get_node(task_id)
            assert child_id_holder[0] in info.children

        await asyncio.create_task(track(parent_coro)())

    async def test_auto_progress_updates_parent_total_and_completed(self) -> None:
        async def child_coro() -> None:
            pass

        async def parent_coro() -> None:
            task_id = await get_node_id(_current_task())

            child_task = asyncio.create_task(track(child_coro)())
            await child_task
            await _flush()

            info = get_node(task_id)
            assert info.total == 1
            assert info.completed == 1

        await asyncio.create_task(track(parent_coro)())

    async def test_children_info_returns_string(self) -> None:
        async def child_coro() -> None:
            await asyncio.sleep(0.01)

        async def parent_coro() -> None:
            child_task = asyncio.create_task(track(child_coro)())
            await asyncio.sleep(0)  # let child start so it's in running_children
            task_id = await get_node_id(_current_task())
            info = get_node(task_id)
            text = info.children_info()
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

        task_id = await get_node_id(task)
        info = get_node(task_id)
        assert "hello" in info.logs
        assert "world" in info.logs

    async def test_log_custom_end(self) -> None:
        async def coro() -> None:
            await log("no-newline", end="")

        task = asyncio.create_task(track(coro)())
        await task

        task_id = await get_node_id(task)
        info = get_node(task_id)
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
# get_node_id / get_node
# ---------------------------------------------------------------------------


class TestGetNodeHelpers:
    async def test_get_node_id_resolves_for_running_task(self) -> None:
        task_id_holder: list[int] = []

        async def coro() -> None:
            task_id_holder.append(await get_node_id(_current_task()))

        task = asyncio.create_task(track(coro)())
        await task
        assert len(task_id_holder) == 1
        assert isinstance(task_id_holder[0], int)

    async def test_get_node_id_times_out_for_untracked_task(self) -> None:
        async def coro() -> None:
            await asyncio.sleep(1)

        task = asyncio.create_task(coro())
        with pytest.raises(TimeoutError):
            await get_node_id(task, timeout=0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_get_node_returns_correct_info(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track(coro)())
        await task

        task_id = await get_node_id(task)
        info = get_node(task_id)
        assert info.id == task_id
        assert info.task is task

    async def test_ids_are_unique(self) -> None:
        ids: list[int] = []

        async def coro() -> None:
            ids.append(await get_node_id(_current_task()))

        tasks = [asyncio.create_task(track(coro)()) for _ in range(5)]
        await asyncio.gather(*tasks)
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# remove_node
# ---------------------------------------------------------------------------


class TestRemoveNode:
    async def test_remove_node_raises_for_unknown_id(self) -> None:
        # Run a tracked task first so the loop state is initialised
        async def coro() -> None:
            pass

        task = asyncio.create_task(track(coro)())
        await task

        with pytest.raises(ValueError, match="No task with id"):
            remove_node(999_999)

    async def test_remove_node_makes_get_node_raise(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track(coro)())
        await task
        await _flush()

        task_id = await get_node_id(task)
        remove_node(task_id)

        with pytest.raises(ValueError, match="No task with id"):
            get_node(task_id)

    async def test_remove_node_clears_task_id_mapping(self) -> None:
        async def coro() -> None:
            pass

        task = asyncio.create_task(track(coro)())
        await task
        await _flush()

        task_id = await get_node_id(task)
        remove_node(task_id)

        # get_node_id polls until task appears in task_ids; after removal it must time out
        with pytest.raises(TimeoutError):
            await get_node_id(task, timeout=0.05)

    async def test_remove_node_removes_descendants(self) -> None:
        child_id_holder: list[int] = []
        grandchild_id_holder: list[int] = []

        async def grandchild_coro() -> None:
            grandchild_id_holder.append(await get_node_id(_current_task()))

        async def child_coro() -> None:
            child_id_holder.append(await get_node_id(_current_task()))
            await asyncio.create_task(track(grandchild_coro)())

        async def parent_coro() -> None:
            await asyncio.create_task(track(child_coro)())

        task = asyncio.create_task(track(parent_coro)())
        await task
        await _flush()

        parent_id = await get_node_id(task)
        child_id = child_id_holder[0]
        grandchild_id = grandchild_id_holder[0]

        remove_node(parent_id)

        for tid in (parent_id, child_id, grandchild_id):
            with pytest.raises(ValueError):
                get_node(tid)

    async def test_remove_child_does_not_affect_parent(self) -> None:
        child_id_holder: list[int] = []

        async def child_coro() -> None:
            child_id_holder.append(await get_node_id(_current_task()))

        async def parent_coro() -> None:
            await asyncio.create_task(track(child_coro)())

        task = asyncio.create_task(track(parent_coro)())
        await task
        await _flush()

        parent_id = await get_node_id(task)
        child_id = child_id_holder[0]

        remove_node(child_id)

        # Parent info must still be accessible
        info = get_node(parent_id)
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
                    up_id = await get_node_id(up)
                    down_id = await get_node_id(down)
                    upstream_ids.append(up_id)
                    downstream_ids.append(down_id)

                tg.create_task(capture(), name="capture")

        await asyncio.create_task(track(run)())
        await _flush()

        up_id = upstream_ids[0]
        down_id = downstream_ids[0]

        up_info = get_node(up_id)
        down_info = get_node(down_id)

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
                    a_ids.append(await get_node_id(a))
                    b_ids.append(await get_node_id(b))
                    c_ids.append(await get_node_id(c))

                tg.create_task(capture(), name="capture")

        await asyncio.create_task(track(run)())
        await _flush()

        a_info = get_node(a_ids[0])
        b_info = get_node(b_ids[0])
        c_info = get_node(c_ids[0])

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
        root_id = await get_node_id(root_task)
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
        root_id = await get_node_id(root_task)
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
        root_id = await get_node_id(root_task)
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
        root_id = await get_node_id(root_task)
        await root_task
        await _flush()

        graph = TaskGraph(root_id=root_id)
        path = graph.critical_path()
        assert isinstance(path, list)
        assert len(path) >= 1
