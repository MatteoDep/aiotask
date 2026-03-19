import asyncio
import functools
import inspect
from collections.abc import Awaitable, Sequence
from typing import Any


def node(func: Any, /, deps: Sequence[Awaitable] | None = None, track: bool = True, auto_progress: bool = True) -> Any:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if track:
            from aiotask import _init_task_info, _start_task

            await _init_task_info(start=False, auto_progress=auto_progress)
            _start = _start_task
        else:

            async def _start() -> None:
                pass

        try:
            if deps:
                await asyncio.gather(*deps)
            resolved_args = [await a if inspect.isawaitable(a) else a for a in args]
            resolved_kwargs = {k: (await v if inspect.isawaitable(v) else v) for k, v in kwargs.items()}
        except Exception as e:
            msg = "Failed while waiting to start."
            raise RuntimeError(msg) from e

        if track:
            from aiotask import _get_state, _register_dep, _task_id

            state = _get_state()
            our_id = _task_id.get()
            dep_tasks: list[asyncio.Task] = []
            if deps:
                dep_tasks.extend(d for d in deps if isinstance(d, asyncio.Task))
            dep_tasks.extend(a for a in args if isinstance(a, asyncio.Task))
            for dep_task in dep_tasks:
                if dep_task in state.task_ids:
                    await _register_dep(our_id, state.task_ids[dep_task])

        await _start()

        result = func(*resolved_args, **resolved_kwargs)
        return await result if inspect.isawaitable(result) else result

    return wrapper
