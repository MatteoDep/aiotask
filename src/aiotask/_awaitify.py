import inspect
from typing import Any


def awaitify(func: Any) -> Any:
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        resolved_args = [await a if inspect.isawaitable(a) else a for a in args]
        resolved_kwargs = {k: (await v if inspect.isawaitable(v) else v) for k, v in kwargs.items()}
        result = func(*resolved_args, **resolved_kwargs)
        return await result if inspect.isawaitable(result) else result

    return wrapper
