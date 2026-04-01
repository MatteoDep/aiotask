"""
aionode demo — dependency graph with walk_tree / walk_dag traversal.

Features showcased:
  - resolve()         pass an asyncio.Task as a coroutine argument
  - wait_for=         declare a side-effect dependency without receiving its result
  - TaskInfo.update() report per-chunk progress from inside a coroutine
  - make_async()      wrap a sync function so it can be tracked
  - subtasks          enrich() fans out into per-chunk node() subtasks
  - walk_tree()       DFS pre-order traversal (tree view)
  - walk_dag()        topological order traversal (DAG view)

DAG shape:

    fetch_data (1s)
         │
         ├──► validate (0.5s)
         │         │
         │         └──► process (1.5s, manual progress) ──► load (0.8s)
         │                                                        │
         └──► enrich (2s) ─────────────────────────────────────► ┘
                                                                  │
                                               notify (wait_for only, no arg)
                                               summarize (sync → make_async)
"""

import asyncio

import aionode

# ── step functions ────────────────────────────────────────────────────────────


async def fetch_data() -> list[int]:
    await asyncio.sleep(1)
    await aionode.log("fetched 100 records")
    return list(range(100))


async def validate(data: list[int]) -> list[int]:
    await asyncio.sleep(0.5)
    valid = [x for x in data if x % 2 == 0]
    await aionode.log(f"kept {len(valid)} valid records")
    return valid


async def _enrich_chunk(chunk: list[int]) -> list[int]:
    await asyncio.sleep(0.5)
    return [x * 10 for x in chunk]


async def enrich(data: list[int]) -> list[int]:
    chunks = [data[i : i + 25] for i in range(0, len(data), 25)]
    async with asyncio.TaskGroup() as tg:
        subtasks = [
            tg.create_task(
                aionode.node(_enrich_chunk)(chunk),
                name=f"enrich-chunk-{i}",
            )
            for i, chunk in enumerate(chunks)
        ]
    result = [x for t in subtasks for x in t.result()]
    await aionode.log(f"enrichment complete: {len(result)} records")
    return result


async def process(data: list[int]) -> list[int]:
    """Process data in chunks, updating progress via TaskInfo.update()."""
    # Retrieve our own TaskInfo so we can push granular progress updates.
    info = aionode.current_task_info()

    chunks = [data[i : i + 10] for i in range(0, len(data), 10)]
    await info.update(total=len(chunks), completed=0)

    result: list[int] = []
    for i, chunk in enumerate(chunks):
        await asyncio.sleep(0.15)
        result.extend(x * 2 for x in chunk)
        await info.update(
            completed=i + 1,
        )

    await aionode.log(f"processed {len(result)} records in {len(chunks)} chunks")
    return result


async def load(processed: list[int], enriched: list[int]) -> int:
    merged = processed + enriched
    await asyncio.sleep(0.8)
    await aionode.log(f"loaded {len(merged)} records")
    return len(merged)


async def notify() -> None:
    """Fire-and-forget notification — waits for load but doesn't need its result."""
    await asyncio.sleep(0.2)
    await aionode.log("downstream systems notified")


def summarize(count: int) -> str:
    """Sync function wrapped with make_async so it can be tracked."""
    return f"Pipeline complete: {count} records processed"


# ── pipeline ──────────────────────────────────────────────────────────────────


async def pipeline() -> None:
    async with asyncio.TaskGroup() as tg:
        fetch = tg.create_task(
            aionode.node(fetch_data)(),
            name="fetch",
        )
        valid = tg.create_task(
            aionode.node(validate)(aionode.resolve(fetch)),
            name="validate",
        )
        enriched = tg.create_task(
            aionode.node(enrich)(aionode.resolve(fetch)),
            name="enrich",
        )
        processed = tg.create_task(
            aionode.node(process)(aionode.resolve(valid)),
            name="process",
        )
        loaded = tg.create_task(
            aionode.node(load)(aionode.resolve(processed), aionode.resolve(enriched)),
            name="load",
        )
        # wait_for: notify runs after load completes without receiving its return value
        tg.create_task(
            aionode.node(notify, wait_for=[loaded])(),
            name="notify",
        )
        # make_async: wrap a sync function so node() can track it
        tg.create_task(
            aionode.node(aionode.make_async(summarize))(aionode.resolve(loaded)),
            name="summarize",
        )


# ── rendering helpers ─────────────────────────────────────────────────────────


def _status_icon(info: aionode.TaskInfo) -> str:
    match info.status:
        case aionode.TaskStatus.DONE:
            return "+"
        case aionode.TaskStatus.FAILED:
            return "x"
        case aionode.TaskStatus.RUNNING:
            return "~"
        case aionode.TaskStatus.CANCELLED:
            return "!"
        case _:
            return "."


def _progress(info: aionode.TaskInfo) -> str:
    if info.total is not None:
        return f" [{info.completed:.0f}/{info.total:.0f}]"
    return ""


def print_tree(root_id: int) -> None:
    for info in aionode.walk_tree(root_id):
        indent = "  " * info.tree_depth
        icon = _status_icon(info)
        print(f"{indent}[{icon}] {info.name}{_progress(info)}  ({info.duration():.2f}s)")


def print_dag(root_id: int) -> None:
    for info in aionode.walk_dag(root_id):
        isolated = " (isolated)" if not info.deps and not info.dependents else ""
        dep_names = ", ".join(
            aionode.get_task_info(d).name for d in info.deps
        )
        deps_str = f"  <- {dep_names}" if dep_names else ""
        icon = _status_icon(info)
        print(f"  [{icon}] {info.name}{_progress(info)}{deps_str}{isolated}")


# ── main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    root = asyncio.create_task(aionode.node(pipeline)(), name="ETL Pipeline")
    root_id = await aionode.get_task_id(root)
    await root

    print("\n── Tree view ───────────────────────────")
    print_tree(root_id)

    print("\n── DAG view ────────────────────────────")
    print_dag(root_id)

    # Show isolated nodes (enrich chunks have no DAG deps/dependents)
    print("\n── Isolated nodes ──────────────────────")
    for info in aionode.walk_dag(root_id):
        if not info.deps and not info.dependents:
            parent_name = aionode.get_task_info(info.parent).name if info.parent is not None else "none"
            print(f"  {info.name} (parent: {parent_name})")


asyncio.run(main())
