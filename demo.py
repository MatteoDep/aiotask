"""
aiotask demo — dependency graph with live progress rendering.

Features showcased:
  - resolve()         pass an asyncio.Task as a coroutine argument
  - wait_for=         declare a side-effect dependency without receiving its result
  - TaskInfo.update() report per-chunk progress from inside a coroutine
  - make_async()      wrap a sync function so it can be tracked
  - subtasks          enrich() fans out into per-chunk node() subtasks

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

import aiotask

# ── step functions ────────────────────────────────────────────────────────────

async def fetch_data() -> list[int]:
    await asyncio.sleep(1)
    await aiotask.log("fetched 100 records")
    return list(range(100))


async def validate(data: list[int]) -> list[int]:
    await asyncio.sleep(0.5)
    valid = [x for x in data if x % 2 == 0]
    await aiotask.log(f"kept {len(valid)} valid records")
    return valid


async def _enrich_chunk(chunk: list[int]) -> list[int]:
    await asyncio.sleep(0.5)
    return [x * 10 for x in chunk]


async def enrich(data: list[int]) -> list[int]:
    chunks = [data[i : i + 25] for i in range(0, len(data), 25)]
    async with asyncio.TaskGroup() as tg:
        subtasks = [
            tg.create_task(
                aiotask.node(_enrich_chunk)(chunk),
                name=f"enrich-chunk-{i}",
            )
            for i, chunk in enumerate(chunks)
        ]
    result = [x for t in subtasks for x in t.result()]
    await aiotask.log(f"enrichment complete: {len(result)} records")
    return result


async def process(data: list[int]) -> list[int]:
    """Process data in chunks, updating progress via TaskInfo.update()."""
    # Retrieve our own TaskInfo so we can push granular progress updates.
    info = aiotask.current_task_info()

    chunks = [data[i : i + 10] for i in range(0, len(data), 10)]
    await info.update(total=len(chunks), completed=0)

    result: list[int] = []
    for i, chunk in enumerate(chunks):
        await asyncio.sleep(0.15)
        result.extend(x * 2 for x in chunk)
        await info.update(
            completed=i + 1,
        )

    await aiotask.log(f"processed {len(result)} records in {len(chunks)} chunks")
    return result


async def load(processed: list[int], enriched: list[int]) -> int:
    merged = processed + enriched
    await asyncio.sleep(0.8)
    await aiotask.log(f"loaded {len(merged)} records")
    return len(merged)


async def notify() -> None:
    """Fire-and-forget notification — waits for load but doesn't need its result."""
    await asyncio.sleep(0.2)
    await aiotask.log("downstream systems notified")


def summarize(count: int) -> str:
    """Sync function wrapped with make_async so it can be tracked."""
    return f"Pipeline complete: {count} records processed"


# ── pipeline ──────────────────────────────────────────────────────────────────

async def pipeline() -> None:
    async with asyncio.TaskGroup() as tg:
        fetch = tg.create_task(
            aiotask.node(fetch_data)(),
            name="fetch",
        )
        valid = tg.create_task(
            aiotask.node(validate)(aiotask.resolve(fetch)),
            name="validate",
        )
        enriched = tg.create_task(
            aiotask.node(enrich)(aiotask.resolve(fetch)),
            name="enrich",
        )
        processed = tg.create_task(
            aiotask.node(process)(aiotask.resolve(valid)),
            name="process",
        )
        loaded = tg.create_task(
            aiotask.node(load)(aiotask.resolve(processed), aiotask.resolve(enriched)),
            name="load",
        )
        # wait_for: notify runs after load completes without receiving its return value
        tg.create_task(
            aiotask.node(notify, wait_for=[loaded])(),
            name="notify",
        )
        # make_async: wrap a sync function so node() can track it
        tg.create_task(
            aiotask.node(aiotask.make_async(summarize))(aiotask.resolve(loaded)),
            name="summarize",
        )


async def run_pipeline() -> tuple[asyncio.Task, aiotask.TaskGraph]:
    root = asyncio.create_task(aiotask.node(pipeline)(), name="ETL Pipeline")
    root_id = await aiotask.get_task_id(root)
    graph = aiotask.TaskGraph(root_id=root_id)
    return root, graph


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Live tree view (default)
    print("\n── Tree view ───────────────────────────")
    root, graph = await run_pipeline()
    await aiotask.watch(graph, interval=0.3)
    await root

    # Live DAG view
    print("\n── DAG view ────────────────────────────")
    dag_render = aiotask.get_render(rich=False, view="dag")
    root, graph = await run_pipeline()
    await aiotask.watch(graph, interval=0.3, renderer=dag_render)
    await root

    # Post-run graph inspection
    print("\n── Summary ─────────────────────────────")
    for status, count in graph.summary().items():
        print(f"  {status.value}: {count}")

    print("\n── Critical path ───────────────────────")
    for info in graph.critical_path():
        print(f"  {info.name}  ({info.duration():.2f}s)")

    print("\n── Upstream of 'load' ──────────────────")
    load_id = next(n.id for n in graph.nodes() if n.name == "load")
    for info in graph.upstream(load_id):
        print(f"  {info.name}")


asyncio.run(main())
