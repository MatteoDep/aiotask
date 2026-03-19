"""
aiotask demo — dependency graph with live progress rendering.

DAG shape:

    fetch_data (1s)
         │
         ├──► validate (0.5s)
         │         │
         │         └──► transform (1.5s) ──► load (0.8s)
         │
         └──► enrich (2s) ──────────────────► load (0.8s)
                                               ↑
                                         (waits for both)
"""

import asyncio

import aiotask


async def fetch_data() -> list[int]:
    await asyncio.sleep(1)
    await aiotask.log("fetched 100 records")
    return list(range(100))


async def validate(data: list[int]) -> list[int]:
    await asyncio.sleep(0.5)
    valid = [x for x in data if x % 2 == 0]
    await aiotask.log(f"kept {len(valid)} valid records")
    return valid


async def enrich(data: list[int]) -> list[int]:
    await asyncio.sleep(2)
    await aiotask.log("enrichment complete")
    return [x * 10 for x in data]


async def transform(data: list[int]) -> list[int]:
    await asyncio.sleep(1.5)
    await aiotask.log("transform complete")
    return sorted(data, reverse=True)


async def load(validated: list[int], enriched: list[int]) -> int:
    merged = validated + enriched
    await asyncio.sleep(0.8)
    await aiotask.log(f"loaded {len(merged)} records")
    return len(merged)


def summarize(count: int) -> str:
    return f"Pipeline complete: {count} records processed"


async def pipeline() -> None:
    async with asyncio.TaskGroup() as tg:
        fetch = tg.create_task(
            aiotask.node(fetch_data)(),
            name="fetch",
        )
        valid = tg.create_task(
            aiotask.node(validate, deps=[fetch])(fetch),
            name="validate",
        )
        enriched = tg.create_task(
            aiotask.node(enrich, deps=[fetch])(fetch),
            name="enrich",
        )
        transformed = tg.create_task(
            aiotask.node(transform, deps=[valid])(valid),
            name="transform",
        )
        loaded = tg.create_task(
            aiotask.node(load, deps=[transformed, enriched])(transformed, enriched),
            name="load",
        )
        # sync function — node wraps it transparently
        tg.create_task(
            aiotask.node(summarize, deps=[loaded])(loaded),
            name="summarize",
        )


async def main() -> None:
    root = asyncio.create_task(aiotask.track(pipeline)(), name="ETL Pipeline")
    root_id = await aiotask.get_node_id(root)
    graph = aiotask.TaskGraph(root_id=root_id)

    await aiotask.watch(graph, interval=0.3, renderer=aiotask.render_text)
    await root

    # Post-run graph inspection
    print("\n── Summary ─────────────────────────────")
    for status, count in graph.summary().items():
        print(f"  {status.value}: {count}")

    print("\n── Critical path ───────────────────────")
    for info in graph.critical_path():
        print(f"  {info.description}  ({info.duration():.2f}s)")

    print("\n── Upstream of 'load' ──────────────────")
    load_id = next(n.id for n in graph.nodes() if n.description == "load")
    for info in graph.upstream(load_id):
        print(f"  {info.description}")

    print("\n── Upstream of root ────────────────────")
    for info in graph.upstream(root_id):
        print(f"  {info.description}")
    root_info = aiotask.get_node(root_id)
    print(f"  root.deps: {root_info.deps}")
    print(f"  root.dependents: {root_info.dependents}")


asyncio.run(main())
