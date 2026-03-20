from __future__ import annotations

from typing import TYPE_CHECKING

from aiotask import TaskStatus

if TYPE_CHECKING:
    from aiotask import TaskInfo
    from aiotask._graph import TaskGraph
    from aiotask._render import RenderConfig


def _dag_order(graph: TaskGraph, nodes: list[TaskInfo]) -> list[tuple[str, TaskInfo, str]]:
    """Walk DAG in tree order (each node once). Returns (prefix, info, dep_annotation)."""
    node_map: dict[int, TaskInfo] = {n.id: n for n in nodes}
    root_id = graph.root_id

    def eff_deps(nid: int) -> list[int]:
        info = node_map.get(nid)
        if info is None:
            return []
        return [d for d in info.deps if d != root_id and d in node_map]

    primary: dict[int, int] = {}
    tree_kids: dict[int, list[int]] = {}
    dag_roots: list[int] = []

    for n in nodes:
        if n.id == root_id:
            continue
        deps = eff_deps(n.id)
        if not deps:
            dag_roots.append(n.id)
        else:
            best = max(deps, key=lambda d: (node_map[d].depth, -d))
            primary[n.id] = best
            tree_kids.setdefault(best, []).append(n.id)

    for pid in tree_kids:
        tree_kids[pid].sort(key=lambda c: (node_map[c].depth, c))

    if not dag_roots:
        dag_roots = sorted(n.id for n in nodes if n.id != root_id)

    result: list[tuple[str, TaskInfo, str]] = []

    def walk(nid: int, prefix: str, connector: str) -> None:
        info = node_map[nid]
        deps = eff_deps(nid)
        non_primary = [d for d in deps if d != primary.get(nid)]
        names = [node_map[d].description for d in non_primary]
        extra = f"  ← {', '.join(names)}" if names else ""
        result.append((connector, info, extra))
        kids = tree_kids.get(nid, [])
        for i, kid in enumerate(kids):
            is_last = i == len(kids) - 1
            if is_last:
                walk(kid, prefix + "     ", prefix + "└──► ")
            else:
                walk(kid, prefix + "│    ", prefix + "├──► ")

    for root_nid in dag_roots:
        walk(root_nid, "", "")

    return result


def _has_asciidag() -> bool:
    try:
        import asciidag  # noqa: F401

        return True
    except ImportError:
        return False


def _render_rich_dag(graph: TaskGraph, config: RenderConfig) -> str:
    """Render DAG view as pre-formatted asciidag text with Rich styling."""
    import io

    from rich.console import Console
    from rich.text import Text

    from aiotask._render import _render_dag_asciidag

    lines = _render_dag_asciidag(graph, config, use_color=False)
    text = Text()
    for i, line in enumerate(lines):
        if i > 0:
            text.append("\n")
        text.append(line)

    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=False, force_terminal=True)
    console.print(text)
    return buf.getvalue()


def _render_rich_table(graph: TaskGraph, config: RenderConfig) -> str:
    """Render graph as a Rich table (tree view, or DAG fallback without asciidag)."""
    import io

    from rich import box
    from rich.console import Console
    from rich.table import Table

    from aiotask._render import _fmt_duration, _progress_bar

    _STATUS_STYLE: dict[TaskStatus, str] = {
        TaskStatus.WAITING: "dim",
        TaskStatus.RUNNING: "yellow",
        TaskStatus.DONE: "green",
        TaskStatus.FAILED: "bold red",
        TaskStatus.CANCELLED: "red",
    }

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Task", style="bold")
    table.add_column("Status")
    table.add_column("Progress", no_wrap=True)
    table.add_column("Time", justify="right")

    nodes = graph.nodes()

    ordered: list[tuple[str, TaskInfo, str]]
    if config.view == "dag":
        ordered = _dag_order(graph, nodes)
    elif graph.root_id is not None:
        root_list = [n for n in nodes if n.id == graph.root_id]
        subtasks = [n for n in nodes if n.id != graph.root_id]
        ordered = [("", root_list[0], "")] if root_list else []
        for i, n in enumerate(subtasks):
            prefix = "└─ " if i == len(subtasks) - 1 else "├─ "
            ordered.append((prefix, n, ""))
    else:
        ordered = [("", n, "") for n in nodes]

    for prefix, info, extra in ordered:
        style = _STATUS_STYLE.get(info.status, "")
        bar = _progress_bar(info.completed, info.total, config.bar_width, config.bar_filled, config.bar_empty)
        total_str = str(int(info.total)) if info.total is not None else "?"
        progress = f"[blue]{bar}[/blue]  ({int(info.completed)}/{total_str})"
        duration = _fmt_duration(info)

        table.add_row(
            f"{prefix}{info.description}{extra}",
            f"[{style}]{info.status.value}[/{style}]",
            progress,
            duration,
        )

    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=True, force_terminal=True)
    console.print(table)
    return buf.getvalue()


def render_rich(graph: TaskGraph, config: RenderConfig | None = None) -> str:
    """Render graph using Rich. Uses asciidag for DAG view when available."""
    from aiotask._render import RenderConfig

    if config is None:
        config = RenderConfig()

    if config.view == "dag" and _has_asciidag():
        return _render_rich_dag(graph, config)

    return _render_rich_table(graph, config)
