from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from aiotask import TaskInfo
    from aiotask._graph import TaskGraph

from aiotask import TaskStatus

_STATUS_LABEL: dict[TaskStatus, str] = {
    TaskStatus.WAITING: "waiting",
    TaskStatus.RUNNING: "running",
    TaskStatus.DONE: "done",
    TaskStatus.FAILED: "failed",
    TaskStatus.CANCELLED: "cancelled",
}

_STATUS_COLOR: dict[TaskStatus, str] = {
    TaskStatus.WAITING: "90",  # dark gray
    TaskStatus.RUNNING: "33",  # yellow
    TaskStatus.DONE: "32",     # green
    TaskStatus.FAILED: "31",   # red
    TaskStatus.CANCELLED: "31",  # red
}

BAR_WIDTH = 20


@dataclass
class RenderConfig:
    """Configuration for task graph rendering."""

    rich: bool | None = None
    bar_width: int = BAR_WIDTH
    bar_filled: str = "▰"
    bar_empty: str = "▱"
    view: Literal["tree", "dag"] = "tree"


def _use_color() -> bool:
    try:
        return os.isatty(sys.stdout.fileno())
    except Exception:
        return False


def _ansi(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _progress_bar(
    completed: float,
    total: float | None,
    width: int = BAR_WIDTH,
    filled: str = "▰",
    empty: str = "▱",
) -> str:
    if total is None or total == 0:
        return "─" * width
    ratio = min(completed / total, 1.0)
    n_filled = round(ratio * width)
    return filled * n_filled + empty * (width - n_filled)


def _fmt_duration(info: TaskInfo) -> str:
    if not info.started():
        return "—"
    return f"{info.duration():.1f}s"


def _fmt_node(
    info: TaskInfo,
    graph: TaskGraph,
    prefix: str = "",
    use_color: bool = False,
    config: RenderConfig | None = None,
) -> str:
    if config is None:
        config = RenderConfig()
    label = f"[{_STATUS_LABEL.get(info.status, info.status.value)}]"
    if use_color:
        label = _ansi(label, _STATUS_COLOR.get(info.status, "0"))

    bar = _progress_bar(info.completed, info.total, config.bar_width, config.bar_filled, config.bar_empty)
    if use_color:
        bar = _ansi(bar, "34")  # blue
    total_str = str(int(info.total)) if info.total is not None else "?"
    progress = f"({int(info.completed)}/{total_str})"
    duration = _fmt_duration(info)

    dep_names: list[str] = []
    for dep_id in info.deps:
        try:
            dep_info = graph.node(dep_id)
            dep_names.append(dep_info.description)
        except Exception:
            pass
    dep_str = f"  ← deps: {', '.join(dep_names)}" if dep_names else ""

    return f"{prefix}{info.description}  {label}  {bar}  {progress}  {duration}{dep_str}"


def _render_tree(graph: TaskGraph, config: RenderConfig, use_color: bool) -> list[str]:
    """Render as a subtask tree (root with nested children)."""
    nodes = graph.nodes()
    if not nodes:
        return []
    lines: list[str] = []
    if graph.root_id is not None:
        root_list = [n for n in nodes if n.id == graph.root_id]
        rest = [n for n in nodes if n.id != graph.root_id]
        if root_list:
            lines.append(_fmt_node(root_list[0], graph, use_color=use_color, config=config))
            if rest:
                lines.append("│")
            for i, n in enumerate(rest):
                prefix = "└─ " if i == len(rest) - 1 else "├─ "
                lines.append(_fmt_node(n, graph, prefix=prefix, use_color=use_color, config=config))
        else:
            for n in nodes:
                lines.append(_fmt_node(n, graph, use_color=use_color, config=config))
    else:
        for n in nodes:
            lines.append(_fmt_node(n, graph, use_color=use_color, config=config))
    return lines


def _fmt_dag_label(
    info: TaskInfo,
    use_color: bool,
    config: RenderConfig,
) -> str:
    status_text = _STATUS_LABEL.get(info.status, info.status.value)
    status_str = f"[{status_text}]"
    if use_color:
        status_str = _ansi(status_str, _STATUS_COLOR.get(info.status, "0"))
    bar = _progress_bar(
        info.completed, info.total, config.bar_width, config.bar_filled, config.bar_empty,
    )
    if use_color:
        bar = _ansi(bar, "34")
    total_str = str(int(info.total)) if info.total is not None else "?"
    progress = f"({int(info.completed)}/{total_str})"
    dur = _fmt_duration(info)
    return f"{info.description}  {status_str}  {bar}  {progress}  {dur}"


def _render_dag_asciidag(graph: TaskGraph, config: RenderConfig, use_color: bool) -> list[str]:
    """Render as a visual DAG using asciidag (actual branch/merge lines)."""
    import io as _io

    from asciidag.graph import Graph as AsciiGraph
    from asciidag.node import Node as AsciiNode

    nodes = graph.nodes()
    if not nodes:
        return []

    node_map: dict[int, TaskInfo] = {n.id: n for n in nodes}
    root_id = graph.root_id

    # Build asciidag Node objects. "parents" in asciidag = dependents in our DAG
    # (reversed edges so the graph flows root→leaf top-to-bottom).
    ascii_nodes: dict[int, AsciiNode] = {}

    # First pass: create all nodes without parents
    for info in nodes:
        if info.id == root_id:
            continue
        label = _fmt_dag_label(info, use_color, config)
        ascii_nodes[info.id] = AsciiNode(label)

    # Second pass: wire up parents (= dependents, filtered to non-root nodes in graph)
    for info in nodes:
        if info.id == root_id:
            continue
        dependents_in_graph = [
            d for d in info.dependents
            if d != root_id and d in ascii_nodes
        ]
        # Sort dependents by depth (shallowest first) for consistent column layout
        dependents_in_graph.sort(key=lambda d: (node_map[d].depth, d))
        ascii_nodes[info.id].parents = [ascii_nodes[d] for d in dependents_in_graph]

    # Display order: topological (depth ASC), which is the order from graph.nodes()
    display_order = [ascii_nodes[info.id] for info in nodes if info.id != root_id and info.id in ascii_nodes]

    buf = _io.StringIO()
    g = AsciiGraph(fh=buf, use_color=False)
    g.show_nodes(display_order)
    output = buf.getvalue()
    # Strip trailing newline, split into lines
    return output.rstrip("\n").split("\n") if output.strip() else []


def _render_dag_fallback(graph: TaskGraph, config: RenderConfig, use_color: bool) -> list[str]:
    """Fallback DAG renderer: flat topological list with indent by depth (when asciidag is not installed)."""
    nodes = graph.nodes()  # already sorted by (depth, id)
    if not nodes:
        return []
    lines: list[str] = []
    for n in nodes:
        indent = "  " * n.depth
        lines.append(_fmt_node(n, graph, prefix=indent, use_color=use_color, config=config))
    return lines


def _render_dag(graph: TaskGraph, config: RenderConfig, use_color: bool) -> list[str]:
    """Render as a visual DAG. Uses asciidag if available, falls back to tree connectors."""
    try:
        return _render_dag_asciidag(graph, config, use_color)
    except ImportError:
        return _render_dag_fallback(graph, config, use_color)


def render_text(graph: TaskGraph, config: RenderConfig | None = None) -> str:
    """Render graph as ANSI text."""
    if config is None:
        config = RenderConfig()
    use_color = _use_color()
    if config.view == "dag":
        lines = _render_dag(graph, config, use_color)
    else:
        lines = _render_tree(graph, config, use_color)
    return "\n".join(lines)


def get_render(
    *,
    rich: bool | None = None,
    bar_width: int = BAR_WIDTH,
    bar_filled: str = "▰",
    bar_empty: str = "▱",
    view: Literal["tree", "dag"] = "tree",
) -> Callable[[TaskGraph], str]:
    """Return a configured render callable."""
    config = RenderConfig(rich=rich, bar_width=bar_width, bar_filled=bar_filled, bar_empty=bar_empty, view=view)

    def _render(graph: TaskGraph) -> str:
        if config.rich is not False:
            try:
                from aiotask._rich_renderer import render_rich

                return render_rich(graph, config)
            except ImportError:
                pass
        return render_text(graph, config)

    return _render


async def watch(
    graph: TaskGraph,
    *,
    interval: float = 0.5,
    renderer: Callable[[TaskGraph], str] | None = None,
) -> None:
    """Redraw graph in-place until all nodes reach a terminal status."""
    from aiotask import TaskStatus

    _TERMINAL = frozenset({TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED})
    render_fn = renderer if renderer is not None else get_render()
    last_line_count = 0

    use_ansi = _use_color()

    while True:
        output = render_fn(graph)
        line_count = output.count("\n") + 1

        if use_ansi and last_line_count > 0:
            sys.stdout.write(f"\033[{last_line_count}A\033[J")

        sys.stdout.write(output + "\n")
        sys.stdout.flush()
        last_line_count = line_count

        nodes = graph.nodes()
        if nodes and all(n.status in _TERMINAL for n in nodes):
            break

        await asyncio.sleep(interval)
