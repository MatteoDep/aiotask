from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiotask import TaskInfo
    from aiotask._graph import TaskGraph

_STATUS_LABEL: dict[str, str] = {
    "waiting to start": "waiting",
    "running": "running",
    "done": "done",
    "failed": "failed",
    "canceled": "canceled",
}

_STATUS_COLOR: dict[str, str] = {
    "waiting to start": "90",  # dark gray
    "running": "33",           # yellow
    "done": "32",              # green
    "failed": "31",            # red
    "canceled": "31",          # red
}

BAR_WIDTH = 20


@dataclass
class RenderConfig:
    rich: bool | None = None
    bar_width: int = BAR_WIDTH
    bar_filled: str = "▰"
    bar_empty: str = "▱"


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
    status_val = info.status.value
    label = f"[{_STATUS_LABEL.get(status_val, status_val)}]"
    if use_color:
        label = _ansi(label, _STATUS_COLOR.get(status_val, "0"))

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

    return f"{prefix}{info.description:<10}  {label:<12}  {bar}  {progress:<8}  {duration}{dep_str}"


def render_text(graph: TaskGraph, config: RenderConfig | None = None) -> str:
    """Render graph as ANSI text tree."""
    if config is None:
        config = RenderConfig()
    use_color = _use_color()
    nodes = graph.nodes()
    if not nodes:
        return ""

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

    return "\n".join(lines)


def get_render(
    *,
    rich: bool | None = None,
    bar_width: int = BAR_WIDTH,
    bar_filled: str = "▰",
    bar_empty: str = "▱",
) -> Callable[[TaskGraph], str]:
    """Return a configured render callable."""
    config = RenderConfig(rich=rich, bar_width=bar_width, bar_filled=bar_filled, bar_empty=bar_empty)

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

    while True:
        output = render_fn(graph)
        line_count = output.count("\n") + 1

        if last_line_count > 0:
            sys.stdout.write(f"\033[{last_line_count}A\033[J")

        sys.stdout.write(output + "\n")
        sys.stdout.flush()
        last_line_count = line_count

        nodes = graph.nodes()
        if nodes and all(n.status in _TERMINAL for n in nodes):
            break

        await asyncio.sleep(interval)
