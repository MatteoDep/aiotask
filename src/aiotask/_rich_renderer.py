from __future__ import annotations

from typing import TYPE_CHECKING

from aiotask import TaskStatus

if TYPE_CHECKING:
    from aiotask._graph import TaskGraph
    from aiotask._render import RenderConfig


def render_rich(graph: TaskGraph, config: RenderConfig | None = None) -> str:
    """Render graph as a Rich table. Requires `rich` to be installed."""
    import io

    from rich import box
    from rich.console import Console
    from rich.table import Table

    from aiotask._render import RenderConfig, _fmt_duration, _progress_bar

    if config is None:
        config = RenderConfig()

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

    if graph._root_id is not None:
        root_list = [n for n in nodes if n.id == graph._root_id]
        children = [n for n in nodes if n.id != graph._root_id]
        ordered = [(root_list[0], "")] if root_list else []
        for i, n in enumerate(children):
            prefix = "└─ " if i == len(children) - 1 else "├─ "
            ordered.append((n, prefix))
    else:
        ordered = [(n, "") for n in nodes]

    for info, prefix in ordered:
        style = _STATUS_STYLE.get(info.status, "")
        bar = _progress_bar(info.completed, info.total, config.bar_width, config.bar_filled, config.bar_empty)
        total_str = str(int(info.total)) if info.total is not None else "?"
        progress = f"[blue]{bar}[/blue]  ({int(info.completed)}/{total_str})"
        duration = _fmt_duration(info)

        table.add_row(
            f"{prefix}{info.description}",
            f"[{style}]{info.status.value}[/{style}]",
            progress,
            duration,
        )

    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=True, force_terminal=True)
    console.print(table)
    return buf.getvalue()
