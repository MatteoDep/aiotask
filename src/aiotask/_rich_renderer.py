from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiotask._graph import TaskGraph


def render_rich(graph: TaskGraph) -> str:
    """Render graph as a Rich table. Requires `rich` to be installed."""
    import io

    from rich import box
    from rich.console import Console
    from rich.table import Table

    from aiotask._render import _fmt_duration, _progress_bar

    _STATUS_STYLE: dict[str, str] = {
        "waiting to start": "dim",
        "running": "yellow",
        "done": "green",
        "failed": "bold red",
        "canceled": "red",
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
        status_val = info.status.value
        style = _STATUS_STYLE.get(status_val, "")
        bar = _progress_bar(info.completed, info.total)
        total_str = str(int(info.total)) if info.total is not None else "?"
        progress = f"{bar}  ({int(info.completed)}/{total_str})"
        duration = _fmt_duration(info)

        table.add_row(
            f"{prefix}{info.description}",
            f"[{style}]{status_val}[/{style}]",
            progress,
            duration,
        )

    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=True)
    console.print(table)
    return buf.getvalue()
