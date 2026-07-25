"""Terminal formatting for the pipeline's run output.

A thin wrapper over `rich` so the whole run prints in one voice: a header panel,
one `▸/✓/⚠/●` line per step, a divider per loop iteration, and a summary panel.
rich owns everything fiddly — measuring styled strings, sizing boxes, and turning
colour off when stdout is redirected, so `uv run main.py > run.log` stays plain
text.

Only `main.py` and `agents/pipeline_graph.py` print through this. The agents keep
their own [nadi]/[sabina]/[freddi] lines, which read fine underneath.
"""

from typing import Iterable, TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.progress import track as _track
from rich.table import Table
from rich.text import Text

console = Console()

T = TypeVar("T")

# icon, icon style, message style — one entry per kind of thing the loop reports.
STEPS = {
    "run": ("▸", "dim", "dim"),          # started; the slow ones say so
    "ok": ("✓", "green", "none"),        # finished, with its result
    "warn": ("⚠", "yellow", "yellow"),   # finished, but the result is degenerate
    "gate": ("●", "blue", "bold"),       # the Manager's verdict
}

# Width of the icon+actor column; also the indent a wrapped message hangs to.
# "  ▸ Sabina" is 10 characters, so 11 leaves exactly one space before the message.
GUTTER = 11


def step(kind: str, actor: str, message: str) -> None:
    """One line of the run's story: `✓ Sabina  val accuracy 0.390 · ★ new best`.

    A two-column grid, so a message too long for the terminal wraps to a hanging
    indent under the message column instead of falling back to column 0. The
    message is a `Text`, never markup, so a stray bracket in a path can't be
    parsed as a style tag.
    """
    icon, icon_style, message_style = STEPS[kind]
    line = Table.grid()
    line.add_column(width=GUTTER, no_wrap=True)
    line.add_column(style=message_style, overflow="fold")
    line.add_row(f"  [{icon_style}]{icon}[/] [cyan]{actor}[/]", Text(message))
    console.print(line)


def rule(label: str) -> None:
    """Divider between loop iterations — turns a flat stream of lines into blocks
    you can actually scan after a 20-minute run. `Text`, not markup, or rich's
    highlighter picks the iteration number out and colours it like a literal."""
    console.rule(Text(label, style="dim"), style="cyan", align="left")


def track(sequence: Iterable[T], description: str) -> Iterable[T]:
    """A percentage bar over an iterable, for fine-tuning's batch loop — the one
    step that otherwise buys several silent minutes with nothing on screen.

    `transient` erases the bar when the loop finishes, leaving the `✓ train` line
    as the permanent record. `disable` off a TTY skips it entirely, so a redirected
    run logs nothing here rather than a blank line where the bar used to be.
    """
    return _track(sequence, description=f"  [cyan]{description}[/]", console=console,
                  complete_style="cyan", finished_style="green", transient=True,
                  disable=not console.is_terminal)


def panel(title: str, rows: list[tuple[str, str]]) -> None:
    """A titled box of label/value rows. Values may carry rich markup."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    for label, value in rows:
        grid.add_row(label, value)
    console.print(Panel(grid, title=f"[bold]{title}[/]", title_align="left",
                        border_style="cyan", padding=(0, 1)))
