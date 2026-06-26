"""Shared agent interface — the team convention (decided 2026-06-26).

Every agent is a LangGraph wrapped behind one `.run()` method. A caller (or
another agent) constructs the agent class and passes contract file paths into
`.run()`; the agent reads them, drives its graph, writes its contract outputs,
and returns the final state. Nobody has to know about graphs, checkpointers, or
thread ids to use an agent.

The base owns the plumbing every agent repeats — building the graph, the
checkpointer, the thread id, and invoking. Each subclass fills two holes:
`build_graph()` (its graph shape) and `run()` (its file I/O + state shaping).

NOTE: this is shared across all five agents. Changing it needs the other owners'
sign-off (Golden rule 2). It is a code convention, not a data-contract change —
no handoff file formats live here.
"""

from abc import ABC, abstractmethod


class Agent(ABC):
    """Base for every pipeline agent. Subclass, implement `build_graph` and
    `run`, then callers just do `MyAgent().run(some_input="path.csv")`."""

    def __init__(self, *, checkpointer=None, thread_id="default"):
        # MemorySaver by default: state survives across `.run()` calls on one
        # instance (e.g. the Manager's iteration counter) without touching disk.
        # Inject a SqliteSaver for cross-process persistence.
        if checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        self._graph = self.build_graph(checkpointer)
        self._config = {"configurable": {"thread_id": thread_id}}

    @abstractmethod
    def build_graph(self, checkpointer):
        """Return the compiled LangGraph for this agent."""

    @abstractmethod
    def run(self, **inputs) -> dict:
        """Read contract input file(s), invoke the graph, write contract
        output file(s), and return the final state dict."""

    def _invoke(self, initial_state: dict) -> dict:
        """Run the graph once on `initial_state`. Subclass `run()` shapes the
        state from file paths and calls this."""
        return self._graph.invoke(initial_state, self._config)
