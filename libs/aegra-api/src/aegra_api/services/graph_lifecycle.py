"""Graph factory lifecycle hooks registered by the entrypoint lifespans.

FollowMe's ``WorkerBootstrap`` builds FollowMe graphs that own resources (HTTP
session pools, clients) created on the event loop that runs them. The generic
fork must close them on shutdown — in the SAME loop, after the executor drained
and before ``db_manager.close`` — without importing or knowing anything about
FollowMe.

Mechanism: ``langgraph_service`` exposes a process-level shutdown registry.

* ``register_graph_shutdown_hook(hook)`` — called by any graph factory at
  registration/load time. Hooks are idempotently deduplicated (an already
  registered callable is not registered twice), so reconfiguring a graph does
  not stack duplicate closures.
* ``run_graph_shutdown_hooks()`` — awaits every hook once, in order. A hook that
  raises is logged, the remaining hooks still run, and the failures are then
  re-raised (original exception if one, else ``ExceptionGroup`` /
  ``BaseExceptionGroup``). Calling it with nothing registered is a no-op;
  calling it twice is safe (the registry is cleared after the run).

The lifespans in ``main.py`` and ``worker.py`` register this right after the
LangGraph service initialises and BEFORE the executor starts, so the LIFO
``AsyncExitStack`` unwind order is: executor drain -> graph hooks ->
``db_manager.close``.

Failures are NOT swallowed: every hook runs, then the collected failures are
re-raised (the original exception when there is exactly one, otherwise an
``ExceptionGroup``/``BaseExceptionGroup``). The stack still runs its remaining
cleanup callbacks, but the process must not report a clean shutdown.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import structlog

logger = structlog.getLogger(__name__)

#: Hooks awaitable with no arguments. Registration order is preserved.
GraphShutdownHook = Callable[[], Awaitable[None]]

_shutdown_hooks: list[GraphShutdownHook] = []
_hooks_running = False


def register_graph_shutdown_hook(hook: GraphShutdownHook) -> None:
    """Register one graph-owned resource closure. Deduplicated, order kept."""
    if not callable(hook):
        raise TypeError("graph shutdown hook must be callable")
    if hook in _shutdown_hooks or any(
        getattr(existing, "__name__", None) == getattr(hook, "__name__", None)
        and getattr(existing, "__self__", None) is getattr(hook, "__self__", None)
        for existing in _shutdown_hooks
    ):
        logger.debug("Graph shutdown hook already registered; keeping one", hook=getattr(hook, "__name__", repr(hook)))
        return
    _shutdown_hooks.append(hook)
    logger.debug("Graph shutdown hook registered", hook=getattr(hook, "__name__", repr(hook)))


async def run_graph_shutdown_hooks() -> None:
    """Await every registered hook once, in registration order.

    Failures are logged and swallowed: one broken factory must not skip the
    others (or the DB close). Safe to call with an empty registry, twice, or
    concurrently (second concurrent call is a no-op).
    """
    global _hooks_running
    if _hooks_running:
        return
    hooks = list(_shutdown_hooks)
    module_hooks = collect_graph_module_hooks()
    if not hooks and not module_hooks:
        return
    _hooks_running = True
    try:
        _shutdown_hooks.clear()
        # Module hooks are appended after explicitly registered ones; a module
        # that is both registered and discoverable is closed once (identity check
        # in ``register_graph_shutdown_hook`` cannot see modules, so compare here).
        for hook in module_hooks:
            if not any(
                getattr(existing, "__self__", None) is getattr(hook, "__self__", None)
                and getattr(existing, "__func__", existing) is getattr(hook, "__func__", hook)
                for existing in hooks
            ):
                hooks.append(hook)

        # Every hook runs; failures are collected and re-raised afterwards, so a
        # broken factory never hides the others (nor the stack's DB/broker
        # cleanup) and shutdown never reports success.
        failures: list[BaseException] = []
        for hook in hooks:
            try:
                await hook()
            except BaseException as exc:  # noqa: BLE001 - collected and re-raised
                logger.exception(
                    "Graph shutdown hook failed; continuing with the rest",
                    hook=getattr(hook, "__name__", repr(hook)),
                )
                failures.append(exc)

        if not failures:
            return
        if len(failures) == 1:
            # Preserve the original exception type (CancelledError, SystemExit,
            # KeyboardInterrupt, ...) instead of wrapping it.
            raise failures[0]
        # ExceptionGroup only accepts Exception, so BaseException members
        # (CancelledError/SystemExit/KeyboardInterrupt) need BaseExceptionGroup;
        # a mixed group keeps every failure instead of dropping the ordinary
        # ones when a cancellation is also present.
        if any(not isinstance(exc, Exception) for exc in failures):
            raise BaseExceptionGroup("graph shutdown hooks failed", failures)
        raise ExceptionGroup("graph shutdown hooks failed", failures)
    finally:
        _hooks_running = False


def reset_graph_shutdown_hooks() -> None:
    """Test helper: drop all hooks without running them."""
    _shutdown_hooks.clear()


def collect_graph_module_hooks() -> list[GraphShutdownHook]:
    """Shutdown hooks exposed by the graph modules that are actually loaded.

    A dynamically loaded graph module may define a module-level ``aclose()``
    (the FollowMe ``WorkerBootstrap`` convention). We only look at modules the
    LangGraph service has already imported — keyed by ``_module_name_for(graph_id)``
    for each registered factory — and never import anything here, so shutdown
    cannot resurrect a graph or execute factory code.

    Deduplicated by module identity: two graph ids that resolve to the same
    module yield one hook. Modules without ``aclose`` (static graphs, plain
    factories) contribute nothing.
    """
    import sys

    from aegra_api.services.langgraph_service import _module_name_for, get_langgraph_service

    hooks: list[GraphShutdownHook] = []
    seen_modules: set[int] = set()
    try:
        service = get_langgraph_service()
        factories = getattr(service, "_graph_factories", {}) or {}
    except Exception:  # pragma: no cover - service not initialised yet
        logger.debug("LangGraph service unavailable while collecting graph shutdown hooks")
        return hooks

    for graph_id in list(factories):
        module = sys.modules.get(_module_name_for(graph_id))
        if module is None:
            continue
        if id(module) in seen_modules:
            continue
        seen_modules.add(id(module))
        # WorkerBootstrap contract: the module exposes a ``graph_lifecycle``
        # object with an async ``aclose()``. A bare module-level ``aclose`` is
        # also honoured, but the named object is what the real business
        # bootstrap registers, so both are checked.
        lifecycle = getattr(module, "graph_lifecycle", None)
        aclose = getattr(lifecycle, "aclose", None)
        if not callable(aclose):
            aclose = getattr(module, "aclose", None)
        if callable(aclose):
            hooks.append(aclose)
    return hooks
