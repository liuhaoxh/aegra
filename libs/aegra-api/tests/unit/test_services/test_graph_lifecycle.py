"""Graph factory lifecycle: shutdown hook registration and lifespan ordering.

The contract with FollowMe's ``WorkerBootstrap``:

* factories register an async closure with
  ``register_graph_shutdown_hook`` when they build a graph that owns resources;
* both entrypoint lifespans register ``run_graph_shutdown_hooks`` once, as the
  last startup step, so it unwinds after the executor drain and before
  ``db_manager.close``, in the same event loop;
* nothing registered → safe no-op; duplicate registration → one closure;
  a failing hook → logged, the remaining hooks still run, other cleanup still
  runs.

Only the IO-resource boundaries are faked (DB/Redis/executor/schedulers); the
``AsyncExitStack`` ordering, the hook registry and the event loop are real.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.services.graph_lifecycle import (
    register_graph_shutdown_hook,
    reset_graph_shutdown_hooks,
    run_graph_shutdown_hooks,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    reset_graph_shutdown_hooks()
    yield
    reset_graph_shutdown_hooks()


# ---------------------------------------------------------------------------
# Registry semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_registry_is_a_noop() -> None:
    await run_graph_shutdown_hooks()  # must not raise


@pytest.mark.asyncio
async def test_hooks_run_in_registration_order_and_once() -> None:
    calls: list[str] = []

    async def first() -> None:
        calls.append("first")

    async def second() -> None:
        calls.append("second")

    register_graph_shutdown_hook(first)
    register_graph_shutdown_hook(second)
    await run_graph_shutdown_hooks()
    assert calls == ["first", "second"]

    await run_graph_shutdown_hooks()  # registry cleared: no repeat
    assert calls == ["first", "second"]


@pytest.mark.asyncio
async def test_duplicate_registration_closes_once() -> None:
    calls: list[int] = []

    class Factory:
        async def aclose(self) -> None:
            calls.append(1)

    factory = Factory()
    register_graph_shutdown_hook(factory.aclose)
    register_graph_shutdown_hook(factory.aclose)  # reconfiguration of the same graph
    register_graph_shutdown_hook(factory.aclose)

    await run_graph_shutdown_hooks()
    assert calls == [1], "a reconfigured graph must not be closed more than once"


@pytest.mark.asyncio
async def test_failing_hook_runs_the_rest_then_raises() -> None:
    """All hooks run; the failure surfaces instead of a clean shutdown."""
    calls: list[str] = []

    async def broken() -> None:
        raise RuntimeError("factory resource close failed")

    async def healthy() -> None:
        calls.append("healthy")

    register_graph_shutdown_hook(broken)
    register_graph_shutdown_hook(healthy)
    # A single failure is re-raised as the ORIGINAL exception (not a group).
    with pytest.raises(RuntimeError, match="factory resource close failed"):
        await run_graph_shutdown_hooks()
    assert calls == ["healthy"], "the remaining hooks must still run"


@pytest.mark.asyncio
async def test_cancellation_runs_remaining_hooks_then_propagates() -> None:
    calls: list[str] = []

    async def cancelled() -> None:
        raise asyncio.CancelledError

    async def healthy() -> None:
        calls.append("healthy")

    register_graph_shutdown_hook(cancelled)
    register_graph_shutdown_hook(healthy)
    with pytest.raises(asyncio.CancelledError):
        await run_graph_shutdown_hooks()
    assert calls == ["healthy"], "cancellation must not skip the remaining cleanup"


@pytest.mark.asyncio
async def test_hook_runs_on_the_running_loop() -> None:
    seen: dict[str, Any] = {}

    async def hook() -> None:
        seen["loop"] = asyncio.get_running_loop()

    register_graph_shutdown_hook(hook)
    await run_graph_shutdown_hooks()
    assert seen["loop"] is asyncio.get_running_loop()


# ---------------------------------------------------------------------------
# Lifespan ordering: graph hooks after executor drain, before DB close
# ---------------------------------------------------------------------------


def _order_recorder(order: list[str]):
    """Fake IO boundaries that record their shutdown order."""

    async def db_close() -> None:
        order.append("db_close")

    async def executor_stop() -> None:
        order.append("executor_stop")

    async def redis_close() -> None:
        order.append("redis_close")

    async def broker_stop() -> None:
        order.append("broker_stop")

    async def reaper_stop() -> None:
        order.append("reaper_stop")

    async def cron_stop() -> None:
        order.append("cron_stop")

    async def ttl_stop() -> None:
        order.append("ttl_stop")

    async def graph_close() -> None:
        order.append("graph_close")

    return {
        "db_close": db_close,
        "executor_stop": executor_stop,
        "redis_close": redis_close,
        "broker_stop": broker_stop,
        "reaper_stop": reaper_stop,
        "cron_stop": cron_stop,
        "ttl_stop": ttl_stop,
        "graph_close": graph_close,
    }


@pytest.mark.asyncio
async def test_worker_lifespan_closes_graphs_after_drain_before_db() -> None:
    from aegra_api import worker as worker_module

    order: list[str] = []
    fakes = _order_recorder(order)
    register_graph_shutdown_hook(fakes["graph_close"])

    db = MagicMock(initialize=AsyncMock(), close=fakes["db_close"])
    redis = MagicMock(initialize=AsyncMock(), close=fakes["redis_close"])
    broker = MagicMock(start=AsyncMock(), stop=fakes["broker_stop"])
    executor = MagicMock(start=AsyncMock(), stop=fakes["executor_stop"])
    reaper = MagicMock(start=AsyncMock(), stop=fakes["reaper_stop"])
    cron = MagicMock(start=AsyncMock(), stop=fakes["cron_stop"])
    lg = MagicMock(initialize=AsyncMock())

    with (
        patch.object(worker_module, "db_manager", db),
        patch.object(worker_module, "redis_manager", redis),
        patch.object(worker_module, "broker_manager", broker),
        patch.object(worker_module, "executor", executor),
        patch.object(worker_module, "lease_reaper", reaper),
        patch.object(worker_module, "cron_scheduler", cron),
        patch.object(worker_module, "get_langgraph_service", return_value=lg),
        patch.object(worker_module.settings.redis, "REDIS_BROKER_ENABLED", True),
        patch.object(worker_module.settings.cron, "CRON_ENABLED", True),
        patch.object(worker_module, "setup_observability"),
    ):
        async with worker_module.worker_lifespan():
            pass

    assert "graph_close" in order, order
    assert order.index("executor_stop") < order.index("graph_close") < order.index("db_close"), order


@pytest.mark.asyncio
async def test_api_lifespan_closes_graphs_after_drain_before_db() -> None:
    from aegra_api import main as main_module

    order: list[str] = []
    fakes = _order_recorder(order)
    register_graph_shutdown_hook(fakes["graph_close"])

    db = MagicMock(initialize=AsyncMock(), close=fakes["db_close"])
    redis = MagicMock(initialize=AsyncMock(), close=fakes["redis_close"])
    broker = MagicMock(start=AsyncMock(), stop=fakes["broker_stop"])
    executor = MagicMock(start=AsyncMock(), stop=fakes["executor_stop"])
    reaper = MagicMock(start=AsyncMock(), stop=fakes["reaper_stop"])
    cron = MagicMock(start=AsyncMock(), stop=fakes["cron_stop"])
    ttl = MagicMock(start=AsyncMock(), stop=fakes["ttl_stop"])
    lg = MagicMock(initialize=AsyncMock())

    with (
        patch.object(main_module, "db_manager", db),
        patch.object(main_module, "redis_manager", redis),
        patch.object(main_module, "broker_manager", broker),
        patch.object(main_module, "executor", executor),
        patch.object(main_module, "lease_reaper", reaper),
        patch.object(main_module, "cron_scheduler", cron),
        patch.object(main_module, "thread_ttl_sweeper", ttl),
        patch.object(main_module, "get_langgraph_service", return_value=lg),
        patch.object(main_module, "get_default_durability", lambda: None),
        patch.object(main_module, "_run_startup_migrations", AsyncMock()),
        patch.object(main_module, "api_process_consumes_queue", lambda: True),
        patch.object(main_module.settings.redis, "REDIS_BROKER_ENABLED", True),
        patch.object(main_module.settings.cron, "CRON_ENABLED", True),
        patch.object(main_module, "get_thread_ttl_config", lambda: object()),
        patch.object(main_module, "setup_observability"),
    ):
        async with main_module.lifespan(MagicMock()):
            pass

    assert "graph_close" in order, order
    assert order.index("executor_stop") < order.index("graph_close") < order.index("db_close"), order

# ---------------------------------------------------------------------------
# Module-level hooks from dynamically loaded graph modules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_module_level_aclose_is_discovered_without_reimport() -> None:
    """A loaded graph module's ``aclose`` runs; nothing is re-imported."""
    import sys
    import types

    from aegra_api.services import langgraph_service as lgs
    from aegra_api.services.graph_lifecycle import collect_graph_module_hooks

    calls: list[str] = []
    module = types.ModuleType("_fake_followme_graph_module")

    async def _aclose() -> None:
        calls.append("closed")

    # WorkerBootstrap contract: a module-level ``graph_lifecycle`` object.
    module.graph_lifecycle = types.SimpleNamespace(aclose=_aclose)
    sys.modules[module.__name__] = module

    service = MagicMock()
    service._graph_factories = {"fake-graph": lambda *a, **k: None}
    try:
        with (
            patch.object(lgs, "get_langgraph_service", return_value=service),
            patch.object(lgs, "_module_name_for", return_value=module.__name__),
        ):
            hooks = collect_graph_module_hooks()
            assert hooks == [module.graph_lifecycle.aclose]
            await run_graph_shutdown_hooks()
        assert calls == ["closed"]
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.mark.asyncio
async def test_module_hooks_dedupe_by_module_identity() -> None:
    import sys
    import types

    from aegra_api.services import langgraph_service as lgs
    from aegra_api.services.graph_lifecycle import collect_graph_module_hooks

    module = types.ModuleType("_fake_shared_module")
    calls: list[int] = []

    async def _aclose() -> None:
        calls.append(1)

    module.graph_lifecycle = types.SimpleNamespace(aclose=_aclose)
    sys.modules[module.__name__] = module

    service = MagicMock()
    service._graph_factories = {"g1": lambda: None, "g2": lambda: None}
    try:
        with (
            patch.object(lgs, "get_langgraph_service", return_value=service),
            patch.object(lgs, "_module_name_for", return_value=module.__name__),
        ):
            assert len(collect_graph_module_hooks()) == 1
            await run_graph_shutdown_hooks()
        assert calls == [1], "two graph ids on one module must close it once"
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.mark.asyncio
async def test_modules_without_hook_are_ignored_and_missing_modules_skipped() -> None:
    import sys
    import types

    from aegra_api.services import langgraph_service as lgs
    from aegra_api.services.graph_lifecycle import collect_graph_module_hooks

    plain = types.ModuleType("_fake_plain_module")
    sys.modules[plain.__name__] = plain

    service = MagicMock()
    service._graph_factories = {"plain": lambda: None, "never-loaded": lambda: None}

    def _name_for(graph_id: str) -> str:
        return plain.__name__ if graph_id == "plain" else "_fake_never_imported_module"

    try:
        with (
            patch.object(lgs, "get_langgraph_service", return_value=service),
            patch.object(lgs, "_module_name_for", side_effect=_name_for),
        ):
            assert collect_graph_module_hooks() == []
            await run_graph_shutdown_hooks()  # no-op, must not raise
    finally:
        sys.modules.pop(plain.__name__, None)
        assert "_fake_never_imported_module" not in sys.modules, "shutdown must not import modules"

# ---------------------------------------------------------------------------
# Raw AsyncExitStack order: the registration order itself must be correct
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stack_unwind_order_is_drain_then_graphs_then_db() -> None:
    """Mirrors the lifespan registration sequence with a real AsyncExitStack.

    ``db_manager.close`` is registered first (unwinds last), the graph hook
    second, the executor stop last (unwinds first). If the graph hook were
    registered after the executor (the earlier bug), this order inverts and the
    test fails.
    """
    order: list[str] = []

    async def db_close() -> None:
        order.append("db_close")

    async def graph_close() -> None:
        order.append("graph_close")

    async def executor_stop() -> None:
        order.append("executor_stop")

    register_graph_shutdown_hook(graph_close)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(db_close)
        stack.push_async_callback(run_graph_shutdown_hooks)
        stack.push_async_callback(executor_stop)

    assert order == ["executor_stop", "graph_close", "db_close"], order


@pytest.mark.asyncio
async def test_stack_still_runs_db_close_when_graph_hook_fails() -> None:
    """A failing graph hook must not prevent the remaining stack callbacks."""
    order: list[str] = []

    async def db_close() -> None:
        order.append("db_close")

    async def broken() -> None:
        raise RuntimeError("graph close failed")

    register_graph_shutdown_hook(broken)
    with pytest.raises(RuntimeError, match="graph close failed"):
        async with AsyncExitStack() as stack:
            stack.push_async_callback(db_close)
            stack.push_async_callback(run_graph_shutdown_hooks)
    assert order == ["db_close"], "DB close must still run after a failing graph hook"

@pytest.mark.asyncio
async def test_single_failure_reraises_original_exception() -> None:
    async def broken() -> None:
        raise RuntimeError("close failed")

    register_graph_shutdown_hook(broken)
    with pytest.raises(RuntimeError, match="close failed"):
        await run_graph_shutdown_hooks()


@pytest.mark.asyncio
async def test_cancellation_and_ordinary_failure_both_survive() -> None:
    """A cancellation must not silently drop an ordinary hook failure."""
    calls: list[str] = []

    async def cancelled() -> None:
        raise asyncio.CancelledError

    async def broken() -> None:
        raise RuntimeError("close failed")

    async def healthy() -> None:
        calls.append("healthy")

    register_graph_shutdown_hook(cancelled)
    register_graph_shutdown_hook(broken)
    register_graph_shutdown_hook(healthy)

    with pytest.raises(BaseExceptionGroup) as exc:
        await run_graph_shutdown_hooks()

    kinds = {type(e).__name__ for e in exc.value.exceptions}
    assert kinds == {"CancelledError", "RuntimeError"}, kinds
    assert calls == ["healthy"]


@pytest.mark.asyncio
async def test_base_exception_group_does_not_masquerade_as_typeerror() -> None:
    """SystemExit is a BaseException: grouping it must not raise TypeError."""

    async def exits() -> None:
        raise SystemExit(3)

    async def broken() -> None:
        raise RuntimeError("close failed")

    register_graph_shutdown_hook(exits)
    register_graph_shutdown_hook(broken)
    with pytest.raises(BaseExceptionGroup) as exc:
        await run_graph_shutdown_hooks()
    assert {type(e).__name__ for e in exc.value.exceptions} == {"SystemExit", "RuntimeError"}
