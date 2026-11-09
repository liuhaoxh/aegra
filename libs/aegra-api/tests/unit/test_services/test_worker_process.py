"""Behavior tests for process role separation, the standalone worker, and
checkpoint-injection policy.

Deployment contract under test (trusted env only, never request/job data):

* ``WORKER_ROLE`` unset or ``all`` — legacy single-process mode: the HTTP API
  process consumes the Redis job queue (pre-role fork default).
* ``WORKER_ROLE=api`` — API process is enqueue-only; never starts consumers.
* ``WORKER_ROLE=worker`` — API entrypoint refuses to boot; the dedicated
  ``python -m aegra_api.worker`` process is the consumer.
* ``AEGRA_PERSISTENCE_REQUIRED=true`` — a graph that cannot receive the
  server checkpointer/store refuses to execute (fail closed). Default false
  keeps the legacy unpersisted fallback for graphs without injection support.
* Server-authenticated identity overrides a client-supplied
  ``configurable.user_id`` (no tenant masquerade).

The tests drive real objects (WorkerExecutor stubs for I/O boundaries,
LangGraphService, the worker lifespan) rather than string mocks.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aegra_api.main as main_module
import aegra_api.worker as worker_module
from aegra_api.core.process_role import ProcessRoleError, api_process_consumes_queue
from aegra_api.services.langgraph_service import LangGraphService, inject_user_context
from aegra_api.settings import settings


def _stub_main_lifespan(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Replace every startup component main's lifespan touches with recording stubs."""
    stubs: dict[str, MagicMock] = {}
    for name, start, stop in (
        ("db_manager", "initialize", "close"),
        ("redis_manager", "initialize", "close"),
        ("broker_manager", "start", "stop"),
        ("executor", "start", "stop"),
        ("lease_reaper", "start", "stop"),
    ):
        obj = MagicMock()
        setattr(obj, start, AsyncMock())
        setattr(obj, stop, AsyncMock())
        monkeypatch.setattr(main_module, name, obj)
        stubs[name] = obj

    graph_service = MagicMock()
    graph_service.initialize = AsyncMock()
    monkeypatch.setattr(main_module, "get_langgraph_service", lambda: graph_service)
    stubs["graph_service"] = graph_service

    # Fully offline: no startup migrations, no database contact anywhere in
    # these lifespan tests.
    monkeypatch.setattr(main_module.settings.app, "RUN_MIGRATIONS_ON_STARTUP", False)
    monkeypatch.setattr(main_module, "run_migrations_async", AsyncMock())
    monkeypatch.setattr(main_module.settings.redis, "REDIS_BROKER_ENABLED", True)
    return stubs


def _stub_worker_lifespan(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Replace worker bootstrap components with recording stubs."""
    stubs: dict[str, MagicMock] = {}
    for name, start, stop in (
        ("db_manager", "initialize", "close"),
        ("redis_manager", "initialize", "close"),
        ("broker_manager", "start", "stop"),
        ("executor", "start", "stop"),
        ("lease_reaper", "start", "stop"),
        ("cron_scheduler", "start", "stop"),
    ):
        obj = MagicMock()
        setattr(obj, start, AsyncMock())
        setattr(obj, stop, AsyncMock())
        monkeypatch.setattr(worker_module, name, obj)
        stubs[name] = obj

    graph_service = MagicMock()
    graph_service.initialize = AsyncMock()
    monkeypatch.setattr(worker_module, "get_langgraph_service", lambda: graph_service)
    stubs["graph_service"] = graph_service
    return stubs


# ------------------------------------------------------------------
# Role resolution
# ------------------------------------------------------------------


class TestProcessRoleResolution:
    def test_unset_role_keeps_legacy_consume(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pre-role deployments (no WORKER_ROLE) keep executing on the API process."""
        monkeypatch.setattr(settings.worker, "WORKER_ROLE", None)
        assert api_process_consumes_queue() is True

    def test_all_role_is_legacy_consume(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.worker, "WORKER_ROLE", "all")
        assert api_process_consumes_queue() is True

    def test_api_role_is_enqueue_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.worker, "WORKER_ROLE", "api")
        assert api_process_consumes_queue() is False

    def test_worker_role_conflicts_with_api_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.worker, "WORKER_ROLE", "worker")
        with pytest.raises(ProcessRoleError, match="aegra_api.worker"):
            api_process_consumes_queue()


# ------------------------------------------------------------------
# API process lifespan
# ------------------------------------------------------------------


class TestApiLifespanQueuePolicy:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_unset_role_starts_consumers_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default (unset) = old deployment behaviour: API process consumes too."""
        stubs = _stub_main_lifespan(monkeypatch)
        monkeypatch.setattr(settings.worker, "WORKER_ROLE", None)

        async with main_module.lifespan(MagicMock()):
            stubs["executor"].start.assert_awaited_once()

        stubs["executor"].stop.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_api_role_is_enqueue_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WORKER_ROLE=api: no consumers start; resources still close cleanly."""
        stubs = _stub_main_lifespan(monkeypatch)
        monkeypatch.setattr(settings.worker, "WORKER_ROLE", "api")

        async with main_module.lifespan(MagicMock()):
            stubs["executor"].start.assert_not_awaited()
            stubs["lease_reaper"].start.assert_awaited_once()

        stubs["executor"].stop.assert_not_awaited()
        stubs["redis_manager"].close.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_worker_role_fails_api_boot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stubs = _stub_main_lifespan(monkeypatch)
        monkeypatch.setattr(settings.worker, "WORKER_ROLE", "worker")

        with pytest.raises(ProcessRoleError):
            async with main_module.lifespan(MagicMock()):
                pass

        stubs["executor"].start.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_local_mode_still_executes_in_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No Redis → no queue exists; LocalExecutor in the API process is correct."""
        stubs = _stub_main_lifespan(monkeypatch)
        monkeypatch.setattr(settings.redis, "REDIS_BROKER_ENABLED", False)

        async with main_module.lifespan(MagicMock()):
            stubs["executor"].start.assert_awaited_once()


# ------------------------------------------------------------------
# Standalone worker process
# ------------------------------------------------------------------


class TestWorkerEntrypoint:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_worker_lifespan_starts_consumer_stack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stubs = _stub_worker_lifespan(monkeypatch)
        monkeypatch.setattr(settings.worker, "WORKER_ROLE", "worker")
        monkeypatch.setattr(settings.cron, "CRON_ENABLED", True)

        async with worker_module.worker_lifespan():
            pass

        stubs["executor"].start.assert_awaited_once()
        stubs["lease_reaper"].start.assert_awaited_once()
        stubs["broker_manager"].start.assert_awaited_once()
        stubs["cron_scheduler"].start.assert_awaited_once()
        # Reverse-order shutdown: consumers stopped before Redis/DB close.
        stubs["executor"].stop.assert_awaited_once()
        stubs["db_manager"].close.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_worker_startup_failure_unwinds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Redis failure after DB start must still close the DB pool; no consumers."""
        stubs = _stub_worker_lifespan(monkeypatch)
        monkeypatch.setattr(settings.worker, "WORKER_ROLE", "worker")
        stubs["redis_manager"].initialize.side_effect = ConnectionError("redis down")

        with pytest.raises(ConnectionError):
            async with worker_module.worker_lifespan():
                pass

        stubs["db_manager"].close.assert_awaited_once()
        stubs["executor"].start.assert_not_awaited()

    def test_worker_main_rejects_api_role(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`python -m aegra_api.worker` with WORKER_ROLE=api exits non-zero before startup."""
        monkeypatch.setattr(settings.worker, "WORKER_ROLE", "api")
        with pytest.raises(SystemExit) as exc:
            worker_module.main()
        assert exc.value.code == 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_worker_stays_alive_until_cancelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A healthy worker parks until SIGTERM/SIGINT cancels it."""
        _stub_worker_lifespan(monkeypatch)
        monkeypatch.setattr(settings.worker, "WORKER_ROLE", "worker")
        monkeypatch.setattr(settings.cron, "CRON_ENABLED", False)

        task = asyncio.create_task(worker_module.run_worker())
        await asyncio.sleep(0.01)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ------------------------------------------------------------------
# Persistence policy (trusted deployment gate)
# ------------------------------------------------------------------


def _static_service(graph_id: str = "g"):
    from aegra_api.services.langgraph_service import LangGraphService

    service = LangGraphService()
    service._base_graph_cache[graph_id] = MagicMock()
    service._graph_registry[graph_id] = {"file_path": "x.py", "export_name": graph_id}
    return service


class TestPersistencePolicy:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_required_persistence_refuses_injection_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AEGRA_PERSISTENCE_REQUIRED=true → no unpersisted execution."""
        monkeypatch.setattr(settings.checkpointer, "AEGRA_PERSISTENCE_REQUIRED", True)
        service = _static_service()
        base_graph = MagicMock()
        base_graph.config = {"metadata": {}}
        base_graph.copy.side_effect = TypeError("no checkpointer slot")
        service._base_graph_cache["g"] = base_graph

        with _memory_persistence():
            with pytest.raises(RuntimeError, match="without persistence"):
                async with service.get_graph("g"):
                    pass

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_legacy_default_allows_unpersisted_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default (legacy) policy keeps running graphs that cannot be injected."""
        monkeypatch.setattr(settings.checkpointer, "AEGRA_PERSISTENCE_REQUIRED", False)
        service = _static_service()
        base_graph = MagicMock()
        base_graph.config = {"metadata": {}}
        base_graph.copy.side_effect = TypeError("no checkpointer slot")
        service._base_graph_cache["g"] = base_graph

        with _memory_persistence():
            async with service.get_graph("g") as graph:
                assert graph is base_graph

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_required_persistence_refuses_missing_saver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Persistence-required mode fails when the saver is unavailable."""
        monkeypatch.setattr(settings.checkpointer, "AEGRA_PERSISTENCE_REQUIRED", True)
        service = _static_service()

        with _memory_persistence(missing_checkpointer=True):
            with pytest.raises(RuntimeError, match="Persistence is required"):
                async with service.get_graph("g"):
                    pass

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_factory_path_uses_same_fail_closed_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Factory graphs cannot bypass the trusted persistence gate."""
        monkeypatch.setattr(settings.checkpointer, "AEGRA_PERSISTENCE_REQUIRED", True)
        service = _static_service("f")
        service._graph_factories["f"] = lambda: None

        unpersisted = MagicMock()
        unpersisted.copy.side_effect = TypeError("no checkpointer slot")

        with (
            _memory_persistence(),
            patch("aegra_api.services.langgraph_service.invoke_factory", MagicMock(return_value=unpersisted)),
            patch(
                "aegra_api.services.langgraph_service.generate_graph",
                MagicMock(side_effect=lambda result, _graph_id: _passthrough(result)),
            ),
            patch.object(service, "_graph_registry", {"f": {"file_path": "x.py", "export_name": "f"}}),
        ):
            with pytest.raises(RuntimeError, match="without persistence"):
                async with service.get_graph("f"):
                    pass


def _memory_persistence(*, missing_checkpointer: bool = False):
    """Patch the DB getter seam with *real* in-memory persistence objects.

    ``missing_checkpointer`` simulates a half-initialized persistence stack;
    the store stays a real ``InMemoryStore``.
    """
    from contextlib import ExitStack

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    saver = None if missing_checkpointer else InMemorySaver()
    stack = ExitStack()
    stack.enter_context(patch("aegra_api.core.database.db_manager.get_checkpointer", MagicMock(return_value=saver)))
    stack.enter_context(
        patch("aegra_api.core.database.db_manager.get_store", MagicMock(return_value=InMemoryStore()))
    )
    return stack


def _passthrough(graph):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _ctx():
        yield graph

    return _ctx()


# ------------------------------------------------------------------
# Identity override
# ------------------------------------------------------------------


class TestIdentityOverride:
    def test_authenticated_identity_wins_over_client_user_id(self) -> None:
        user = MagicMock()
        user.identity = "auth-user-1"
        user.display_name = "Auth"

        config = inject_user_context(user, {"configurable": {"user_id": "attacker", "team": "t1"}})
        assert config["configurable"]["user_id"] == "auth-user-1"
        # Non-identity custom client fields survive.
        assert config["configurable"]["team"] == "t1"
        assert config["configurable"]["langgraph_auth_user"] is user

    def test_no_user_keeps_client_fields(self) -> None:
        config = inject_user_context(None, {"configurable": {"user_id": "anon-provided"}})
        # Without an authenticated subject there is no trusted identity to
        # inject; the client value is not upgraded into one.
        assert config["configurable"]["user_id"] == "anon-provided"
        assert "langgraph_auth_user" not in config["configurable"]

    def test_shared_base_config_does_not_leak_identity_across_users(self) -> None:
        """One base config building A then B must keep A=A, B=B, base untouched."""
        base_config = {"configurable": {"thread_id": "t-1", "team": "t1"}}
        user_a = MagicMock()
        user_a.identity = "user-a"
        user_a.display_name = "A"
        user_b = MagicMock()
        user_b.identity = "user-b"
        user_b.display_name = "B"

        config_a = inject_user_context(user_a, base_config)
        config_b = inject_user_context(user_b, base_config)

        assert config_a["configurable"]["user_id"] == "user-a"
        assert config_b["configurable"]["user_id"] == "user-b"
        assert config_a["configurable"]["langgraph_auth_user"] is user_a
        assert config_b["configurable"]["langgraph_auth_user"] is user_b
        # Neither call mutates the caller's reusable template.
        assert base_config == {"configurable": {"thread_id": "t-1", "team": "t1"}}

    def test_missing_display_name_falls_back_to_trusted_identity(self) -> None:
        """A user object without display_name still yields the authenticated id."""
        user = SimpleNamespace(identity="user-c")

        config = inject_user_context(user, {"configurable": {}})

        assert config["configurable"]["user_id"] == "user-c"
        assert config["configurable"]["user_display_name"] == "user-c"
        assert config["configurable"]["langgraph_auth_user"] is user


# ------------------------------------------------------------------
# Factory reload through the public get_graph consumer path
# ------------------------------------------------------------------

_RELOAD_PROBE_GRAPH_ID = "consumer-factory-reload-probe"


class TestFactoryConsumerPath:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalidated_factory_reloads_once_and_runs_with_trusted_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Public get_graph: after invalidation the real factory is loaded once,
        invoked once with the authenticated runtime, and yields an executable graph.

        The factory raises if it is ever invoked without an authenticated user,
        so an anonymous pre-execution (e.g. schema defaults) fails this test
        instead of silently succeeding.
        """
        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.store.memory import InMemoryStore
        from langgraph_sdk.runtime import ServerRuntime

        from aegra_api.models.auth import User
        from aegra_api.services.graph_factory import classify_factory

        monkeypatch.setattr(settings.checkpointer, "AEGRA_PERSISTENCE_REQUIRED", True)
        service = LangGraphService()
        registry_entry = {"file_path": "probe.py", "export_name": "make_graph"}
        service._graph_registry[_RELOAD_PROBE_GRAPH_ID] = registry_entry

        factory_runtimes: list[ServerRuntime] = []

        def make_graph(runtime: ServerRuntime) -> StateGraph:
            factory_runtimes.append(runtime)
            if runtime.user is None:
                raise AssertionError("factory invoked without an authenticated user")
            trusted_identity = runtime.user.identity

            def noop(state: MessagesState) -> dict:
                return {"messages": [AIMessage(content=f"noop:{trusted_identity}")]}

            builder = StateGraph(MessagesState)
            builder.add_node("noop", noop)
            builder.add_edge(START, "noop")
            builder.add_edge("noop", END)
            return builder

        loader_calls: list[str] = []

        async def fake_load(graph_id: str, info: dict[str, str]) -> None:
            loader_calls.append(graph_id)
            classify_factory(make_graph, graph_id)
            service._graph_factories[graph_id] = make_graph

        monkeypatch.setattr(service, "_load_graph_from_file", fake_load)
        user = User(identity="trusted-user")

        with (
            patch("aegra_api.core.database.db_manager.get_checkpointer", MagicMock(return_value=InMemorySaver())),
            patch("aegra_api.core.database.db_manager.get_store", MagicMock(return_value=InMemoryStore())),
        ):
            async with service.get_graph(
                _RELOAD_PROBE_GRAPH_ID,
                user=user,
                access_context="threads.create_run",
            ) as graph:
                # Observable behavior, not a type check: the noop node appends
                # a marker message proving the trusted identity reached the graph.
                final_state = await graph.ainvoke(
                    {"messages": [HumanMessage(content="ping")]},
                    {"configurable": {"thread_id": "reload-probe-thread"}},
                )
                assert final_state["messages"][-1].content == "noop:trusted-user"

        # Exactly one module reload, and the real factory ran exactly once.
        assert loader_calls == [_RELOAD_PROBE_GRAPH_ID]
        assert len(factory_runtimes) == 1
        # The trusted subject reached the factory (no anonymous invocation).
        assert factory_runtimes[0].user is user
        assert factory_runtimes[0].user.identity == "trusted-user"
