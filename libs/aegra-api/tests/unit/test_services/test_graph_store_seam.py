"""Graph↔store/checkpointer seam isolation tests (real compiled Pregel).

Covers the two injection paths that bypass ``ServerRuntime.store``:

* the factory path compiles a graph and previously re-injected the RAW store via
  ``graph_obj.copy(update={"store": store})`` — a node using LangGraph's runtime
  store could therefore reach another tenant's data and disagree with what
  ``/store`` returns for the same principal;
* ``Pregel.copy`` replaces attributes wholesale, so a graph that arrives with its
  own checkpointer (FollowMe boundary saver) must keep it — the fork must never
  overwrite a business saver with the deployment one.

Assertions use the REAL compiled-graph surface: a ``CompiledStateGraph`` exposes
``.store`` / ``.checkpointer`` at the top level (``.config`` is ``None``), so the
tests read those attributes and drive an actual node execution plus real state
read/write through ``aupdate_state`` / ``aget_state``. The only stand-ins are the
graph's own IO sinks (``InMemoryStore``/``InMemorySaver``), never the seam.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from aegra_api.models.auth import User
from aegra_api.services.langgraph_service import _attach_persistence
from aegra_api.services.tenant_store import TenantScopedStore, scoped_namespace

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
OWNER = "owner-shared"


class State(TypedDict):
    messages: Annotated[list[str], add]


def _minimal_graph() -> Pregel:
    """A real compiled Pregel whose node can reach ``runtime.store``."""

    async def node(state: State, runtime: Runtime) -> dict[str, Any]:
        store = runtime.store
        assert store is not None, "node executed without an injected store"
        await store.aput(("injected",), "marker", {"seen": True})
        item = await store.aget(("injected",), "marker")
        return {"messages": [f"store:{item is not None}"]}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.set_entry_point("node")
    builder.add_edge("node", "__end__")
    return builder.compile()


@pytest.mark.asyncio
async def test_injected_store_is_scoped_to_trusted_principal() -> None:
    """The graph's top-level ``store`` is the tenant-scoped view."""
    raw = InMemoryStore()
    graph = _minimal_graph()
    user = User(identity=OWNER, tenant_id=TENANT_A)

    attached = _attach_persistence(graph, None, raw, user, "g")
    # Real CompiledStateGraph: injected values live on top-level attributes,
    # there is no ``.config`` payload carrying them.
    assert getattr(attached, "config", None) is None
    assert isinstance(attached.store, TenantScopedStore)


@pytest.mark.asyncio
async def test_node_execution_writes_only_inside_caller_scope() -> None:
    """Execute the real graph, then verify the physical namespace it wrote.

    The node calls ``store.aput(("injected",), ...)`` through the injected view;
    the item must land under the trusted scope in the raw store, never at the
    raw ``("injected",)`` path, and tenant B must not see it.
    """
    raw = InMemoryStore()
    graph = _minimal_graph()
    user_a = User(identity=OWNER, tenant_id=TENANT_A)
    attached = _attach_persistence(graph, None, raw, user_a, "g")

    result = await attached.ainvoke({"messages": []})
    assert result["messages"][0] == "store:True", result

    # 1. raw path untouched
    assert await raw.aget(("injected",), "marker") is None
    # 2. correctly attributed physical namespace
    scoped = await raw.aget(scoped_namespace(TENANT_A, OWNER, ("injected",)), "marker")
    assert scoped is not None and scoped.value == {"seen": True}
    # 3. the same principal reading through its view sees it
    assert (await attached.store.aget(("injected",), "marker")).value == {"seen": True}
    # 4. another tenant (same owner identity) does not
    other = TenantScopedStore(raw, TENANT_B, OWNER)
    assert await other.aget(("injected",), "marker") is None


@pytest.mark.asyncio
async def test_two_tenants_do_not_share_graph_store_data() -> None:
    raw = InMemoryStore()
    a = _attach_persistence(_minimal_graph(), None, raw, User(identity=OWNER, tenant_id=TENANT_A), "g")
    b = _attach_persistence(_minimal_graph(), None, raw, User(identity=OWNER, tenant_id=TENANT_B), "g")

    await a.store.aput(("shared",), "k", {"v": "a"})
    assert await b.store.aget(("shared",), "k") is None
    await b.store.aput(("shared",), "k", {"v": "b"})
    assert (await a.store.aget(("shared",), "k")).value == {"v": "a"}


@pytest.mark.asyncio
async def test_no_principal_means_no_raw_store_fail_closed() -> None:
    """``user=None`` must not receive the raw store: execution and reads both
    need authorization."""
    raw = InMemoryStore()
    with pytest.raises(Exception) as exc:
        _attach_persistence(_minimal_graph(), None, raw, None, "g")
    assert "authenticated principal" in str(exc.value)


@pytest.mark.asyncio
async def test_graph_supplied_checkpointer_is_preserved() -> None:
    """``Pregel.copy`` replaces attributes wholesale: a business saver must
    survive persistence attachment untouched."""
    raw = InMemoryStore()
    business_saver = InMemorySaver()  # stand-in for a boundary-aware saver
    graph_with_saver = _minimal_graph().copy(update={"checkpointer": business_saver})

    attached = _attach_persistence(
        graph_with_saver, InMemorySaver(), raw, User(identity=OWNER, tenant_id=TENANT_A), "g"
    )

    assert attached.checkpointer is business_saver, (
        "the graph's own checkpointer was clobbered by the deployment saver"
    )
    assert isinstance(attached.store, TenantScopedStore)


@pytest.mark.asyncio
async def test_deployment_checkpointer_attaches_when_graph_has_none() -> None:
    raw = InMemoryStore()
    deployment_saver = InMemorySaver()
    attached = _attach_persistence(
        _minimal_graph(), deployment_saver, raw, User(identity=OWNER, tenant_id=TENANT_A), "g"
    )
    assert attached.checkpointer is deployment_saver


@pytest.mark.asyncio
async def test_state_write_and_read_use_the_attached_saver() -> None:
    """Real state seam: ``aupdate_state`` then ``aget_state`` through the saver
    that persistence attachment installed."""
    raw = InMemoryStore()
    saver = InMemorySaver()
    attached = _attach_persistence(
        _minimal_graph(), saver, raw, User(identity=OWNER, tenant_id=TENANT_A), "g"
    )
    cfg = {"configurable": {"thread_id": "thread-state-1"}}

    await attached.aupdate_state(cfg, {"messages": ["from-state"]})
    snapshot = await attached.aget_state(cfg)
    assert snapshot.values["messages"] == ["from-state"], snapshot.values

    # The same thread id read directly off the saver proves the physical binding.
    checkpoint_tuple = await saver.aget_tuple({"configurable": {"thread_id": "thread-state-1"}})
    assert checkpoint_tuple is not None


@pytest.mark.asyncio
async def test_state_read_does_not_double_scope_namespaces() -> None:
    """``scoped_namespace`` is idempotent: handing an already-scoped namespace to
    the scoped store must not prefix it twice (API contract)."""
    raw = InMemoryStore()
    attached = _attach_persistence(
        _minimal_graph(), None, raw, User(identity=OWNER, tenant_id=TENANT_A), "g"
    )
    already_scoped = scoped_namespace(TENANT_A, OWNER, ("state",))
    await attached.store.aput(already_scoped, "thread-state", {"ok": 1})

    assert (await raw.aget((*scoped_namespace(TENANT_A, OWNER, ()), "state"), "thread-state")) is not None
    doubled = (*already_scoped, *already_scoped)
    assert await raw.aget(doubled, "thread-state") is None, "namespace was scoped twice"
    assert await raw.aget(("state",), "thread-state") is None


# ---------------------------------------------------------------------------
# Restricted/proxy graphs: persistence is read statically, never by triggering
# their __getattr__ (which raises a business error), and never by assuming
# "no saver" and overwriting one.
# ---------------------------------------------------------------------------


class BusinessGraphError(RuntimeError):
    """Stand-in for the root-side error a restricted graph raises."""


class RestrictedGraph:
    """Proxy that forbids attribute access, like the root-side restricted graph.

    The seam is a real ``property`` (as on the root side), so the static probe
    sees a descriptor and the value is materialised by normal attribute access.
    """

    def __init__(self, *, seam_saver: Any = None, raise_on_seam: bool = False) -> None:
        self._seam_saver = seam_saver
        self._raise_on_seam = raise_on_seam
        self.copied_with: dict[str, Any] | None = None

    @property
    def readonly_checkpointer(self) -> Any:
        if self._raise_on_seam:
            raise BusinessGraphError("readonly_checkpointer evaluation refused")
        return self._seam_saver

    def __getattr__(self, name: str) -> Any:
        raise BusinessGraphError(f"access to {name!r} is not allowed on a restricted graph")

    def copy(self, update: dict[str, Any] | None = None) -> RestrictedGraph:
        clone = RestrictedGraph(seam_saver=self._seam_saver, raise_on_seam=self._raise_on_seam)
        clone.copied_with = dict(update or {})
        return clone


@pytest.mark.asyncio
async def test_restricted_graph_property_seam_saver_is_preserved() -> None:
    """The property's real value (a saver) is preserved, not the descriptor."""
    raw = InMemoryStore()
    business_saver = InMemorySaver()
    graph = RestrictedGraph(seam_saver=business_saver)

    attached = _attach_persistence(
        graph, InMemorySaver(), raw, User(identity=OWNER, tenant_id=TENANT_A), "restricted"
    )

    assert attached.copied_with is not None
    assert "checkpointer" not in attached.copied_with, (
        "the boundary saver must be preserved, not replaced by the deployment saver"
    )
    assert isinstance(attached.copied_with["store"], TenantScopedStore)


@pytest.mark.asyncio
async def test_restricted_graph_property_that_raises_fails_closed() -> None:
    """A property that exists but raises is fail-closed, not 'absent'."""
    with pytest.raises(RuntimeError) as exc:
        _attach_persistence(
            RestrictedGraph(seam_saver=InMemorySaver(), raise_on_seam=True),
            InMemorySaver(),
            InMemoryStore(),
            User(identity=OWNER, tenant_id=TENANT_A),
            "restricted",
        )
    assert "persistence seam is unusable" in str(exc.value)
    assert "readonly_checkpointer" in str(exc.value)


@pytest.mark.asyncio
async def test_restricted_graph_property_without_saver_fails_closed() -> None:
    """A property whose value carries no saver is unusable, not 'no saver'."""

    class NoSaverSeamGraph(RestrictedGraph):
        @property
        def readonly_checkpointer(self) -> Any:
            return object()  # exists, but exposes no checkpointer

    with pytest.raises(RuntimeError) as exc:
        _attach_persistence(
            NoSaverSeamGraph(),
            InMemorySaver(),
            InMemoryStore(),
            User(identity=OWNER, tenant_id=TENANT_A),
            "g",
        )
    assert "persistence seam is unusable" in str(exc.value)


@pytest.mark.asyncio
async def test_restricted_graph_without_seam_fails_closed() -> None:
    class NoSeamGraph:
        def __getattr__(self, name: str) -> Any:
            raise BusinessGraphError(name)

    with pytest.raises(RuntimeError) as exc:
        _attach_persistence(
            NoSeamGraph(),
            InMemorySaver(),
            InMemoryStore(),
            User(identity=OWNER, tenant_id=TENANT_A),
            "restricted",
        )
    assert "no persistence seam" in str(exc.value)
    assert not isinstance(exc.value, BusinessGraphError)
