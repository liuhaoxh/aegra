"""End-to-end store isolation against the REAL InMemoryStore.

No fakes: these tests build an actual ``langgraph.store.memory.InMemoryStore``,
populate two tenants and two owners, wrap it in ``TenantScopedStore`` for each
principal, and exercise the locked ``BaseStore`` surface — single ops, the native
``GetOp/PutOp/SearchOp/ListNamespacesOp`` NamedTuples through ``batch``/``abatch``,
empty/prefix/suffix listings, max_depth, pagination, TTL parameters, and wildcard
refusal — in sync and async forms.

Backend semantics this file respects (verified against the locked implementation):

* ``list_namespaces(prefix=...)`` matches by prefix, so a parent prefix returns
  the parent AND its descendants;
* ``max_depth`` truncates namespaces counted from the store root, not from the
  caller's prefix — with the trusted scope occupying the first three elements;
* ``batch`` does not promise that a write in a batch is visible to a read in the
  same batch (the backend reads before writing), so write batches and read
  batches are separate calls, and the wrapper's results are compared against a
  native ``inner.batch`` of the equivalent explicitly-scoped operations;
* ``InMemoryStore`` does not support TTL; both the raw store and the wrapper must
  fail the same way, and real TTL behaviour belongs to the isolated
  Redis/Postgres backends, not to this test.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langgraph.store.base import GetOp, ListNamespacesOp, PutOp, SearchOp
from langgraph.store.memory import InMemoryStore

from aegra_api.services.tenant_store import (
    NamespaceEscape,
    TenantScopedStore,
    scoped_namespace,
)

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
OWNER = "owner-shared"  # same identity in both tenants: the cross-tenant case
OWNER2 = "owner-2"

SCOPE_A = ("users", TENANT_A, OWNER)
SCOPE_B = ("users", TENANT_B, OWNER)
SCOPE_A2 = ("users", TENANT_A, OWNER2)


def _view(inner: InMemoryStore, tenant: str, owner: str) -> TenantScopedStore:
    return TenantScopedStore(inner, tenant, owner)


async def _seed(inner: InMemoryStore) -> None:
    """Private data for tenant B and for another owner, put in the raw store."""
    await inner.aput((*SCOPE_B, "private"), "secret", {"v": "b-secret"})
    await inner.aput((*SCOPE_A, "own"), "mine", {"v": "a-mine"})
    await inner.aput((*SCOPE_A2, "own"), "other-owner", {"v": "a-other"})
    await inner.aput((*SCOPE_A, "own", "nested"), "deep", {"v": "deep"})


# ---------------------------------------------------------------------------
# Namespace algebra
# ---------------------------------------------------------------------------


def test_scoped_namespace_is_idempotent_and_nesting() -> None:
    assert scoped_namespace(TENANT_A, OWNER, ("x",)) == (*SCOPE_A, "x")
    assert scoped_namespace(TENANT_A, OWNER, (*SCOPE_A, "x")) == (*SCOPE_A, "x")
    # A client namespace shaped like a foreign scope is nested, never trusted.
    assert scoped_namespace(TENANT_A, OWNER, (*SCOPE_B, "x")) == (*SCOPE_A, *SCOPE_B, "x")


def test_scope_helpers_agree_with_http_layout() -> None:
    from aegra_api.api.store import apply_namespace_scoping
    from aegra_api.models.auth import User

    http_ns = apply_namespace_scoping(["notes"], User(identity=OWNER, tenant_id=TENANT_A))
    assert tuple(http_ns) == scoped_namespace(TENANT_A, OWNER, ("notes",))


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_namespaces_without_prefix_lists_only_own_scope() -> None:
    inner = InMemoryStore()
    await _seed(inner)
    view = _view(inner, TENANT_A, OWNER)

    found = {tuple(ns) for ns in view.list_namespaces()}
    assert found, "no namespaces returned"
    assert all(ns[:3] == SCOPE_A for ns in found), found
    assert (*SCOPE_B, "private") not in found
    assert (*SCOPE_A2, "own") not in found


@pytest.mark.asyncio
async def test_list_namespaces_prefix_includes_descendants_and_stays_scoped() -> None:
    """Prefix matching is prefix semantics: parent AND its nested children."""
    inner = InMemoryStore()
    await _seed(inner)
    view = _view(inner, TENANT_A, OWNER)

    found = {tuple(ns) for ns in view.list_namespaces(prefix=("own",))}
    assert found == {(*SCOPE_A, "own"), (*SCOPE_A, "own", "nested")}, found

    # Compare with the native call using the explicitly scoped prefix.
    native = {tuple(ns) for ns in inner.list_namespaces(prefix=(*SCOPE_A, "own"))}
    assert found == native

    # No foreign tenant/owner namespaces leak in, whatever the prefix.
    for ns in found:
        assert ns[:3] == SCOPE_A


@pytest.mark.asyncio
async def test_list_namespaces_suffix_max_depth_and_pagination() -> None:
    inner = InMemoryStore()
    await _seed(inner)
    view = _view(inner, TENANT_A, OWNER)

    # suffix: only namespaces ending with 'nested', inside our scope
    found = {tuple(ns) for ns in view.list_namespaces(suffix=("nested",))}
    assert found == {(*SCOPE_A, "own", "nested")}
    assert found == {tuple(ns) for ns in inner.list_namespaces(suffix=("nested",))} & {
        (*SCOPE_A, "own", "nested")
    }

    # max_depth truncates from the store ROOT (the scope occupies 3 elements),
    # exactly as the native store does — the wrapper does not shift it.
    shallow = {tuple(ns) for ns in view.list_namespaces(max_depth=3)}
    assert shallow == {SCOPE_A}, shallow
    assert shallow == {tuple(ns) for ns in inner.list_namespaces(max_depth=3)} & {SCOPE_A}
    deeper = {tuple(ns) for ns in view.list_namespaces(max_depth=4)}
    assert deeper == {(*SCOPE_A, "own")}, deeper

    # pagination applies to the scoped result set, in sorted order
    all_ns = view.list_namespaces()
    assert view.list_namespaces(limit=1, offset=0) == all_ns[:1]
    assert view.list_namespaces(limit=100, offset=1) == all_ns[1:]


@pytest.mark.asyncio
async def test_list_namespaces_multiple_match_conditions_are_anded() -> None:
    """A caller condition that is not under our scope yields nothing."""
    from langgraph.store.base import MatchCondition

    inner = InMemoryStore()
    await _seed(inner)
    view = _view(inner, TENANT_A, OWNER)

    foreign = (
        ListNamespacesOp(
            match_conditions=(MatchCondition(match_type="prefix", path=(*SCOPE_B,)),),
            max_depth=None,
            limit=100,
            offset=0,
        ),
    )
    assert view.batch(foreign) == [[]]

    # A suffix condition combined with the trusted prefix stays scoped.
    suffix_ops = (
        ListNamespacesOp(
            match_conditions=(MatchCondition(match_type="suffix", path=("nested",)),),
            max_depth=None,
            limit=100,
            offset=0,
        ),
    )
    result = view.batch(suffix_ops)[0]
    assert {tuple(ns) for ns in result} == {(*SCOPE_A, "own", "nested")}


# ---------------------------------------------------------------------------
# Single-item round trips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_put_delete_search_roundtrip_matches_http_layout() -> None:
    inner = InMemoryStore()
    await _seed(inner)
    view = _view(inner, TENANT_A, OWNER)

    view.put(("notes",), "k", {"v": 1})
    item = view.get(("notes",), "k")
    assert item is not None and item.value == {"v": 1}
    # The raw store layout is exactly what /store writes for the same principal.
    raw = await inner.aget((*SCOPE_A, "notes"), "k")
    assert raw is not None and raw.value == {"v": 1}

    results = view.search(("notes",), query="k")
    assert any(r.key == "k" for r in results)

    view.delete(("notes",), "k")
    assert view.get(("notes",), "k") is None


# ---------------------------------------------------------------------------
# Native ops through batch / abatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_write_then_read_roundtrip_matches_native() -> None:
    """Writes and reads are separate batches: a batch does not order them."""
    inner = InMemoryStore()
    await _seed(inner)
    view = _view(inner, TENANT_A, OWNER)

    writes = (
        PutOp(namespace=("batch",), key="one", value={"n": 1}),
        PutOp(namespace=("batch",), key="two", value={"n": 2}),
    )
    assert view.batch(writes) == [None, None]

    reads: tuple[Any, ...] = (
        GetOp(namespace=("batch",), key="two", refresh_ttl=False),
        GetOp(namespace=("batch",), key="one", refresh_ttl=False),
        GetOp(namespace=("private",), key="secret", refresh_ttl=False),  # tenant B -> miss
        SearchOp(namespace_prefix=("batch",), query=None, filter=None, limit=10, offset=0, refresh_ttl=False),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=100, offset=0),
    )
    results = view.batch(reads)

    values = [r.value["n"] if r is not None else None for r in results[:2]]
    assert values == [2, 1], "result order must follow the ops order"
    assert results[2] is None, "tenant B's private item must not be visible"
    assert {r.key for r in results[3]} == {"one", "two"}
    assert all(tuple(ns)[:3] == SCOPE_A for ns in results[4])

    # Native control: the same logical ops with explicitly scoped namespaces.
    native = inner.batch(
        (
            GetOp(namespace=(*SCOPE_A, "batch"), key="two", refresh_ttl=False),
            GetOp(namespace=(*SCOPE_A, "batch"), key="one", refresh_ttl=False),
            GetOp(namespace=(*SCOPE_A, "private"), key="secret", refresh_ttl=False),
            SearchOp(
                namespace_prefix=(*SCOPE_A, "batch"),
                query=None,
                filter=None,
                limit=10,
                offset=0,
                refresh_ttl=False,
            ),
        )
    )
    assert [r.value["n"] if r is not None else None for r in native[:2]] == [2, 1]
    assert native[2] is None
    assert {r.key for r in native[3]} == {r.key for r in results[3]}


@pytest.mark.asyncio
async def test_abatch_write_then_read_roundtrip_matches_native() -> None:
    inner = InMemoryStore()
    await _seed(inner)
    view = _view(inner, TENANT_A, OWNER)

    assert await view.abatch((PutOp(namespace=("abatch",), key="k", value={"ok": True}),)) == [None]

    results = await view.abatch(
        (
            GetOp(namespace=("private",), key="secret", refresh_ttl=False),
            GetOp(namespace=("abatch",), key="k", refresh_ttl=False),
            ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        )
    )
    assert results[0] is None
    assert results[1] is not None and results[1].value == {"ok": True}
    assert all(tuple(ns)[:3] == SCOPE_A for ns in results[2])

    native = await inner.abatch(
        (
            GetOp(namespace=(*SCOPE_A, "private"), key="secret", refresh_ttl=False),
            GetOp(namespace=(*SCOPE_A, "abatch"), key="k", refresh_ttl=False),
        )
    )
    assert native[0] is None and native[1] is not None and native[1].value == {"ok": True}


# ---------------------------------------------------------------------------
# Async single operations and two-principal isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_read_paths_isolated() -> None:
    inner = InMemoryStore()
    await _seed(inner)
    view = _view(inner, TENANT_A, OWNER)

    assert await view.aget(("private",), "secret") is None
    mine = await view.aget(("own",), "mine")
    assert mine is not None and mine.value == {"v": "a-mine"}

    other = await view.asearch(("own",))
    assert {r.key for r in other} <= {"mine", "deep"}
    assert await view.aget(("own",), "other-owner") is None

    found = await view.alist_namespaces()
    assert all(tuple(ns)[:3] == SCOPE_A for ns in found)


@pytest.mark.asyncio
async def test_two_principals_do_not_see_each_others_items() -> None:
    inner = InMemoryStore()
    await _seed(inner)
    a = _view(inner, TENANT_A, OWNER)
    b = _view(inner, TENANT_B, OWNER)
    a2 = _view(inner, TENANT_A, OWNER2)

    assert a.get(("own",), "mine") is not None
    assert b.get(("own",), "mine") is None  # same owner, other tenant
    assert a2.get(("own",), "other-owner") is not None
    assert a2.get(("own",), "mine") is None  # same tenant, other owner

    b.put(("own",), "mine", {"v": "b-overwrite-attempt"})
    assert a.get(("own",), "mine").value == {"v": "a-mine"}, "cross-tenant overwrite must not land"
    assert b.get(("own",), "mine").value == {"v": "b-overwrite-attempt"}


# ---------------------------------------------------------------------------
# TTL: the in-memory backend does not support it — wrapper and native agree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ttl_is_rejected_identically_by_wrapper_and_native_store() -> None:
    inner = InMemoryStore()
    view = _view(inner, TENANT_A, OWNER)

    with pytest.raises(NotImplementedError):
        inner.put((*SCOPE_A, "ttl"), "k", {"v": 1}, ttl=3600.0)
    with pytest.raises(NotImplementedError):
        view.put(("ttl",), "k", {"v": 1}, ttl=3600.0)

    # Without ttl the wrapper behaves like the native store.
    view.put(("ttl",), "k", {"v": 1})
    assert view.get(("ttl",), "k").value == {"v": 1}
    assert view.search(("ttl",), refresh_ttl=False).__len__() == 1


# ---------------------------------------------------------------------------
# Wildcards cannot widen the trusted prefix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_caller_wildcards_work_inside_the_scope_only() -> None:
    """LangGraph wildcards are kept: they widen within the caller's subtree.

    ``prefix=("own", "*")`` must match the caller's own children, while tenant
    B's identically shaped namespace stays invisible because the injected fixed
    ``(users, tenant, owner)`` condition is ANDed with it.
    """
    inner = InMemoryStore()
    await _seed(inner)
    view = _view(inner, TENANT_A, OWNER)

    found = {tuple(ns) for ns in view.list_namespaces(prefix=("own", "*"))}
    assert found == {(*SCOPE_A, "own", "nested")}, found

    # Native control: the same condition, explicitly scoped, returns the same set.
    native = {tuple(ns) for ns in inner.list_namespaces(prefix=(*SCOPE_A, "own", "*"))}
    assert found == native

    # A wildcard cannot reach another tenant: the scope condition is exact.
    foreign = {tuple(ns) for ns in view.list_namespaces(prefix=("*",))}
    assert all(ns[:3] == SCOPE_A for ns in foreign), foreign
    assert (*SCOPE_B, "private") not in foreign

    # Suffix wildcards behave the same way.
    suffix = {tuple(ns) for ns in view.list_namespaces(suffix=("*",))}
    assert suffix and all(ns[:3] == SCOPE_A for ns in suffix), suffix


@pytest.mark.asyncio
async def test_wildcard_match_condition_via_batch_is_scoped() -> None:
    from langgraph.store.base import MatchCondition

    inner = InMemoryStore()
    await _seed(inner)
    view = _view(inner, TENANT_A, OWNER)

    ops = (
        ListNamespacesOp(
            match_conditions=(MatchCondition(match_type="prefix", path=("own", "*")),),
            max_depth=None,
            limit=100,
            offset=0,
        ),
    )
    found = {tuple(ns) for ns in view.batch(ops)[0]}
    assert found == {(*SCOPE_A, "own", "nested")}, found

    # Native control: the equivalent condition with the scope already applied.
    native = {
        tuple(ns)
        for ns in inner.list_namespaces(prefix=(*SCOPE_A, "own", "*"))
    }
    assert found == native, (found, native)

    # Async parity: abatch must behave exactly like batch.
    async_found = {tuple(ns) for ns in (await view.abatch(ops))[0]}
    assert async_found == found, (async_found, found)

    # Two conditions are ANDed: a caller prefix pointing at tenant B's shape
    # matches nothing inside our scope.
    foreign_ops = (
        ListNamespacesOp(
            match_conditions=(
                MatchCondition(match_type="prefix", path=("users", TENANT_B, OWNER)),
            ),
            max_depth=None,
            limit=100,
            offset=0,
        ),
    )
    assert view.batch(foreign_ops) == [[]]


@pytest.mark.asyncio
async def test_scope_labels_reject_wildcard_and_control_characters() -> None:
    inner = InMemoryStore()
    await _seed(inner)
    with pytest.raises(NamespaceEscape):
        _view(inner, "*", OWNER)
    with pytest.raises(NamespaceEscape):
        _view(inner, TENANT_A, "*")
    with pytest.raises(NamespaceEscape):
        _view(inner, "tenant\x1fa", OWNER)
    # Caller namespaces are NOT validated: native wildcard semantics are kept.
    assert scoped_namespace(TENANT_A, OWNER, ("own", "*")) == (*SCOPE_A, "own", "*")


# ---------------------------------------------------------------------------
# Sync surface without a running loop
# ---------------------------------------------------------------------------


def test_sync_surface_works_inside_a_loop() -> None:
    async def scenario() -> dict[str, Any]:
        inner = InMemoryStore()
        view = _view(inner, TENANT_A, OWNER)
        view.put(("sync",), "k", {"v": "init"})
        got = view.batch([GetOp(namespace=("sync",), key="k", refresh_ttl=False)])
        return {"value": got[0].value}

    assert asyncio.run(scenario())["value"] == {"v": "init"}
