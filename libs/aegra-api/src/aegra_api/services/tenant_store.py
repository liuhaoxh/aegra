"""Tenant-scoped store view for graph runtimes.

The HTTP ``/store`` endpoints confine a principal to ``["users", tenant, owner]``
(see ``api/store.py``). Graphs receive ``ServerRuntime.store``; handing them the
raw ``BaseStore`` would let a graph read or write any namespace and disagree with
what the same principal sees over HTTP.

``TenantScopedStore`` applies the identical prefix to every entry point of the
locked LangGraph ``BaseStore`` API (``langgraph.store.base``), including the
native ``GetOp`` / ``PutOp`` / ``SearchOp`` / ``ListNamespacesOp`` NamedTuples
used by ``batch``/``abatch``. Those are rewritten with ``_replace`` and handed to
the inner store in ONE call, so batching, ordering, TTL and refresh semantics are
the store's own rather than re-implemented here.

Isolation rules:

* the scope comes from the trusted principal only — a client-supplied namespace
  is nested underneath it and can never address another tenant;
* caller wildcards are KEPT: LangGraph matches prefix/suffix conditions
  element-wise against the full namespace and ANDs all conditions, so
  ``prefix=("own", "*")`` matches inside the caller's subtree while the fixed
  ``(users, tenant, owner)`` condition still pins the first three elements.
  Blanket-rejecting ``*`` would break a documented LangGraph feature for no
  security gain;
* only the trusted scope components are validated: a ``*`` in ``tenant`` or
  ``owner`` (or a control character that could alias a namespace label) is
  refused, because that would turn the injected prefix itself into a pattern;
* listing always carries the scope: with no caller prefix it lists exactly the
  caller's subtree; ``max_depth`` keeps the native root-absolute meaning (the
  scope occupies the first three elements).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

#: Must match ``api.store._USER_SCOPE_PREFIX``.
USER_SCOPE_PREFIX = "users"

_WILDCARD = "*"

#: Characters that would alias distinct namespace labels once a backend joins
#: them into a single string (Postgres/SQLite store the path as an array, but a
#: separator character inside a label is never a legitimate identity).
_FORBIDDEN_SCOPE_CHARS = ("\x1f", "\x00", "\n", "\r", "\t")


class NamespaceEscape(ValueError):
    """Raised when the trusted scope itself cannot be used as a fixed prefix."""


def _check_scope_label(label: str, what: str) -> None:
    """Validate one trusted scope component (never a caller namespace)."""
    text = str(label)
    if _WILDCARD in text:
        raise NamespaceEscape(f"{what} must not contain a wildcard: {text!r}")
    if any(ch in text for ch in _FORBIDDEN_SCOPE_CHARS):
        raise NamespaceEscape(f"{what} contains a character that cannot be a namespace label: {text!r}")


def scoped_namespace(
    tenant_id: str, owner_id: str, namespace: tuple[str, ...] | list[str] | None
) -> tuple[str, ...]:
    """Prefix ``namespace`` with the trusted scope, idempotently."""
    head = scope_path(tenant_id, owner_id)
    ns = tuple(namespace or ())
    if ns[: len(head)] == head:
        return ns
    return (*head, *ns)


def scope_path(tenant_id: str, owner_id: str) -> tuple[str, ...]:
    """The trusted prefix; both components are validated, not just namespaces.

    A ``*`` (or any wildcard-bearing value) in ``tenant_id`` or ``owner_id``
    would turn the trusted prefix itself into a pattern and expose every
    tenant's data — validated here, at the one place a scope is built.
    """
    _check_scope_label(tenant_id, "tenant")
    _check_scope_label(owner_id, "owner")
    return (USER_SCOPE_PREFIX, tenant_id, owner_id)


class TenantScopedStore(BaseStore):
    """``BaseStore`` view fixed to one trusted ``(tenant, owner)`` scope."""

    def __init__(self, inner: BaseStore, tenant_id: str, owner_id: str) -> None:
        self._scope = scope_path(tenant_id, owner_id)  # fail-closed on wildcards
        self._inner = inner
        self._tenant_id = tenant_id
        self._owner_id = owner_id

    # -- helpers ---------------------------------------------------------
    def _ns(self, namespace: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
        return scoped_namespace(self._tenant_id, self._owner_id, namespace)

    def _scoped_conditions(
        self, conditions: tuple[MatchCondition, ...] | None
    ) -> tuple[MatchCondition, ...]:
        """Pin the trusted scope and keep the caller's conditions working.

        LangGraph's prefix matcher compares from the START of the namespace, so a
        caller condition such as ``prefix=("own", "*")`` only matches if it is
        rewritten to ``(users, tenant, owner, "own", "*")`` — merely ANDing a
        scope condition would make the caller's own prefix compare against
        ``"users"`` and return nothing (the batch path diverging from
        ``list_namespaces``). Suffix conditions are anchored at the END, so they
        stay as-is; the injected scope condition already pins the start.
        """
        base = (MatchCondition(match_type="prefix", path=self._scope),)
        rewritten: list[MatchCondition] = []
        for condition in conditions or ():
            if condition.match_type == "prefix":
                path = tuple(condition.path)
                if path[: len(self._scope)] == self._scope:
                    rewritten.append(condition)
                else:
                    rewritten.append(MatchCondition(match_type="prefix", path=(*self._scope, *path)))
            else:
                rewritten.append(condition)
        return base + tuple(rewritten)

    @property
    def supports_ttl(self) -> bool:  # type: ignore[override]
        return getattr(self._inner, "supports_ttl", False)

    # -- single operations ----------------------------------------------
    def get(self, namespace, key, *, refresh_ttl: bool | None = None) -> Item | None:
        return self._inner.get(self._ns(namespace), key, refresh_ttl=refresh_ttl)

    def put(self, namespace, key, value, index=None, *, ttl=NOT_PROVIDED) -> None:
        kwargs: dict[str, Any] = {} if ttl is NOT_PROVIDED else {"ttl": ttl}
        return self._inner.put(self._ns(namespace), key, value, index, **kwargs)

    def delete(self, namespace, key) -> None:
        return self._inner.delete(self._ns(namespace), key)

    def search(
        self,
        namespace_prefix,
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        return self._inner.search(
            self._ns(namespace_prefix),
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )

    def list_namespaces(
        self,
        *,
        prefix=None,
        suffix=None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return self._inner.list_namespaces(
            prefix=self._ns(prefix),
            suffix=suffix,
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )

    # -- async single operations ---------------------------------------
    async def aget(self, namespace, key, *, refresh_ttl: bool | None = None) -> Item | None:
        return await self._inner.aget(self._ns(namespace), key, refresh_ttl=refresh_ttl)

    async def aput(self, namespace, key, value, index=None, *, ttl=NOT_PROVIDED) -> None:
        kwargs: dict[str, Any] = {} if ttl is NOT_PROVIDED else {"ttl": ttl}
        return await self._inner.aput(self._ns(namespace), key, value, index, **kwargs)

    async def adelete(self, namespace, key) -> None:
        return await self._inner.adelete(self._ns(namespace), key)

    async def asearch(
        self,
        namespace_prefix,
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        return await self._inner.asearch(
            self._ns(namespace_prefix),
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )

    async def alist_namespaces(
        self,
        *,
        prefix=None,
        suffix=None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return await self._inner.alist_namespaces(
            prefix=self._ns(prefix),
            suffix=suffix,
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )

    # -- batch: rewrite the native ops, delegate once ------------------
    def _scope_op(self, op: Op) -> Op:
        if isinstance(op, GetOp):
            return op._replace(namespace=self._ns(op.namespace))
        if isinstance(op, PutOp):
            return op._replace(namespace=self._ns(op.namespace))
        if isinstance(op, SearchOp):
            return op._replace(namespace_prefix=self._ns(op.namespace_prefix))
        if isinstance(op, ListNamespacesOp):
            return op._replace(match_conditions=self._scoped_conditions(op.match_conditions))
        raise ValueError(f"Unsupported store operation for scoped view: {type(op).__name__}")

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return self._inner.batch([self._scope_op(op) for op in ops])

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await self._inner.abatch([self._scope_op(op) for op in ops])
