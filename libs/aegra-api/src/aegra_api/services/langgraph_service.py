"""LangGraph integration service.

Architecture:
- Base graph definitions are cached (safe, immutable)
- Each request gets a fresh graph copy with checkpointer/store injected
- Factory graphs are rebuilt per-request with the appropriate ServerRuntime
- Thread-safe by design without locks
"""

import asyncio
import inspect
import copy
import importlib.util
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NotRequired, TypedDict, TypeVar, cast
from uuid import uuid5

import structlog
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.runnables import RunnableConfig
from fastapi import HTTPException
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langgraph.pregel import Pregel
from langgraph_sdk.auth.types import BaseUser
from sqlalchemy import func, select

from aegra_api.constants import ASSISTANT_NAMESPACE_UUID
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import AssistantVersion as AssistantVersionORM
from aegra_api.core.orm import get_session
from aegra_api.models.auth import User
from aegra_api.observability.base import (
    get_tracing_callbacks,
    get_tracing_metadata,
)
from aegra_api.services.graph_factory import (
    AccessContext,
    build_server_runtime,
    classify_factory,
    clear_factory_registry,
    coerce_context,
    generate_graph,
    invoke_factory,
    is_factory,
)
from aegra_api.settings import settings
from aegra_api.utils.run_utils import strip_pinned_config_keys

State = TypeVar("State")
logger = structlog.get_logger(__name__)


class GraphConfigEntry(TypedDict):
    """Object form of a graph configuration entry."""

    path: str
    description: NotRequired[str | None]


class GraphRegistryEntry(TypedDict):
    """Normalized graph metadata used by graph consumers."""

    file_path: str
    export_name: str
    description: NotRequired[str]


def _module_name_for(graph_id: str) -> str:
    """Return a safe ``sys.modules`` key for a dynamically loaded graph.

    Sanitises the *graph_id* so it cannot shadow real packages (e.g. ``os.path``,
    ``json``) by replacing dots, slashes, and hyphens with underscores and
    placing the result under a private ``aegra_graphs.`` namespace.
    """
    safe_id = graph_id.replace(".", "_").replace("/", "_").replace("-", "_")
    return f"aegra_graphs.{safe_id}"


def _copy_graph_config(config: RunnableConfig | None) -> RunnableConfig | None:
    """Deep-copy a graph's config but keep its callback handlers shared."""
    if not config:
        return config
    callbacks = config.get("callbacks")
    if isinstance(callbacks, BaseCallbackManager):
        handlers = [*callbacks.handlers, *callbacks.inheritable_handlers]
    else:
        handlers = callbacks if isinstance(callbacks, list) else [] if callbacks is None else [callbacks]
    # Handlers like Langfuse's hold clients deepcopy cannot rebuild; pre-seeding the memo reuses them as-is.
    return copy.deepcopy(config, memo={id(handler): handler for handler in handlers})


def _parse_graph_config_entry(graph_id: str, graph_config: object) -> GraphRegistryEntry:
    """Validate and normalize one graph configuration entry."""
    description: str | None = None
    if isinstance(graph_config, str):
        graph_path = graph_config
    elif isinstance(graph_config, dict):
        graph_config_data = cast(dict[str, object], graph_config)
        if "path" not in graph_config_data:
            raise ValueError(f"Graph '{graph_id}' configuration is missing required 'path'")
        graph_path = graph_config_data["path"]
        if not isinstance(graph_path, str):
            raise ValueError(f"Graph '{graph_id}' field 'path' must be a string")
        raw_description = graph_config_data.get("description")
        if raw_description is not None and not isinstance(raw_description, str):
            raise ValueError(f"Graph '{graph_id}' field 'description' must be a string or null")
        description = raw_description
    else:
        raise ValueError(f"Graph '{graph_id}' configuration must be a string or object")

    if ":" not in graph_path:
        raise ValueError(f"Invalid graph path format for '{graph_id}': {graph_path}")

    file_path, export_name = graph_path.split(":", 1)
    if not file_path or not export_name:
        raise ValueError(f"Invalid graph path format for '{graph_id}': {graph_path}")
    registry_entry = GraphRegistryEntry(file_path=file_path, export_name=export_name)
    if description is not None:
        registry_entry["description"] = description
    return registry_entry


def _business_graphs() -> frozenset[str]:
    """Graph ids that MUST NOT run without persistence (FollowMe business
    graphs). Trusted server config: ``AEGRA_BUSINESS_GRAPHS`` (comma- or
    space-separated). Empty set ⇒ no graph is classified business (pure
    legacy deployments keep the compat behavior)."""
    raw = settings.app.AEGRA_BUSINESS_GRAPHS or ""
    return frozenset(part.strip() for part in raw.replace(",", " ").split() if part.strip())


def _apply_injection_policy(graph_to_use: Pregel, graph_id: str, exc: Exception) -> Pregel:
    """Resolve a failed checkpointer/store injection per deployment policy.

    FollowMe business graphs (``AEGRA_BUSINESS_GRAPHS``) and any graph when
    the deployment mandates persistence (``AEGRA_PERSISTENCE_REQUIRED``)
    FAIL CLOSED: refuse to execute rather than silently dropping checkpoint
    state. Other graphs keep the legacy fallback (warn + unpersisted run).
    """
    if settings.checkpointer.AEGRA_PERSISTENCE_REQUIRED or graph_id in _business_graphs():
        logger.error(
            "graph_checkpointer_injection_failed",
            graph_id=graph_id,
            exc=str(exc),
            msg="Refusing to execute graph without persistence",
        )
        raise RuntimeError(
            f"Graph {graph_id!r} does not support checkpointer/store injection; "
            "refusing to run without persistence"
        ) from exc
    logger.warning(
        "graph_checkpointer_injection_failed",
        graph_id=graph_id,
        exc=str(exc),
        msg="Graph does not support checkpointer injection; running without persistence",
    )
    return graph_to_use


def _require_persistence_if_configured(checkpointer: Any, store: Any, graph_id: str) -> None:
    """Refuse execution when the deployment mandates persistence but the
    saver/store stack is half-initialized. Trusted server config only."""
    if settings.checkpointer.AEGRA_PERSISTENCE_REQUIRED and (checkpointer is None or store is None):
        raise RuntimeError(
            f"Persistence is required (AEGRA_PERSISTENCE_REQUIRED=true) but "
            f"checkpointer/store is unavailable for graph {graph_id!r}"
        )


#: The single authoritative seam a restricted graph exposes for read-path
#: persistence. Root-side restricted graphs raise a business error from
#: ``__getattr__``, so the seam is read statically — never by swallowing the
#: error and assuming the graph has no business saver. Plain compiled Pregel
#: graphs use their native ``checkpointer`` attribute instead.
RESTRICTED_GRAPH_CHECKPOINTER_ATTR = "readonly_checkpointer"

_MISSING = object()


def _static_attr(obj: Any, name: str) -> Any:
    """Existence probe only: never invokes ``__getattr__``/descriptors."""
    try:
        return inspect.getattr_static(obj, name, _MISSING)
    except Exception:  # pragma: no cover - pathological descriptors
        return _MISSING


def _eval_attr(obj: Any, name: str) -> Any:
    """Evaluate an attribute that the static probe proved exists.

    ``inspect.getattr_static`` returns a *descriptor* for properties (e.g. the
    root-side ``readonly_checkpointer`` is a property), so the value must be
    fetched through normal attribute access — inside a guard: a seam that exists
    but cannot be evaluated is fail-closed, never "absent".
    """
    try:
        return getattr(obj, name)
    except Exception as exc:
        raise RuntimeError(f"{name} exists but could not be evaluated: {exc}") from exc


def _resolve_graph_persistence(graph_obj: Any) -> tuple[Any, bool]:
    """Return ``(checkpointer, found)`` for a graph.

    ``getattr_static`` is used only to decide whether an attribute exists, so a
    restricted graph's ``__getattr__`` business error is never triggered while
    probing. The value itself is then read with normal attribute access, which
    correctly materialises property-based seams; evaluation failures propagate
    as ``RuntimeError`` (fail-closed) instead of being mistaken for "no saver".

    Order: the native ``checkpointer`` attribute (plain compiled Pregel), then
    the single restricted-graph seam ``readonly_checkpointer``.
    """
    if _static_attr(graph_obj, "checkpointer") is not _MISSING:
        return _eval_attr(graph_obj, "checkpointer"), True

    if _static_attr(graph_obj, RESTRICTED_GRAPH_CHECKPOINTER_ATTR) is _MISSING:
        return None, False

    seam = _eval_attr(graph_obj, RESTRICTED_GRAPH_CHECKPOINTER_ATTR)
    if seam is None:
        return None, True
    if isinstance(seam, BaseCheckpointSaver):
        return seam, True
    if _static_attr(seam, "checkpointer") is _MISSING:
        raise RuntimeError(
            f"{RESTRICTED_GRAPH_CHECKPOINTER_ATTR} is present but does not expose a checkpointer "
            f"(type {type(seam).__name__}); refusing to attach persistence blindly"
        )
    nested = _eval_attr(seam, "checkpointer")
    if nested is None:
        raise RuntimeError(
            f"{RESTRICTED_GRAPH_CHECKPOINTER_ATTR}.checkpointer is None "
            f"(type {type(seam).__name__}); refusing to attach persistence blindly"
        )
    return nested, True


def _attach_persistence(
    graph_obj: Any,
    checkpointer: BaseCheckpointSaver | None,
    store: BaseStore | None,
    user: Any,
    graph_id: str,
) -> Any:
    """Attach persistence to a compiled graph without clobbering a business saver.

    ``Pregel.copy(update={...})`` replaces attributes wholesale. If the graph (or
    its factory) already carries a checkpointer — for FollowMe graphs that is a
    boundary-aware saver with its own credential and physical-key policy — a
    blanket update silently overwrites it. The rules:

    * ``store``: always scoped to the trusted principal via
      :func:`TenantScopedStore`; raw stores are never handed to a graph.
      Execution AND read paths both need authorization, so ``user=None`` means
      no store is attached at all (fail-closed), except for the system
      registration path which does not go through this helper.
    * ``checkpointer``: attach the deployment checkpointer only when the graph
      does not already have one. A graph-supplied checkpointer is preserved
      untouched — this fork never imports or guesses a business saver class, and
      never shares a mutable saver between principals.

    Returns the graph instance to use.
    """
    from aegra_api.models.auth import User as _User
    from aegra_api.services.tenant_store import TenantScopedStore

    scoped_store: BaseStore | None = None
    if store is not None:
        if not isinstance(user, _User):
            raise HTTPException(
                403,
                f"Graph '{graph_id}' requires an authenticated principal for store access",
            )
        from aegra_api.services.run_preparation import trusted_identity

        tenant_id, owner_id = trusted_identity(user)
        scoped_store = TenantScopedStore(store, tenant_id, owner_id)

    try:
        existing_checkpointer, graph_supplies_persistence = _resolve_graph_persistence(graph_obj)
    except Exception as exc:
        # A seam that exists but cannot be evaluated is a hard failure: never
        # downgrade it to "graph has no saver".
        raise RuntimeError(f"Graph '{graph_id}' persistence seam is unusable: {exc}") from exc
    if not graph_supplies_persistence:
        if not isinstance(graph_obj, Pregel):
            # A non-Pregel object with no persistence seam is a restricted or
            # proxied graph whose execution contract we do not own. Guessing
            # "no saver" would overwrite a boundary saver; fail closed instead.
            raise RuntimeError(
                f"Graph '{graph_id}' exposes no persistence seam "
                f"(type {type(graph_obj).__name__}); refusing to attach persistence blindly"
            )
        # Plain compiled graph: attach the deployment checkpointer as before.
        return graph_obj.copy(update={"checkpointer": checkpointer, "store": scoped_store})
    if existing_checkpointer is None:
        # A seam explicitly declared "no checkpointer": attach the deployment one.
        return graph_obj.copy(update={"checkpointer": checkpointer, "store": scoped_store})

    logger.info(
        "Preserving graph-supplied checkpointer",
        graph_id=graph_id,
        saver_type=type(existing_checkpointer).__name__,
    )
    return graph_obj.copy(update={"store": scoped_store})


class LangGraphService:
    """Service to work with LangGraph CLI configuration and graphs.

    Architecture:
    - Caches base graph definitions (raw StateGraph/Pregel before checkpointer)
    - Yields fresh copies per-request with checkpointer/store injected
    - Thread-safe without locks via immutable cached state
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else Path("aegra.json")
        self._explicit_config = config_path is not None
        self.config: dict[str, Any] | None = None
        self._graph_registry: dict[str, GraphRegistryEntry] = {}
        # Cache for base graph definitions (without checkpointer/store).
        # For factory graphs, this holds the default-compiled graph (for schema extraction).
        self._base_graph_cache: dict[str, Pregel] = {}
        # Factory callables — stored alongside the base cache so they can be
        # invoked per-request with the appropriate ServerRuntime/config.
        self._graph_factories: dict[str, Callable] = {}

    async def initialize(self) -> None:
        """Load configuration file and setup graph registry.

        Resolution order:
        1) Explicit config_path passed to constructor (if it exists)
        2) Shared resolution (AEGRA_CONFIG env var → aegra.json → langgraph.json)
        """
        from aegra_api.config import _resolve_config_path

        # 1) Explicit path wins if provided and exists
        if self._explicit_config and self.config_path.exists():
            resolved_path = self.config_path
        # 2) Otherwise use shared resolution (warn if explicit path was missing)
        else:
            if self._explicit_config:
                logger.warning(f"Explicit config path '{self.config_path}' not found, falling back to config discovery")
            resolved_path = _resolve_config_path()

        if not resolved_path or not resolved_path.exists():
            raise ValueError(
                "Configuration file not found. Expected one of: AEGRA_CONFIG path, ./aegra.json, or ./langgraph.json"
            )

        self.config_path = resolved_path

        with self.config_path.open() as f:
            self.config = json.load(f)

        # Setup dependency paths before loading graphs
        self._setup_dependencies()

        # Load graph registry from config
        self._load_graph_registry()

        # Eagerly load all graph modules so factory graphs are classified
        # before the first request arrives.  This populates _graph_factories
        # (and _base_graph_cache for static graphs) without calling factory
        # functions with empty user/config — fixing the bug where the first
        # request would fall through to the static path and get a graph
        # compiled with user=None.
        await self._load_all_graph_modules()

        # Pre-register assistants for each graph using deterministic UUIDs so
        # clients can pass graph_id directly.
        await self._ensure_default_assistants()

    def _load_graph_registry(self) -> None:
        """Load graph definitions from aegra.json"""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        raw_graphs_config = self.config.get("graphs", {})
        if not isinstance(raw_graphs_config, dict):
            raise ValueError("Configuration field 'graphs' must be an object")
        graphs_config = cast(dict[str, str | GraphConfigEntry], raw_graphs_config)

        # Detect graph_ids that collide after sanitisation (e.g. "a.b" and
        # "a_b" both map to aegra_graphs.a_b in sys.modules).
        seen_modules: dict[str, str] = {}
        for graph_id in graphs_config:
            mod_name = _module_name_for(graph_id)
            if mod_name in seen_modules:
                raise ValueError(
                    f"Graph IDs '{seen_modules[mod_name]}' and '{graph_id}' "
                    f"collide after sanitisation (both map to {mod_name})"
                )
            seen_modules[mod_name] = graph_id

        for graph_id, graph_config in graphs_config.items():
            self._graph_registry[graph_id] = _parse_graph_config_entry(graph_id, graph_config)

    async def _load_all_graph_modules(self) -> None:
        """Eagerly load all graph modules, classifying factories without calling them.

        For static graphs, the compiled graph is cached in ``_base_graph_cache``.
        For factory graphs, the callable is stored in ``_graph_factories`` for
        per-request invocation — the factory is NOT called here, so no graph is
        compiled with ``user=None``.
        """
        for graph_id, graph_info in self._graph_registry.items():
            raw_graph = await self._load_graph_from_file(graph_id, graph_info)
            if raw_graph is not None:
                # Static graph — compile and cache
                if isinstance(raw_graph, StateGraph):
                    logger.info("compiling_graph", graph_id=graph_id)
                    raw_graph = raw_graph.compile()
                self._base_graph_cache[graph_id] = raw_graph

    def _setup_dependencies(self) -> None:
        """Add dependency paths to sys.path for graph imports.

        Supports paths from the 'dependencies' config key.
        Paths are resolved relative to the config file location.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded")
        dependencies = self.config.get("dependencies", [])
        if not dependencies:
            return

        config_dir = self.config_path.parent

        # Iterate in reverse so first dependency in config has highest priority
        for dep in reversed(dependencies):
            dep_path = Path(dep)

            # Resolve relative paths from config directory
            if not dep_path.is_absolute():
                dep_path = (config_dir / dep_path).resolve()
            else:
                dep_path = dep_path.resolve()

            # Add to sys.path if exists and not already present
            path_str = str(dep_path)
            if dep_path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)
                logger.info(f"Added dependency path to sys.path: {path_str}")
            elif not dep_path.exists():
                logger.warning(f"Dependency path does not exist: {path_str}")

    async def _ensure_default_assistants(self) -> None:
        """Create default assistants and sync their configured descriptions.

        Uses uuid5 with a fixed namespace so that the same graph_id maps
        to the same assistant_id across restarts. Idempotent.
        """
        # Fixed namespace used to derive assistant IDs from graph IDs
        NS = ASSISTANT_NAMESPACE_UUID
        session_gen = get_session()
        session = await anext(session_gen)
        try:
            for graph_id, graph_config in self._graph_registry.items():
                assistant_id = str(uuid5(NS, graph_id))
                description = graph_config.get("description", f"Default assistant for graph '{graph_id}'")
                existing = await session.scalar(
                    select(AssistantORM).where(AssistantORM.assistant_id == assistant_id).with_for_update()
                )
                if existing:
                    if existing.user_id == "system" and existing.description != description:
                        max_version = await session.scalar(
                            select(func.max(AssistantVersionORM.version)).where(
                                AssistantVersionORM.assistant_id == assistant_id
                            )
                        )
                        new_version = max(existing.version, max_version or 0) + 1
                        now = datetime.now(UTC)
                        session.add(
                            AssistantVersionORM(
                                assistant_id=assistant_id,
                                version=new_version,
                                name=existing.name,
                                description=description,
                                graph_id=existing.graph_id,
                                config=existing.config,
                                context=existing.context,
                                metadata_dict=existing.metadata_dict,
                                created_at=now,
                            )
                        )
                        existing.description = description
                        existing.version = new_version
                        existing.updated_at = now
                    continue
                session.add(
                    AssistantORM(
                        assistant_id=assistant_id,
                        name=graph_id,
                        description=description,
                        graph_id=graph_id,
                        config={},
                        user_id="system",
                        metadata_dict={"created_by": "system"},
                    )
                )
                session.add(
                    AssistantVersionORM(
                        assistant_id=assistant_id,
                        version=1,
                        name=graph_id,
                        description=description,
                        graph_id=graph_id,
                        metadata_dict={"created_by": "system"},
                    )
                )
            await session.commit()
        finally:
            await session.close()

    async def _get_base_graph(self, graph_id: str) -> Pregel:
        """Get the base compiled graph without checkpointer/store.

        Caches the compiled graph structure for reuse. This is safe because
        the base graph is immutable - we create copies with checkpointer/store
        injected per-request.

        For factory graphs whose modules were already loaded by
        ``_load_all_graph_modules``, this lazily calls the factory with
        default args to produce a base graph for schema extraction.

        @param graph_id: The graph identifier from aegra.json
        @returns: Compiled Pregel graph (without checkpointer/store)
        @raises ValueError: If graph_id not found or loading fails
        """
        if graph_id not in self._graph_registry:
            raise ValueError(f"Graph not found: {graph_id}")

        # Return cached base graph if available
        if graph_id in self._base_graph_cache:
            return self._base_graph_cache[graph_id]

        # Factory graph that was classified but never called with defaults yet.
        # Call now to produce a base graph for schema extraction.
        if graph_id in self._graph_factories:
            raw_graph = await self._call_factory_with_defaults(self._graph_factories[graph_id], graph_id)
        else:
            graph_info = self._graph_registry[graph_id]
            # Load graph from file — may return None if the module turns out
            # to be a factory (discovered during lazy loading).
            raw_graph = await self._load_graph_from_file(graph_id, graph_info)

        # _load_graph_from_file returns None for factory graphs that were
        # just discovered and stored in _graph_factories. Retry via the
        # factory path now that the callable is registered.
        if raw_graph is None and graph_id in self._graph_factories:
            raw_graph = await self._call_factory_with_defaults(self._graph_factories[graph_id], graph_id)

        if raw_graph is None:
            raise ValueError(f"Failed to load graph '{graph_id}': module did not produce a graph")

        # Compile if it's a StateGraph
        if isinstance(raw_graph, StateGraph):
            logger.info(f"🔧 Compiling graph '{graph_id}'")
            compiled_graph = raw_graph.compile()
        else:
            compiled_graph = raw_graph

        # Only cache static graphs — factory graphs must be re-invoked
        # each time so their result reflects the current user/config context.
        if graph_id not in self._graph_factories:
            self._base_graph_cache[graph_id] = compiled_graph
        return compiled_graph

    async def _resolve_factory(self, graph_id: str) -> Callable | None:
        """Resolve the factory callable, lazily re-loading after invalidation.

        Normally populated by _load_all_graph_modules() at startup, but may
        be empty after invalidate_cache(). In that case re-load the module
        to re-classify the factory without invoking it with user=None.
        """
        factory = self._graph_factories.get(graph_id)
        if not factory and graph_id not in self._base_graph_cache:
            # Neither factory nor cached static graph — re-load from file
            # (post-invalidation path).
            graph_info = self._graph_registry.get(graph_id)
            if graph_info:
                await self._load_graph_from_file(graph_id, graph_info)
            factory = self._graph_factories.get(graph_id)
        return factory

    @asynccontextmanager
    async def get_graph(
        self,
        graph_id: str,
        *,
        config: dict[str, Any] | None = None,
        access_context: AccessContext = "threads.create_run",
        user: User | BaseUser | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[Pregel]:
        """Get a graph instance for execution with checkpointer/store injected.

        For factory graphs, the factory is invoked per-request with the
        appropriate ``ServerRuntime`` and config. For static graphs, the cached
        base graph is returned with checkpointer/store injected.

        This is a context manager that yields a fresh graph copy per-request.
        Thread-safe without locks since each request gets its own instance.

        Usage::

            async with langgraph_service.get_graph(
                "react_agent",
                config=run_config,
                access_context="threads.create_run",
                user=user,
                context=context,
            ) as graph:
                async for event in graph.astream(input, config):
                    ...

        Args:
            graph_id: The graph identifier from aegra.json.
            config: The ``RunnableConfig`` dict for this request.
            access_context: Why the graph is being accessed.
            user: The authenticated user (or ``None`` for anonymous access).
            context: The raw request context dict. For factories with
                ``ServerRuntime[T]``, this is coerced to ``T`` and passed
                to ``_ExecutionRuntime.context``.

        Yields:
            Compiled ``Pregel`` graph with Postgres checkpointer/store attached.

        Raises:
            ValueError: If *graph_id* not found or loading fails.
        """
        from aegra_api.core.database import db_manager

        checkpointer = db_manager.get_checkpointer()
        store = db_manager.get_store()

        _require_persistence_if_configured(checkpointer, store, graph_id)

        factory = await self._resolve_factory(graph_id)

        if factory:
            # Factory path — invoke per-request with ServerRuntime
            run_config = config or {"configurable": {}}
            coerced_context = coerce_context(context, graph_id)
            server_runtime = build_server_runtime(
                access_context=access_context,
                store=store,
                user=user,
                context=coerced_context,
            )

            result = invoke_factory(factory, graph_id, run_config, server_runtime)

            async with generate_graph(result, graph_id) as graph_obj:
                if isinstance(graph_obj, StateGraph):
                    graph_obj = graph_obj.compile()

                # Injection failure handling is deployment policy
                # (_apply_injection_policy). Do this BEFORE yield to avoid
                # dual-yield when exceptions occur in the context body.
                try:
                    graph_to_use = _attach_persistence(
                        graph_obj, checkpointer, store, user, graph_id
                    )
                except Exception as exc:
                    graph_to_use = _apply_injection_policy(graph_obj, graph_id, exc)

                yield graph_to_use
        else:
            # Static path — use cached base graph (already loaded above)
            base_graph = await self._get_base_graph(graph_id)
            # Try to create a copy with checkpointer/store injected.
            # NOTE: Do this BEFORE yield to avoid dual-yield when exceptions
            # occur in the context body.
            # Deep-copy the graph's config to prevent astream from mutating
            # the cached base graph's nested dicts (e.g. config.metadata).
            # Pregel.copy() is shallow, so without this, all copies share
            # the same metadata dict, and writes during execution pollute
            # the base graph — breaking subsequent runs.

            try:
                graph_to_use = base_graph.copy(
                    update={
                        "checkpointer": checkpointer,
                        "store": store,
                        "config": _copy_graph_config(base_graph.config),
                    }
                )
            except Exception as exc:
                graph_to_use = _apply_injection_policy(base_graph, graph_id, exc)

            yield graph_to_use

    async def get_graph_for_validation(
        self,
        graph_id: str,
        *,
        config: dict[str, Any] | None = None,
        access_context: AccessContext = "assistants.read",
        user: User | BaseUser | None = None,
    ) -> Pregel:
        """Get a graph instance for validation/schema extraction only.

        For factory graphs, the factory is invoked with the given access context
        (default ``assistants.read``) to produce a fresh graph. For static graphs,
        returns the cached base graph.

        Does NOT include checkpointer/store — use ``get_graph()`` for execution.

        Args:
            graph_id: The graph identifier from aegra.json.
            config: Optional ``RunnableConfig`` dict (passed to factory).
            access_context: Why the graph is being accessed (default: schema read).
            user: The authenticated user (or ``None`` for anonymous access).

        Returns:
            Compiled ``Pregel`` graph (without checkpointer/store).

        Raises:
            ValueError: If *graph_id* not found or loading fails.
        """
        if graph_id in self._graph_factories:
            factory = self._graph_factories[graph_id]
            run_config = config or {"configurable": {}}
            server_runtime = build_server_runtime(
                access_context=access_context,
                store=None,
                user=user,
            )

            result = invoke_factory(factory, graph_id, run_config, server_runtime)
            async with generate_graph(result, graph_id) as graph_obj:
                if isinstance(graph_obj, StateGraph):
                    graph_obj = graph_obj.compile()
                # Capture the graph before exiting the context manager.
                # For async-CM-backed factories, __aexit__ fires after this
                # block. Since validation only reads schema metadata, the
                # compiled Pregel is safe to use after CM teardown.
                validated_graph = graph_obj
            return validated_graph

        return await self._get_base_graph(graph_id)

    async def _load_graph_from_file(self, graph_id: str, graph_info: GraphRegistryEntry) -> Pregel | StateGraph | None:
        """Load graph from filesystem.

        Paths are resolved relative to the config file's directory.

        For callable exports, the function inspects the signature to detect
        factory patterns. 0-arg factories are called once at load time.
        Config/runtime factories are registered in ``_graph_factories`` for
        per-request invocation and ``None`` is returned — no default-args
        call is made here, so factory graphs are never compiled with
        ``user=None`` at startup.

        Args:
            graph_id: The graph identifier from aegra.json.
            graph_info: Dict with ``file_path`` and ``export_name`` keys.

        Returns:
            A compiled ``Pregel`` or uncompiled ``StateGraph`` for static
            graphs, or ``None`` for factory graphs (the factory is stored
            in ``_graph_factories`` instead).

        Raises:
            ValueError: If the file or export is not found.
        """
        raw_path = graph_info["file_path"]
        file_path = Path(raw_path)

        # Resolve relative paths from config file directory
        if not file_path.is_absolute():
            file_path = (self.config_path.parent / file_path).resolve()

        if not file_path.exists():
            raise ValueError(f"Graph file not found: {file_path}")

        # Dynamic import of graph module
        spec = importlib.util.spec_from_file_location(_module_name_for(graph_id), str(file_path.resolve()))
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load graph module: {file_path}")

        module = importlib.util.module_from_spec(spec)
        module_name = spec.name
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        # Get the exported graph
        export_name = graph_info["export_name"]
        if not hasattr(module, export_name):
            raise ValueError(f"Graph export not found: {export_name} in {file_path}")

        graph = getattr(module, export_name)

        if callable(graph):
            # Classify the factory signature (populates the dispatch hook registry)
            classify_factory(graph, graph_id)

            if not is_factory(graph_id):
                # 0-arg factory — call once at load time
                if asyncio.iscoroutinefunction(graph):
                    graph = await graph()
                else:
                    graph = graph()
                # Handle async context managers / coroutines returned by 0-arg factories
                async with generate_graph(graph, graph_id) as resolved:
                    graph = resolved
            else:
                # Config/runtime factory — store the callable for per-request
                # invocation.  Do NOT call with defaults here; the factory will
                # be invoked with the real user/config on each request.  If a
                # base graph is needed later for schema extraction, _get_base_graph
                # will call _call_factory_with_defaults lazily.
                self._graph_factories[graph_id] = graph
                return None

        return graph

    async def _call_factory_with_defaults(self, fn: Callable, graph_id: str) -> Pregel | StateGraph:
        """Call a factory with minimal args to get a base graph for schema extraction.

        Intended **only** for introspection (schema extraction, graph
        structure discovery) — never for the hot execution path. Uses
        ``assistants.read`` access context and ``user=None``.

        .. note::

            ``build_server_runtime(user=None)`` may fall back to
            ``get_auth_ctx()`` if called within an HTTP request context,
            so the runtime may carry a real user. This is incidental,
            not guaranteed — callers must not rely on it.

        Args:
            fn: The factory callable.
            graph_id: The graph identifier.

        Returns:
            A compiled ``Pregel`` or uncompiled ``StateGraph``.
        """
        empty_config: dict[str, Any] = {"configurable": {}}
        runtime = build_server_runtime(
            access_context="assistants.read",
            store=None,
            user=None,
        )
        result = invoke_factory(fn, graph_id, empty_config, runtime)

        # Resolve async context managers, coroutines, etc.
        async with generate_graph(result, graph_id) as resolved:
            graph = resolved
        return graph

    def list_graphs(self) -> dict[str, str]:
        """List all available graphs."""
        return {graph_id: info["file_path"] for graph_id, info in self._graph_registry.items()}

    def invalidate_cache(self, graph_id: str | None = None) -> None:
        """Invalidate graph cache for hot-reload.

        Clears both the base graph cache and the factory callable registry,
        as well as the factory dispatch hooks in the ``graph_factory`` module.
        The module is also removed from ``sys.modules`` so the next import
        re-executes the file.

        After invalidation, ``_graph_factories`` is empty. The next call to
        ``_get_base_graph`` will fall through to ``_load_graph_from_file``,
        which re-discovers and re-classifies the factory, then retries via
        the factory path — so callers do not need to re-run
        ``_load_all_graph_modules`` after invalidation.

        Args:
            graph_id: Specific graph to invalidate, or ``None`` to clear all.
        """
        if graph_id:
            self._base_graph_cache.pop(graph_id, None)
            self._graph_factories.pop(graph_id, None)
            sys.modules.pop(_module_name_for(graph_id), None)
        else:
            self._base_graph_cache.clear()
            self._graph_factories.clear()
            for key in list(sys.modules.keys()):
                if key.startswith("aegra_graphs."):
                    sys.modules.pop(key, None)
        clear_factory_registry(graph_id)

    def get_config(self) -> dict[str, Any] | None:
        """Get loaded configuration"""
        return self.config

    def get_dependencies(self) -> list:
        """Get dependencies from config"""
        if self.config is None:
            return []
        return self.config.get("dependencies", [])

    def get_http_config(self) -> dict[str, Any] | None:
        """Get HTTP configuration from loaded config file.

        Returns:
            HTTP configuration dict or None if not configured
        """
        if self.config is None:
            return None
        return self.config.get("http")


# Global service instance
_langgraph_service = None


def get_langgraph_service() -> LangGraphService:
    """Get global LangGraph service instance"""
    global _langgraph_service
    if _langgraph_service is None:
        _langgraph_service = LangGraphService()
    return _langgraph_service


def inject_user_context(user: Any | None, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inject user context into LangGraph configuration for user isolation.

    Passes ALL user fields (including custom auth handler fields like
    subscription_tier, team_id, etc.) to the graph config under
    'langgraph_auth_user'.

    Args:
        user: User object with identity and optional extra fields
        base_config: Base configuration to extend

    Returns:
        Configuration dict with user context injected
    """
    config: dict[str, Any] = (base_config or {}).copy()
    # Copy the nested dict too: callers reuse a base config across runs/users
    # (e.g. one template for A and B), so writing trusted identity into a
    # shared ``configurable`` would leak A's user_id into B's config.
    config["configurable"] = dict(config.get("configurable") or {})

    # All user-related data injection (only if user exists)
    if user:
        # Server-authenticated identity always wins: a client-supplied
        # configurable.user_id must never masquerade as another tenant.
        # Non-identity custom fields from the client config are preserved.
        config["configurable"]["user_id"] = user.identity
        config["configurable"]["user_display_name"] = getattr(user, "display_name", None) or user.identity

        config["configurable"]["langgraph_auth_user"] = user

    return config


def create_thread_config(
    thread_id: str, user: User | BaseUser | None, *, additional_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create LangGraph configuration for a specific thread with user context"""
    base_config = {"configurable": {"thread_id": thread_id}}

    if additional_config:
        base_config.update(additional_config)

    return inject_user_context(user, base_config)


def create_run_config(
    run_id: str,
    thread_id: str,
    user: User | BaseUser | None,
    *,
    additional_config: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create LangGraph configuration for a specific run with full context.

    Additive for client keys, except thread_id/run_id which are forced: a
    body-supplied configurable.thread_id would redirect execution to another
    user's thread (the checkpointer keys on thread_id alone).
    """
    from copy import deepcopy

    cfg: dict = deepcopy(additional_config) if additional_config else {}
    cfg.setdefault("configurable", {})

    # Server-authoritative — overwrite, never honor a client override.
    cfg["configurable"]["thread_id"] = thread_id
    cfg["configurable"]["run_id"] = run_id

    # Ensure the root run ID is set to match so that astream_events recognizes it
    cfg.setdefault("run_id", run_id)

    # Add observability callbacks from various potential sources
    tracing_callbacks = get_tracing_callbacks()
    if tracing_callbacks:
        existing_callbacks = cfg.get("callbacks", [])
        if not isinstance(existing_callbacks, list):
            # If we want to be more robust, we can log a warning here
            existing_callbacks = []

        # Combine existing callbacks with new tracing callbacks to be non-destructive
        cfg["callbacks"] = existing_callbacks + tracing_callbacks

    # Add metadata from all observability providers (independent of callbacks)
    cfg.setdefault("metadata", {})
    user_identity = user.identity if user else None
    observability_metadata = get_tracing_metadata(run_id, thread_id, user_identity)
    cfg["metadata"].update(observability_metadata)

    # Apply checkpoint parameters if provided. Strip pinned identity keys so a
    # client checkpoint can't redirect execution to another user's thread.
    if checkpoint and isinstance(checkpoint, dict):
        safe = strip_pinned_config_keys(checkpoint)
        cfg["configurable"].update({k: v for k, v in safe.items() if v is not None})

    # Finally inject user context via existing helper
    return inject_user_context(user, cfg)
