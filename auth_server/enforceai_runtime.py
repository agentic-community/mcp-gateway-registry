from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class _EnforceAIRuntime:
    DependencyUnavailableError: type[Exception]
    EnforceAIError: type[Exception]
    evaluate_tool_call: object
    resolve_callable_tools_for_server: object
    load_scope_catalog: object
    get_enforceai_settings: object
    get_enforceai_stores: object
    get_identity_resolver: object
    get_upstream_oauth_token_client: object


@lru_cache(maxsize=1)
def _load_enforceai_runtime() -> _EnforceAIRuntime:
    for base in ("auth_server.enforceai", "enforceai"):
        try:
            importlib.import_module(base)
        except ModuleNotFoundError:
            continue

        errors_module = importlib.import_module(f"{base}.errors")
        evaluate_module = importlib.import_module(f"{base}.fgac.evaluate")
        catalog_module = importlib.import_module(f"{base}.fgac.catalog")
        dependency_module = importlib.import_module(f"{base}.auth.dependency")

        return _EnforceAIRuntime(
            DependencyUnavailableError=errors_module.DependencyUnavailableError,
            EnforceAIError=errors_module.EnforceAIError,
            evaluate_tool_call=evaluate_module.evaluate_tool_call,
            resolve_callable_tools_for_server=evaluate_module.resolve_callable_tools_for_server,
            load_scope_catalog=catalog_module.load_scope_catalog,
            get_enforceai_settings=dependency_module.get_enforceai_settings,
            get_enforceai_stores=dependency_module.get_enforceai_stores,
            get_identity_resolver=dependency_module.get_identity_resolver,
            get_upstream_oauth_token_client=dependency_module.get_upstream_oauth_token_client,
        )

    raise RuntimeError("EnforceAI runtime could not be imported")


def get_identity_resolver():
    return _load_enforceai_runtime().get_identity_resolver()


def get_enforceai_settings():
    return _load_enforceai_runtime().get_enforceai_settings()


def get_enforceai_stores():
    return _load_enforceai_runtime().get_enforceai_stores()


def get_upstream_oauth_token_client():
    return _load_enforceai_runtime().get_upstream_oauth_token_client()


def load_scope_catalog(
    *,
    path: Optional[Path] = None,
):
    return _load_enforceai_runtime().load_scope_catalog(path=path)


def evaluate_tool_call(
    *,
    identity,
    catalog,
    server: str,
    tool: str,
    allowed_tools,
):
    return _load_enforceai_runtime().evaluate_tool_call(
        identity=identity,
        catalog=catalog,
        server=server,
        tool=tool,
        allowed_tools=allowed_tools,
    )


def resolve_callable_tools_for_server(
    *,
    identity,
    catalog,
    server: str,
    allowed_tools,
):
    return _load_enforceai_runtime().resolve_callable_tools_for_server(
        identity=identity,
        catalog=catalog,
        server=server,
        allowed_tools=allowed_tools,
    )


def load_enforceai_management_router():
    for base in ("auth_server.enforceai", "enforceai"):
        try:
            importlib.import_module(base)
        except ModuleNotFoundError:
            continue

        try:
            module = importlib.import_module(f"{base}.api.management_routes")
        except ModuleNotFoundError:
            continue

        router = getattr(module, "router", None)
        if router is not None:
            return router

    return None

