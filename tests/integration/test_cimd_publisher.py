"""#992 integration: the registry publishes its own CIMD (Client ID Metadata
Document) so it can authenticate as an OAuth CLIENT to external CIMD-aware IdPs.
In-process (fake settings, no network), matching the discovery-chain style."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api.wellknown_routes import cimd_router

pytestmark = [pytest.mark.integration]

TEST_URL = "https://mcpgateway.test"
CIMD_PATH = "/oauth/client-metadata.json"


def _app(monkeypatch, **overrides) -> FastAPI:
    from registry.api import wellknown_routes as wkr
    from registry.auth import oauth_metadata as om

    class _Stub:
        registry_url = TEST_URL
        egress_oauth_callback_base = TEST_URL
        mcp_https_required = True
        cimd_publisher_enabled = True
        cimd_cache_ttl = 3600
        cimd_client_name = "AI Registry Tools"
        cimd_redirect_uris = ""
        cimd_scope = ""
        cimd_logo_uri = ""
        cimd_contacts = ""

    for k, v in overrides.items():
        setattr(_Stub, k, v)
    stub = _Stub()
    monkeypatch.setattr(wkr, "settings", stub)
    monkeypatch.setattr(om, "settings", stub)

    app = FastAPI()
    app.include_router(cimd_router)
    return app


def test_disabled_returns_404(monkeypatch):
    client = TestClient(_app(monkeypatch, cimd_publisher_enabled=False))
    assert client.get(CIMD_PATH).status_code == 404


def test_document_shape_and_client_id_parity(monkeypatch):
    client = TestClient(_app(monkeypatch))
    r = client.get(CIMD_PATH)

    assert r.status_code == 200
    assert r.headers["cache-control"] == "public, max-age=3600"
    assert r.headers["content-type"].startswith("application/json")

    doc = r.json()
    for field in (
        "client_id",
        "client_name",
        "client_uri",
        "redirect_uris",
        "scope",
        "grant_types",
        "token_endpoint_auth_method",
    ):
        assert field in doc, f"missing required field {field}"

    # The client_id IS the canonical URL of this very document (byte-for-byte).
    assert doc["client_id"] == f"{TEST_URL}{CIMD_PATH}"
    assert doc["token_endpoint_auth_method"] == "none"
    assert "authorization_code" in doc["grant_types"]
    assert "refresh_token" in doc["grant_types"]
    assert doc["client_uri"] == TEST_URL
    # Defaults derive from registry_url; optional fields omitted when unset.
    assert doc["redirect_uris"] == [f"{TEST_URL}/oauth2/egress/callback"]
    assert "logo_uri" not in doc
    assert "contacts" not in doc


def test_optional_fields_and_overrides(monkeypatch):
    client = TestClient(
        _app(
            monkeypatch,
            cimd_logo_uri="https://cdn.test/logo.png",
            cimd_contacts="ops@test, sec@test",
            cimd_redirect_uris="https://a.test/cb, https://b.test/cb",
            cimd_scope="openid profile",
        )
    )
    doc = client.get(CIMD_PATH).json()
    assert doc["logo_uri"] == "https://cdn.test/logo.png"
    assert doc["contacts"] == ["ops@test", "sec@test"]
    assert doc["redirect_uris"] == ["https://a.test/cb", "https://b.test/cb"]
    assert doc["scope"] == "openid profile"


def test_field_order_is_stable(monkeypatch):
    client = TestClient(_app(monkeypatch))
    doc = client.get(CIMD_PATH).json()
    assert list(doc.keys())[:8] == [
        "client_id",
        "client_name",
        "client_uri",
        "redirect_uris",
        "grant_types",
        "response_types",
        "token_endpoint_auth_method",
        "scope",
    ]


def test_client_id_uses_egress_callback_base_over_registry_url(monkeypatch):
    """client_id and the default redirect derive from the egress OAuth callback
    base, not registry_url -- so a deployment whose external callback host differs
    from registry_url still publishes a reachable client_id/redirect (issue #992)."""
    client = TestClient(_app(monkeypatch, egress_oauth_callback_base="https://callback.test"))
    doc = client.get(CIMD_PATH).json()
    assert doc["client_id"] == f"https://callback.test{CIMD_PATH}"
    assert doc["client_uri"] == "https://callback.test"
    assert doc["redirect_uris"] == ["https://callback.test/oauth2/egress/callback"]


def test_client_id_path_matches_actual_router_mount():
    """The client_id constant must equal the route's served path, so a route
    rename or an accidental prefix on the mount can't silently break client_id /
    served-URL parity -- the drift the in-process tests above (which hardcode the
    path) would otherwise miss."""
    import re
    from pathlib import Path

    from registry.api.wellknown_routes import cimd_router
    from registry.auth.oauth_metadata import CIMD_PATH

    route_paths = [
        r.path for r in cimd_router.routes if getattr(r, "name", "") == "get_oauth_client_metadata"
    ]
    assert route_paths == [CIMD_PATH]

    # cimd_router MUST be mounted at root (no prefix) so the served path -- and
    # therefore the client_id -- is exactly CIMD_PATH.
    main_src = (Path(__file__).resolve().parents[2] / "registry" / "main.py").read_text()
    assert re.search(r"include_router\(\s*cimd_router\s*,", main_src), (
        "cimd_router not mounted in registry/main.py"
    )
    assert not re.search(r"include_router\(\s*cimd_router\s*,[^)]*prefix=", main_src), (
        "cimd_router must be mounted at root (no prefix) so client_id == served path"
    )


def test_client_id_has_exactly_one_construction_site():
    """`build_cimd_client_id_url` must stay the only place the client_id is built.

    The URL IS the client_id, so two construction sites that drift produce a
    document advertising one value while an outbound /authorize sends another,
    and the IdP rejects the request with nothing in the document to explain why.
    #992's outbound wiring is not in this branch yet; this guards the seam so the
    future path reuses the helper instead of re-deriving the URL.

    Enforced by source scan rather than by call graph: a new f-string like
    f"{base}/oauth/client-metadata.json" in some other module is exactly the
    mistake, and it is invisible to a test that only calls the helper.
    """
    import re
    from pathlib import Path

    from registry.auth.oauth_metadata import CIMD_PATH

    registry_root = Path(__file__).resolve().parents[2] / "registry"

    # Where the path literal or the constant may legitimately appear:
    #   oauth_metadata.py  - defines CIMD_PATH and the one builder
    #   wellknown_routes.py - the route decorator, pinned to CIMD_PATH by
    #                         test_client_id_matches_served_path above
    ALLOWED = {"oauth_metadata.py", "wellknown_routes.py"}

    # A construction site joins the path onto something else: an f-string with a
    # preceding brace, or explicit concatenation. A bare mention (a comment, a
    # docstring, an equality check) is not.
    joined = re.compile(
        rf"(\{{[^}}]*\}}\s*{re.escape(CIMD_PATH)})"  # f"{base}/oauth/client-metadata.json"
        rf"|(\+\s*(?:CIMD_PATH|[\"']{re.escape(CIMD_PATH)}[\"']))"  # base + CIMD_PATH
        rf"|(\{{[^}}]*\}}\{{CIMD_PATH\}})"  # f"{base}{CIMD_PATH}"
    )

    offenders = []
    for path in sorted(registry_root.rglob("*.py")):
        if path.name in ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            if joined.search(code):
                rel = path.relative_to(registry_root.parent)
                offenders.append(f"  {rel}:{lineno}  {line.strip()}")

    assert not offenders, (
        "The CIMD client_id is constructed somewhere other than "
        "build_cimd_client_id_url(). Call that helper instead, so the published "
        "document and any outbound client_id cannot diverge:\n" + "\n".join(offenders)
    )
