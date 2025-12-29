from __future__ import annotations

import importlib.util
import json
from pathlib import (
    Path,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY_CLIENT_PATH = _REPO_ROOT / "api" / "registry_client.py"

_SPEC = importlib.util.spec_from_file_location(
    "registry_client",
    _REGISTRY_CLIENT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

InternalServiceRegistration = _MODULE.InternalServiceRegistration


class TestInternalServiceRegistration:
    def test_model_dump_matches_form_contract(self) -> None:
        registration = InternalServiceRegistration(
            service_path="/svc",
            name="svc",
            description="svc",
            proxy_pass_url="http://example.invalid",
            supported_transports=json.dumps(["streamable-http"]),
            headers=json.dumps({"X-Test": "1"}),
            tool_list_json=json.dumps([{"name": "tool1"}]),
        )

        payload = registration.model_dump(
            exclude_none=True,
            by_alias=True,
        )

        assert payload["path"] == "/svc"
        assert payload["name"] == "svc"
        assert payload["description"] == "svc"
        assert payload["proxy_pass_url"] == "http://example.invalid"
        assert payload["supported_transports"] == json.dumps(["streamable-http"])
        assert payload["headers"] == json.dumps({"X-Test": "1"})
        assert payload["tool_list_json"] == json.dumps([{"name": "tool1"}])
        assert payload["overwrite"] is True
