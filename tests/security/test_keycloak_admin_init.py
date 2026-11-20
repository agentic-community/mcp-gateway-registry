"""Exercise the deployed reconciliation script over real HTTP, without a cluster."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import pytest

SCRIPT = Path(__file__).parents[2] / "charts" / "keycloak" / "files" / "initialize-admin.py"
PASSWORD = "synthetic-admin-credential"


@pytest.fixture
def admin_endpoint():
    state = {
        "sslRequired": "external",
        "auth_calls": 0,
        "updates": 0,
        "reject_auth": False,
        "ignore_update": False,
        "redirect": False,
        "forwarded_requests": 0,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, body=None):
            self.send_response(status)
            self.end_headers()
            if body is not None:
                self.wfile.write(json.dumps(body).encode())

        def do_POST(self):
            if self.path == "/unexpected":
                state["forwarded_requests"] += 1
                return self.respond(500)
            state["auth_calls"] += 1
            fields = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
            if state["redirect"]:
                self.send_response(307)
                self.send_header("Location", "/unexpected")
                self.end_headers()
                return
            valid = (
                self.path == "/realms/master/protocol/openid-connect/token"
                and fields.get("password") == [PASSWORD]
                and fields.get("username") == ["operator"]
                and self.headers.get("X-Forwarded-Proto") == "https"
                and self.headers.get("X-Forwarded-Port") == "443"
            )
            if state["reject_auth"] or not valid:
                return self.respond(401, {"error_description": PASSWORD})
            self.respond(200, {"access_token": "synthetic-admin-token"})

        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer synthetic-admin-token":
                return self.respond(401)
            self.respond(200, {"sslRequired": state["sslRequired"]})

        def do_PUT(self):
            state["updates"] += 1
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not state["ignore_update"]:
                state["sslRequired"] = payload["sslRequired"]
            self.respond(204)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def run():
        env = {
            **os.environ,
            "KEYCLOAK_ADMIN_URL": f"http://127.0.0.1:{server.server_port}",
            "KEYCLOAK_HOSTNAME": "https://gateway.example.com",
            "KEYCLOAK_ADMIN_REALM": "master",
            "KEYCLOAK_ADMIN_USERNAME": "operator",
            "KEYCLOAK_ADMIN_PASSWORD": PASSWORD,
        }
        return subprocess.run(
            [sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10
        )

    try:
        yield state, run
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_reconciles_and_reads_back_without_rewriting_verified_state(admin_endpoint):
    state, run = admin_endpoint
    first = run()
    assert first.returncode == 0, first.stderr
    assert state["sslRequired"] == "none"
    assert state["updates"] == 1
    second = run()
    assert second.returncode == 0, second.stderr
    assert state["updates"] == 1


def test_credential_drift_fails_once_without_mutation_or_secret_logs(admin_endpoint):
    state, run = admin_endpoint
    state["reject_auth"] = True
    result = run()
    assert result.returncode == 1
    assert state["auth_calls"] == 1
    assert state["updates"] == 0
    assert state["sslRequired"] == "external"
    assert PASSWORD not in result.stdout + result.stderr


def test_success_response_without_matching_readback_fails(admin_endpoint):
    state, run = admin_endpoint
    state["ignore_update"] = True
    result = run()
    assert result.returncode == 1
    assert state["updates"] == 1
    assert state["sslRequired"] == "external"


def test_redirect_never_forwards_admin_credentials(admin_endpoint):
    state, run = admin_endpoint
    state["redirect"] = True
    result = run()
    assert result.returncode == 1
    assert state["auth_calls"] == 1
    assert state["forwarded_requests"] == 0
    assert state["updates"] == 0
