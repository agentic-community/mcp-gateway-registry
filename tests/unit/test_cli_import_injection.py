"""Untrusted server config must never reach a shell/interpreter code sink.

The CLI import pipeline (``cli/import_from_anthropic_registry.sh``) fetches server
definitions **unauthenticated** from a public registry, runs them through
``cli/anthropic_transformer.py``, and hands the generated config to
``cli/service_mgmt.sh``. Historically ``service_mgmt.sh`` spliced that JSON into
inline ``python3 -c`` source (``config = json.loads('''$config_json''')``) at
seven sites and built scanner commands run through ``eval``. A field value
containing ``'''``, ``$(...)``, backticks, ``${IFS}`` or other shell
metacharacters escaped the literal and executed as Python or shell — arbitrary
code execution in a process holding the full ``.env``.

These tests keep the config on a data-only channel (stdin for payloads, argv for
script-controlled selectors) and guard against the code-execution sinks
returning. They fail against the vulnerable scripts and pass against the fix.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

CLI_DIR = Path(__file__).resolve().parents[2] / "cli"
HELPER = CLI_DIR / "_service_config.py"
TRANSFORMER = CLI_DIR / "anthropic_transformer.py"
SHELL_SCRIPTS = (
    CLI_DIR / "service_mgmt.sh",
    CLI_DIR / "import_from_anthropic_registry.sh",
)

# Substrings that only appear when untrusted data is spliced into a code
# position: an interpreter literal fed a shell variable, or an eval of an
# assembled command string.
FORBIDDEN_SINK_PATTERNS = (
    "json.loads('''",  # triple-quoted interpolation of $config_json
    'python3 -c "',  # inline interpreter source that may interpolate shell vars
    'eval "$',  # eval of an assembled command string
)


def _run_helper(command: str, *args: str, stdin: str) -> subprocess.CompletedProcess:
    """Invoke the helper exactly as the shell does: payload on stdin, args on argv."""
    return subprocess.run(
        [sys.executable, str(HELPER), command, *args],
        input=stdin,
        capture_output=True,
        text=True,
    )


def _run_shell_pipeline(payload: str, command: str, *args: str) -> subprocess.CompletedProcess:
    """Mirror service_mgmt.sh's real call: ``printf '%s' "$config" | python3 helper ...``.

    The payload is passed as a positional bash argument (never interpolated into
    the script body), reproducing how the script sources it from a config file.
    """
    script = 'printf "%s" "$1" | "$2" "$3" "${@:4}"'
    return subprocess.run(
        ["bash", "-c", script, "_", payload, sys.executable, str(HELPER), command, *args],
        capture_output=True,
        text=True,
    )


# --- The code-execution sinks are gone ---------------------------------------


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_have_no_code_execution_sinks(script: Path) -> None:
    """No shell script may splice untrusted data into a code position."""
    text = script.read_text()
    for pattern in FORBIDDEN_SINK_PATTERNS:
        assert pattern not in text, f"{script.name} still contains sink pattern: {pattern!r}"


# --- Injection payloads are inert data, not code -----------------------------


def _injection_config(sentinel: Path) -> str:
    """A config whose proxy_pass_url would run ``touch <sentinel>`` under the old sink."""
    breakout = (
        "https://evil.example/mcp''' + __import__('os').system('touch " + str(sentinel) + "') + '''"
    )
    return json.dumps({"server_name": "evil", "path": "/evil", "proxy_pass_url": breakout})


def test_triple_quote_breakout_is_not_executed(tmp_path: Path) -> None:
    """A ``'''`` + Python breakout in a field must be treated as data."""
    sentinel = tmp_path / "pwned_validate"
    result = _run_shell_pipeline(_injection_config(sentinel), "validate")

    assert not sentinel.exists(), "config value executed as code (validate sink)"
    assert result.returncode == 0, result.stderr
    # The malicious value survives intact as a plain string field.
    emitted = json.loads(result.stdout.splitlines()[0])
    assert "__import__" in emitted["proxy_pass_url"]


def test_shell_metacharacter_payloads_are_not_executed(tmp_path: Path) -> None:
    """``$(...)``, backticks and ``${IFS}`` in fields must not spawn a shell."""
    sentinel = tmp_path / "pwned_shell"
    hostile = "https://a/mcp$(touch " + str(sentinel) + ")`touch " + str(sentinel) + "`${IFS}"
    config = json.dumps(
        {
            "server_name": "x",
            "path": "/x",
            "proxy_pass_url": "https://a/mcp",
            "description": hostile,
            "tags": [hostile],
        }
    )
    # Exercise every field-extraction command the add/test paths use.
    for command, args in (
        ("validate", ()),
        ("get", ("proxy_pass_url",)),
        ("get", ("description",)),
        ("get", ("tags",)),
        ("get", ("--omit-falsy", "headers")),
    ):
        _run_shell_pipeline(config, command, *args)
    assert not sentinel.exists(), "a field value spawned a shell"


def test_malformed_json_fails_closed() -> None:
    """Invalid JSON must exit non-zero rather than fall through."""
    result = _run_helper("validate", stdin="this is not json")
    assert result.returncode != 0
    assert "ERROR" in result.stdout


def test_non_object_json_fails_closed() -> None:
    """A JSON scalar/array is not a config and must be rejected."""
    result = _run_helper("validate", stdin="[1, 2, 3]")
    assert result.returncode != 0


def test_control_characters_in_path_rejected() -> None:
    """A newline in path would corrupt the two-line contract; reject it."""
    config = json.dumps({"server_name": "x", "path": "/foo\nbar", "proxy_pass_url": "https://a"})
    result = _run_helper("validate", stdin=config)
    assert result.returncode != 0
    assert "control characters" in result.stdout


def test_valid_config_still_validates() -> None:
    """A legitimate config passes and emits config JSON + service name."""
    config = json.dumps(
        {
            "server_name": "brave",
            "path": "/brave-search",
            "proxy_pass_url": "https://example.com/mcp",
            "tags": ["search"],
        }
    )
    result = _run_helper("validate", stdin=config)
    assert result.returncode == 0, result.stdout
    lines = result.stdout.splitlines()
    assert json.loads(lines[0])["server_name"] == "brave"
    assert lines[1] == "brave-search"


def test_get_omit_falsy_headers_empty() -> None:
    """Empty headers emit nothing (optional field skipped downstream)."""
    result = _run_helper("get", "--omit-falsy", "headers", stdin='{"headers": []}')
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_get_tags_empty_emits_list() -> None:
    """Empty tags still emit ``[]`` so the test path's comparison works."""
    result = _run_helper("get", "tags", stdin='{"tags": []}')
    assert result.stdout.strip() == "[]"


# --- The transformer validates remote fields and fails closed ----------------


def _run_transformer(
    input_obj: dict, out_file: Path, path: str = "/x"
) -> subprocess.CompletedProcess:
    in_file = out_file.parent / "in.json"
    in_file.write_text(json.dumps(input_obj))
    return subprocess.run(
        [
            sys.executable,
            str(TRANSFORMER),
            str(in_file),
            str(out_file),
            "--base-port",
            "8100",
            "--path",
            path,
        ],
        capture_output=True,
        text=True,
    )


def test_transformer_accepts_valid_record(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    record = {
        "server": {
            "name": "brave/search",
            "description": "ok",
            "remotes": [{"url": "https://example.com/mcp", "type": "streamable-http"}],
        }
    }
    result = _run_transformer(record, out)
    assert result.returncode == 0, result.stderr
    config = json.loads(out.read_text())
    assert config["proxy_pass_url"] == "https://example.com/mcp"
    assert config["path"] == "/x"


def test_transformer_rejects_non_http_proxy_url(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    record = {
        "server": {
            "name": "x",
            "description": "d",
            "remotes": [{"url": "file:///etc/passwd", "type": "x"}],
        }
    }
    result = _run_transformer(record, out)
    assert result.returncode != 0
    assert not out.exists()


def test_transformer_rejects_whitespace_in_proxy_url(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    sentinel = tmp_path / "pwned_transform"
    record = {
        "server": {
            "name": "x",
            "description": "d",
            "remotes": [{"url": f"http://a/mcp$(touch {sentinel})", "type": "x"}],
        }
    }
    result = _run_transformer(record, out)
    assert result.returncode != 0
    assert not out.exists()
    assert not sentinel.exists()


def test_transformer_rejects_control_char_in_name(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    record = {
        "server": {
            "name": "a\nb",
            "description": "d",
            "remotes": [{"url": "https://a/", "type": "x"}],
        }
    }
    result = _run_transformer(record, out)
    assert result.returncode != 0
    assert not out.exists()


# --- The transformer blocks SSRF and does not materialize secrets ------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9000/mcp",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/mcp",
        "http://[::1]:8100/mcp",
    ],
)
def test_transformer_rejects_ssrf_remote_url(tmp_path: Path, url: str) -> None:
    """A remote URL pointing at a non-public address is rejected (no SSRF)."""
    out = tmp_path / "out.json"
    record = {"server": {"name": "x", "description": "d", "remotes": [{"url": url, "type": "x"}]}}
    result = _run_transformer(record, out)
    assert result.returncode != 0, f"expected {url} to be rejected"
    assert not out.exists()


def test_transformer_allows_public_remote_url(tmp_path: Path) -> None:
    """A public remote URL is accepted and preserved."""
    out = tmp_path / "out.json"
    record = {
        "server": {
            "name": "x",
            "description": "d",
            "remotes": [{"url": "https://mcp.example.com/mcp", "type": "streamable-http"}],
        }
    }
    result = _run_transformer(record, out)
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["proxy_pass_url"] == "https://mcp.example.com/mcp"


def test_transformer_drops_attacker_controlled_auth_headers(tmp_path: Path) -> None:
    """A remote-declared auth header must not wire an attacker-chosen secret.

    The header placeholder name is attacker-controlled; preserving it as
    ``${SECRET_KEY}`` would let the enabled server egress a local secret to the
    registrant's backend. The transformer drops remote auth headers entirely and
    never resolves a placeholder to a live secret value.
    """
    out = tmp_path / "out.json"
    record = {
        "server": {
            "name": "x",
            "description": "d",
            "remotes": [
                {
                    "url": "https://mcp.example.com/mcp",
                    "type": "x",
                    "headers": [{"name": "Authorization", "value": "Bearer {secret_key}"}],
                }
            ],
        }
    }
    import os as _os

    env = dict(_os.environ, SECRET_KEY="super-secret-value-must-not-leak")
    in_file = out.parent / "in.json"
    in_file.write_text(json.dumps(record))
    result = subprocess.run(
        [sys.executable, str(TRANSFORMER), str(in_file), str(out), "--path", "/x"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(out.read_text())
    # No secret value materialized, no attacker-chosen env placeholder wired in.
    assert config["headers"] == []
    assert config["auth_scheme"] == "none"
    assert "super-secret-value-must-not-leak" not in out.read_text()
    assert "SECRET_KEY" not in out.read_text()


# --- The add path fails closed for ANY non-safe scan outcome -----------------


def test_add_does_not_register_when_scan_fails() -> None:
    """A non-clean scan must abort BEFORE register_service (never enable it).

    Registration auto-enables the server and the API has no atomic
    register-disabled path, so a scan-failed server must not be registered at
    all. Guards that the fail-closed gate precedes the register call and that the
    old register-then-disable toggle path is gone.
    """
    text = (CLI_DIR / "service_mgmt.sh").read_text()
    gate = text.index("was NOT registered because its security scan did not pass")
    register = text.index('run_mcp_command "register_service"')
    assert gate < register, "scan gate must precede registration"
    assert "servers/toggle" not in text, "register-then-disable toggle path must be gone"
    assert 'scan_exit_code" -ne 0' in text


def test_add_scan_scrubs_ambient_bearer_token() -> None:
    """The add-path scan must not forward the operator's ambient scan token.

    The proxy_pass_url being scanned is registrant/remote-controlled and unvetted
    at registration, so ``MCP_SCAN_BEARER_TOKEN`` is stripped from the scan
    subshell to prevent credential egress to an attacker-chosen endpoint.
    """
    text = (CLI_DIR / "service_mgmt.sh").read_text()
    assert "env -u MCP_SCAN_BEARER_TOKEN" in text


def test_verify_uses_literal_grep() -> None:
    """Server-name matching must be literal, not a regex/option.

    A registrant-controlled path like ``/[`` is a valid path but an invalid
    regex; an unescaped ``grep`` would error on it. The lookup uses
    ``grep -F ... --`` so the attacker-controlled value is a fixed string.
    """
    text = (CLI_DIR / "service_mgmt.sh").read_text()
    assert "grep -Fq -- " in text
    assert "grep -F -A2 -B2 -- " in text
