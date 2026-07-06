"""Guard test: the egress OAuth subprocess consumer must never log captured stdout.

`credentials-provider/oauth/generic_oauth_flow.py` emits the OAuth token JSON on
stdout as a deliberate parent<->child IPC channel; `egress_oauth.py` captures it
with subprocess.run(capture_output=True) and json.loads() the last line. Logging
`result.stdout` would write the access/refresh token to logs in clear text
(CodeQL py/clear-text-logging-sensitive-data). This test fails if any logger call
is fed the captured stdout, so the fix cannot silently regress.
"""

import re
from pathlib import Path

_SRC = (
    Path(__file__).resolve().parents[2]
    / "credentials-provider"
    / "oauth"
    / "egress_oauth.py"
)


def test_captured_stdout_is_never_logged() -> None:
    source = _SRC.read_text()
    # Any logger.<level>(...) call whose argument references result.stdout.
    offenders = re.findall(r"logger\.\w+\([^)]*result\.stdout[^)]*\)", source)
    assert not offenders, (
        "egress_oauth.py logs captured subprocess stdout (contains OAuth tokens) "
        f"in clear text: {offenders}"
    )


def test_stderr_only_diagnostics_still_logged() -> None:
    """stderr (diagnostic, no token) may still be logged for troubleshooting."""
    source = _SRC.read_text()
    assert "result.stderr" in source, "expected stderr to remain available for diagnostics"
