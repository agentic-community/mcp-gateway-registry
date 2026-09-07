#!/usr/bin/env python3
"""Drive the OpenAI SDK through the gateway's generic proxy (PR #1714).

Exercises the gateway-proxy-any-resource feature end to end against a proxied
custom-entity record that fronts ``https://api.openai.com``: the official
``openai`` client is pointed at the entity's gateway client URL, so the SDK's own
``Authorization: Bearer <openai-key>`` header travels the caller-passthrough path.

Credential split (both required; swapping them fails closed):

- ``.token``           -> ``X-Authorization``  gateway JWT authenticating ingress.
- ``.scratchpad/.oai`` -> ``Authorization``    the backend key the gateway forwards,
  admitted only because the entity registers ``Authorization`` as caller-
  OVERRIDABLE (a FIXED ``Authorization`` value is rejected by the registry --
  operator-owned bearers belong in the egress credential vault). Sending the
  gateway JWT in both trips ``_assert_generic_authorization_not_gateway_cred``.

What each mode proves:

- ``models``  GET verb authz + buffered hop (``GET /v1/models``).
- ``chat``    POST verb authz + CSRF Bearer exemption + buffered hop.
- ``stream``  the signed ``streaming`` token claim: SSE chunks are reported with
              time-to-first-chunk, chunk count, and the largest inter-chunk gap,
              so an accidentally BUFFERED response (one chunk, or every chunk at
              the same instant) fails loudly instead of passing on a 200.

Prerequisites (verified and reported before the first OpenAI call):

1. ``GATEWAY_GENERIC_PROXY_ENABLED=true`` in ``.env``, stack rebuilt.
2. A proxied custom record with ``proxy_target_url=https://api.openai.com`` and an
   overridable ``Authorization`` upstream header. Inspect it with::

       uv run python api/registry_management.py --registry-url http://localhost \\
           --token-file .token custom-record-list --type rest-endpoint --json

3. A scope grant for the entity's authz key. The legacy ``methods:["all"]``
   wildcard deliberately does NOT authorize an HTTP verb, so the admin group needs
   an explicit rule; ``--ensure-scope`` writes it idempotently.

Examples::

    uv run python tests/scripts/openai_gateway_client.py --ensure-scope
    uv run python tests/scripts/openai_gateway_client.py --mode stream \\
        --prompt "Count slowly from 1 to 10."
    uv run python tests/scripts/openai_gateway_client.py \\
        --client-path /gateway/skill/skills/openai-proxy --mode chat

Neither secret is logged: the key goes straight to the SDK, the JWT only ever
travels as a header.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

# tests/scripts is not a package (these are run directly, like call_mcp_tool.py),
# so make the sibling support module importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gateway_test_support as gw  # noqa: E402

logger = logging.getLogger(__name__)

API_KEY_ENV: str = "OPENAI_API_KEY"
DEFAULT_API_KEY_FILE: str = ".scratchpad/.oai"
DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_PROMPT: str = "Say hello in exactly five words."
# Backend substring identifying this client's entity during discovery.
TARGET_HINT: str = "api.openai.com"
# GET /v1/models and POST /v1/chat/completions.
REQUIRED_VERBS: tuple[str, ...] = ("GET", "POST")


def build_client(
    base_url: str,
    api_key: str,
    gateway_token: str,
    timeout: int,
) -> Any:
    """Construct an OpenAI SDK client that speaks through the gateway.

    The SDK puts ``api_key`` in ``Authorization`` -- exactly the header the entity
    registers as caller-overridable -- while the gateway credential rides in
    ``X-Authorization``. Retries are disabled so a gateway 5xx surfaces instead of
    being masked by a retry.

    Args:
        base_url: OpenAI-compatible base, i.e. the gateway client URL + ``/v1``.
        api_key: The OpenAI API key (forwarded to the backend).
        gateway_token: The gateway JWT authenticating ingress.
        timeout: Per-request timeout in seconds.

    Returns:
        A configured ``openai.OpenAI`` instance.

    Raises:
        GatewayClientError: The ``openai`` package is not installed.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise gw.GatewayClientError(
            "The 'openai' package is required: uv pip install openai"
        ) from exc

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={"X-Authorization": f"Bearer {gateway_token}"},
        timeout=float(timeout),
        max_retries=0,
    )


def check_models(
    client: Any,
) -> bool:
    """Run ``GET /v1/models`` through the gateway (buffered hop, GET authz).

    Args:
        client: The configured OpenAI client.

    Returns:
        True on success.
    """
    started = time.monotonic()
    models = client.models.list()
    ids = [m.id for m in models.data][:5]
    logger.info(
        "models      : OK %d models in %.2fs (e.g. %s)",
        len(models.data),
        time.monotonic() - started,
        ", ".join(ids),
    )
    return True


def check_chat(
    client: Any,
    model: str,
    prompt: str,
) -> bool:
    """Run a non-streaming chat completion (buffered hop, POST authz + CSRF gate).

    Args:
        client: The configured OpenAI client.
        model: Model id.
        prompt: User prompt.

    Returns:
        True on success.
    """
    started = time.monotonic()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    content = (completion.choices[0].message.content or "").strip()
    logger.info("chat        : OK in %.2fs -> %r", time.monotonic() - started, content[:160])
    return True


def check_stream(
    client: Any,
    model: str,
    prompt: str,
) -> bool:
    """Run a streaming chat completion and report incremental-delivery evidence.

    Times the first chunk, counts chunks, and records the largest inter-chunk gap.
    A single chunk (or a zero span across many) means the response was buffered
    somewhere -- the entity's ``proxy_streaming`` claim, nginx ``proxy_buffering``,
    or the hop -- so the check fails instead of passing on a 200.

    Args:
        client: The configured OpenAI client.
        model: Model id.
        prompt: User prompt.

    Returns:
        True when more than one chunk arrived over a non-zero span.
    """
    started = time.monotonic()
    first_chunk_at: float | None = None
    last_chunk_at = started
    max_gap = 0.0
    chunks = 0
    pieces: list[str] = []

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        now = time.monotonic()
        if first_chunk_at is None:
            first_chunk_at = now
        else:
            max_gap = max(max_gap, now - last_chunk_at)
        last_chunk_at = now
        chunks += 1
        for choice in chunk.choices:
            if choice.delta and choice.delta.content:
                pieces.append(choice.delta.content)

    if first_chunk_at is None:
        logger.error("stream      : FAILED - the stream produced no chunks")
        return False

    span = last_chunk_at - first_chunk_at
    logger.info(
        "stream      : %d chunks, first at %.2fs, span %.2fs, max gap %.3fs -> %r",
        chunks,
        first_chunk_at - started,
        span,
        max_gap,
        "".join(pieces).strip()[:160],
    )
    if chunks <= 1 or span <= 0.0:
        logger.error(
            "stream      : FAILED - not incremental (chunks=%d, span=%.3fs). Check "
            "proxy_streaming on the entity and that nginx rendered 'proxy_buffering "
            "off' for this route.",
            chunks,
            span,
        )
        return False
    logger.info("stream      : OK (incremental delivery confirmed)")
    return True


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Exercise gateway generic-proxy routing with the OpenAI SDK.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    gw.add_common_arguments(parser, API_KEY_ENV, DEFAULT_API_KEY_FILE)
    parser.add_argument(
        "--mode",
        choices=["models", "chat", "stream", "all"],
        default="all",
        help="Which checks to run (default: all)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt for chat/stream")
    return parser.parse_args()


def main() -> int:
    """Resolve the target, verify prerequisites, and run the selected checks.

    Returns:
        0 when every selected check passed, 1 on a check failure, 2 on a
        prerequisite/configuration error.
    """
    args = _parse_args()
    gw.configure_logging(args.debug)

    try:
        token, api_key, client_path = gw.preflight(
            args,
            api_key_env=API_KEY_ENV,
            verbs=REQUIRED_VERBS,
            target_hint=TARGET_HINT,
            expect_streaming=args.mode in {"stream", "all"},
        )
        base_url = f"{args.registry_url.rstrip('/')}{client_path.rstrip('/')}/v1"
        logger.info("OpenAI base : %s", base_url)
        client = build_client(
            base_url=base_url,
            api_key=api_key,
            gateway_token=token,
            timeout=args.timeout,
        )
    except gw.GatewayClientError as exc:
        logger.error("%s", exc)
        return 2

    checks = {
        "models": lambda: check_models(client),
        "chat": lambda: check_chat(client, args.model, args.prompt),
        "stream": lambda: check_stream(client, args.model, args.prompt),
    }
    selected = list(checks) if args.mode == "all" else [args.mode]
    return gw.run_checks(checks, selected)


if __name__ == "__main__":
    sys.exit(main())
