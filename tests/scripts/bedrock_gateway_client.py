#!/usr/bin/env python3
"""Drive the Amazon Bedrock Converse API through the gateway's generic proxy.

Sibling of ``openai_gateway_client.py``: same gateway feature (PR #1714), a
different upstream. Bedrock accepts a long-term API key as a plain
``Authorization: Bearer`` header, so it exercises the identical caller-passthrough
path -- no SigV4, no boto3 -- and the request the gateway forwards is exactly:

    POST https://bedrock-runtime.<region>.amazonaws.com/model/<model-id>/converse
    Authorization: Bearer <bedrock api key>
    {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}

Credential split (both required; swapping them fails closed):

- ``.token``               -> ``X-Authorization``  gateway JWT authenticating ingress.
- ``.scratchpad/.bedrock`` -> ``Authorization``    the Bedrock API key the gateway
  forwards, admitted only because the entity registers ``Authorization`` as
  caller-OVERRIDABLE. Sending the gateway JWT in both trips the equal-token guard.

Modes:

- ``converse``  buffered hop: ``POST /model/<id>/converse``, prints the reply text.
- ``stream``    ``POST /model/<id>/converse-stream``, whose body is an AWS
                event-stream (length-prefixed binary frames, NOT SSE). Frames are
                decoded with ``botocore.eventstream`` and reported with
                time-to-first-chunk, chunk count, and the largest inter-chunk gap,
                so a buffered response fails instead of passing on a 200.

Examples::

    # register the backing entity once (the custom type must already exist)
    uv run python api/registry_management.py --registry-url http://localhost \\
        --token-file .token custom-proxy-create --type rest-endpoint \\
        --name bedrock-proxy --target-url https://bedrock-runtime.us-east-1.amazonaws.com \\
        --streaming true --auth-passthrough

    # then run the checks
    uv run python tests/scripts/bedrock_gateway_client.py --ensure-scope
    uv run python tests/scripts/bedrock_gateway_client.py --mode stream \\
        --model us.anthropic.claude-opus-4-8 --prompt "Count slowly from 1 to 10."

Neither secret is logged: both are read from files and only ever sent as headers.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

# tests/scripts is not a package (these are run directly, like call_mcp_tool.py),
# so make the sibling support module importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gateway_test_support as gw  # noqa: E402

logger = logging.getLogger(__name__)

API_KEY_ENV: str = "AWS_BEARER_TOKEN_BEDROCK"
DEFAULT_API_KEY_FILE: str = ".scratchpad/.bedrock"
DEFAULT_MODEL: str = "us.anthropic.claude-opus-4-8"
DEFAULT_PROMPT: str = "Hello"
DEFAULT_MAX_TOKENS: int = 128
# Backend substring identifying this client's entity during discovery.
TARGET_HINT: str = "bedrock-runtime"
# Converse and ConverseStream are both POST; this client uses no GET surface.
REQUIRED_VERBS: tuple[str, ...] = ("POST",)


def _converse_payload(
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Build a minimal Converse request body.

    Args:
        prompt: User text.
        max_tokens: Output cap, keeping test invocations cheap.

    Returns:
        The Converse JSON payload.
    """
    return {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }


def _request_headers(
    gateway_token: str,
    api_key: str,
) -> dict[str, str]:
    """Build the two-credential header set for a proxied Bedrock call.

    Args:
        gateway_token: The gateway JWT (ingress).
        api_key: The Bedrock API key (forwarded upstream).

    Returns:
        Headers for the gateway request.
    """
    return {
        "X-Authorization": f"Bearer {gateway_token}",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _reply_text(
    body: dict[str, Any],
) -> str:
    """Extract the assistant text from a Converse response.

    Args:
        body: Decoded Converse response.

    Returns:
        Concatenated text blocks, or "" when the shape is unexpected.
    """
    blocks = body.get("output", {}).get("message", {}).get("content", []) or []
    return "".join(block.get("text", "") for block in blocks if isinstance(block, dict)).strip()


def check_converse(
    base_url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> bool:
    """Call ``/converse`` through the gateway (buffered hop, POST authz).

    Args:
        base_url: Gateway client URL for the entity, no trailing slash.
        headers: Two-credential header set.
        model: Bedrock model id.
        prompt: User text.
        max_tokens: Output cap.
        timeout: Request timeout in seconds.

    Returns:
        True when Bedrock returns 200 with assistant text.
    """
    url = f"{base_url}/model/{model}/converse"
    started = time.monotonic()
    response = requests.post(
        url,
        json=_converse_payload(prompt, max_tokens),
        headers=headers,
        timeout=timeout,
    )
    elapsed = time.monotonic() - started

    if response.status_code != 200:
        logger.error("converse    : FAILED - HTTP %s", response.status_code)
        gw.explain_status(response.status_code, response.text)
        return False

    body = response.json()
    text = _reply_text(body)
    logger.info(
        "converse    : OK in %.2fs (stop=%s, tokens=%s) -> %r",
        elapsed,
        body.get("stopReason"),
        body.get("usage", {}).get("totalTokens"),
        text[:160],
    )
    return bool(text)


class _EventStreamReader:
    """Decode AWS event-stream frames into Converse stream events.

    Bedrock's ConverseStream body is ``application/vnd.amazon.eventstream``:
    length-prefixed binary frames, not SSE, so it needs a real framing decoder.
    ``botocore.eventstream`` ships one and botocore is already a dependency. If it
    is unavailable, decoding degrades to counting raw chunks -- the
    incremental-delivery timing (the property under test) still holds.
    """

    def __init__(self) -> None:
        """Initialize the frame buffer, degrading when botocore is absent."""
        self.available = True
        try:
            from botocore.eventstream import EventStreamBuffer
        except ImportError:  # pragma: no cover - environment dependent
            self.available = False
            logger.warning("botocore.eventstream unavailable; counting raw chunks only")
            return
        self._buffer = EventStreamBuffer()

    def feed(
        self,
        data: bytes,
    ) -> list[dict[str, Any]]:
        """Add received bytes and return any newly completed events.

        Args:
            data: Raw bytes from the response body.

        Returns:
            Decoded event payloads (empty while a frame is still incomplete).
        """
        if not self.available:
            return []
        self._buffer.add_data(data)
        events: list[dict[str, Any]] = []
        for message in self._buffer:
            try:
                events.append(json.loads(message.payload))
            except (ValueError, UnicodeDecodeError):
                continue
        return events


def check_stream(
    base_url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> bool:
    """Call ``/converse-stream`` and assert the body arrives incrementally.

    Times the first chunk, counts chunks, decodes event frames, and records the
    largest inter-chunk gap. One chunk (or a zero span across many) means the
    response was buffered somewhere -- the entity's ``proxy_streaming`` claim,
    nginx ``proxy_buffering``, or the hop -- so the check fails.

    Args:
        base_url: Gateway client URL for the entity, no trailing slash.
        headers: Two-credential header set.
        model: Bedrock model id.
        prompt: User text.
        max_tokens: Output cap.
        timeout: Request timeout in seconds.

    Returns:
        True when more than one chunk arrived over a non-zero span.
    """
    url = f"{base_url}/model/{model}/converse-stream"
    started = time.monotonic()
    reader = _EventStreamReader()

    with requests.post(
        url,
        json=_converse_payload(prompt, max_tokens),
        headers=headers,
        timeout=timeout,
        stream=True,
    ) as response:
        if response.status_code != 200:
            logger.error("stream      : FAILED - HTTP %s", response.status_code)
            gw.explain_status(response.status_code, response.text)
            return False

        content_type = response.headers.get("content-type", "")
        first_chunk_at: float | None = None
        last_chunk_at = started
        max_gap = 0.0
        chunks = 0
        events = 0
        pieces: list[str] = []
        stop_reason: str | None = None

        for chunk in response.iter_content(chunk_size=None):
            if not chunk:
                continue
            now = time.monotonic()
            if first_chunk_at is None:
                first_chunk_at = now
            else:
                max_gap = max(max_gap, now - last_chunk_at)
            last_chunk_at = now
            chunks += 1
            for event in reader.feed(chunk):
                events += 1
                delta = event.get("delta")
                if isinstance(delta, dict) and delta.get("text"):
                    pieces.append(delta["text"])
                if event.get("stopReason"):
                    stop_reason = event["stopReason"]

    if first_chunk_at is None:
        logger.error("stream      : FAILED - the stream produced no chunks")
        return False

    span = last_chunk_at - first_chunk_at
    logger.info(
        "stream      : %d chunks / %d events, first at %.2fs, span %.2fs, max gap %.3fs, "
        "content-type=%s, stop=%s",
        chunks,
        events,
        first_chunk_at - started,
        span,
        max_gap,
        content_type or "(none)",
        stop_reason,
    )
    logger.info("stream      : text -> %r", "".join(pieces).strip()[:160])
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
        description="Exercise gateway generic-proxy routing against Amazon Bedrock Converse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    gw.add_common_arguments(parser, API_KEY_ENV, DEFAULT_API_KEY_FILE)
    parser.add_argument(
        "--mode",
        choices=["converse", "stream", "all"],
        default="all",
        help="Which checks to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Bedrock model id or inference profile (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt text")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"inferenceConfig.maxTokens (default: {DEFAULT_MAX_TOKENS})",
    )
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
    except gw.GatewayClientError as exc:
        logger.error("%s", exc)
        return 2

    base_url = f"{args.registry_url.rstrip('/')}{client_path.rstrip('/')}"
    headers = _request_headers(token, api_key)
    logger.info("Bedrock base: %s", base_url)
    logger.info("Model       : %s", args.model)

    checks = {
        "converse": lambda: check_converse(
            base_url, headers, args.model, args.prompt, args.max_tokens, args.timeout
        ),
        "stream": lambda: check_stream(
            base_url, headers, args.model, args.prompt, args.max_tokens, args.timeout
        ),
    }
    selected = list(checks) if args.mode == "all" else [args.mode]
    return gw.run_checks(checks, selected)


if __name__ == "__main__":
    sys.exit(main())
