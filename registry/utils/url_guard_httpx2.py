"""SSRF-guarded transport for the MCP SDK's ``httpx2`` client.

The MCP SDK 2.x runs on ``httpx2``, a separate package from ``httpx``, so the
``GuardedAsyncTransport`` in ``url_guard`` cannot be mounted on its clients. This
module reuses the same validate-resolve-pin logic (``_PinnedResolverMixin``) on an
``httpx2`` transport, so an SDK connection gets the same rebinding protection as
every other registry fetch: the connection goes only to an IP validated inside the
same transport call, for the request and for every redirect the SDK follows.

It lives apart from ``url_guard`` so that module, which the auth server imports,
does not need ``httpx2``.
"""

from __future__ import annotations

import logging
import ssl

import httpx2

from .url_guard import (
    SKILL_PROFILE,
    _delegate_kwargs_with_ca_bundle,
    _forward_proxy_ssl_context,
    _PinnedResolverMixin,
    _Profile,
)

logger = logging.getLogger(__name__)


def _build_httpx2_proxy(
    proxy_url: str,
) -> httpx2.Proxy:
    """Build the ``httpx2.Proxy`` a forward-proxy delegate connects through.

    Mirrors ``url_guard._build_proxy``: the CA bundle is attached only for an
    ``https://`` proxy, because httpcore refuses ``proxy_ssl_context`` for ``http``.

    Args:
        proxy_url: The forward proxy URL from the operator's configuration.

    Returns:
        An ``httpx2.Proxy`` carrying any userinfo as proxy auth.
    """
    context = _forward_proxy_ssl_context()
    proxy_is_https = proxy_url.lower().startswith("https://")
    return httpx2.Proxy(
        url=proxy_url,
        ssl_context=context if proxy_is_https and isinstance(context, ssl.SSLContext) else None,
    )


def _copy_request(
    request: httpx2.Request,
) -> httpx2.Request:
    """Return a request to send that shares the body but not the URL or headers.

    Args:
        request: The request httpx2 handed to the transport.

    Returns:
        A new ``httpx2.Request`` with the same method, URL, headers, body
        stream and extensions, safe to rewrite for pinning.
    """
    return httpx2.Request(
        request.method,
        request.url,
        headers=request.headers.copy(),
        stream=request.stream,
        extensions=dict(request.extensions),
    )


class GuardedHttpx2AsyncTransport(_PinnedResolverMixin, httpx2.AsyncHTTPTransport):
    """``httpx2`` async transport that validates and pins every request.

    Behaves like ``url_guard.GuardedAsyncTransport``: each request, and each
    redirect hop, is checked against ``guard_profile``, resolved once, and
    rewritten to connect to the validated IP while keeping the original ``Host``
    header and TLS SNI. When a forward proxy applies, the already-validated request
    is handed to a delegate transport built with the same TLS settings plus
    ``proxy=``, so the proxied path cannot skip validation.
    """

    def __init__(
        self,
        *,
        guard_profile: _Profile = SKILL_PROFILE,
        **kwargs: object,
    ) -> None:
        self._guard_profile = guard_profile
        self._delegate_kwargs = dict(kwargs)
        self._proxy_delegates: dict[str, httpx2.AsyncHTTPTransport] = {}
        super().__init__(**kwargs)  # type: ignore[arg-type]

    def _proxy_delegate(
        self,
        proxy_url: str,
    ) -> httpx2.AsyncHTTPTransport:
        """Return (creating once) the delegate transport for this proxy."""
        delegate = self._proxy_delegates.get(proxy_url)
        if delegate is None:
            delegate = httpx2.AsyncHTTPTransport(
                proxy=_build_httpx2_proxy(proxy_url),
                **_delegate_kwargs_with_ca_bundle(self._delegate_kwargs),  # type: ignore[arg-type]
            )
            self._proxy_delegates[proxy_url] = delegate
        return delegate

    async def handle_async_request(
        self,
        request: httpx2.Request,
    ) -> httpx2.Response:
        """Validate, pin and send one request.

        Raises:
            UrlValidationError: The target, or a redirect hop, is not permitted
                under the profile. The request is never sent.
        """
        # Pin a copy, never the caller's request: httpx2 stores the request it
        # passed in as ``response.request``, and the MCP SDK resolves and judges
        # a redirect against that URL. Rewriting it to the IP made a same-origin
        # redirect back to the hostname look like a different origin.
        outbound = _copy_request(request)
        routed = await self._route_request_async(outbound)  # type: ignore[arg-type]
        if routed.proxy_url is None:
            return await super().handle_async_request(routed.request)  # type: ignore[arg-type]
        delegate = self._proxy_delegate(routed.proxy_url)
        return await delegate.handle_async_request(routed.request)  # type: ignore[arg-type]

    async def aclose(self) -> None:
        """Close the direct pool and every proxy delegate."""
        for delegate in list(self._proxy_delegates.values()):
            try:
                await delegate.aclose()
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.debug("error closing a forward-proxy delegate", exc_info=True)
        self._proxy_delegates.clear()
        await super().aclose()
