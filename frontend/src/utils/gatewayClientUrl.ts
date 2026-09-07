/**
 * Client-facing URL for a gateway-proxied entity.
 *
 * The registry stores `proxy_client_url` without a trailing slash (see
 * build_proxy_client_path in registry/schemas/proxy_mixin.py), but the nginx
 * location it generates ends in "/". nginx answers a request for the bare path
 * with a 301 that adds the slash, and most HTTP clients downgrade a POST to GET
 * when following a 301 -- so a URL we hand a user to copy must already carry it,
 * or a copied base URL silently loses its method and body on a write.
 *
 * @param clientUrl - Server-derived client path, e.g. "/gateway/skill/pdf".
 * @returns Absolute same-origin URL with exactly one trailing slash, or "" when
 *   the entity is not proxied (no client path).
 */
export const gatewayClientUrl = (clientUrl: string | null | undefined): string => {
  if (!clientUrl) return '';
  return `${window.location.origin}${clientUrl.replace(/\/+$/, '')}/`;
};
