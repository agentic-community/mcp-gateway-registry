"""Deterministic, collision-free vault-key canonicalization.

The vault key is ``(auth_method, user_id, provider, server_path)``.
The raw values are NOT safe to use verbatim as path/map segments:

- ``user_id`` varies by IdP. Cognito/Okta ``sub`` are UUID/alphanumeric (safe),
  but Keycloak ``preferred_username`` can contain spaces/accents, Entra UPNs
  contain ``@``/``+``, and **Auth0 ``sub`` is ``auth0|abc`` -- the ``|`` would
  collide with the Secrets Manager map delimiter and break OpenBao paths.**
- ``server_path`` contains ``/`` -- the OpenBao KV path separator.

The fix is ONE canonical encoding applied uniformly in every backend:
**Unicode-NFC-normalize, then base64url-encode (unpadded)** each segment so no
segment can contain a separator (``/`` or ``|``) or a Unicode look-alike. NFC
first prevents NFC-vs-NFD silent collisions on non-ASCII usernames.

base64url (RFC 4648 §5, unpadded) is used deliberately instead of
percent-encoding: its alphabet is ``[A-Za-z0-9_-]`` only, so an encoded segment
contains **no** ``/``, ``|``, ``%``, or ``=``. Percent-encoding is unsafe here
because the OpenBao HTTP client (hvac) re-quotes the path it is given, turning a
percent-encoded ``/`` (``%2F``) into ``%252F`` on the wire -- a path OpenBao
will not match (encoded slashes defeat its ACL/path routing), which broke every
vault read. base64url has no such reserved characters, so it survives hvac
re-quoting and OpenBao path routing unchanged.

All SecretStore backends MUST build keys through these helpers so they
agree byte-for-byte (asserted by the cross-store round-trip test).
"""

import base64
import unicodedata

# The map delimiter used by the one-secret-per-principal backends (Secrets
# Manager) to join (provider, server_path) into a single JSON key.
# Encoding guarantees "|" never appears inside an encoded segment, so this is
# an unambiguous split point.
MAP_KEY_DELIMITER = "|"


def encode_segment(value: str) -> str:
    """NFC-normalize then base64url-encode (unpadded) a single key segment.

    The result contains only ``[A-Za-z0-9_-]`` -- never ``/``, ``|``, ``%``, or
    ``=`` -- so it is safe as an OpenBao path segment (and survives hvac's URL
    re-quoting), a Secrets Manager name component, and a JSON map-key component.
    """
    normalized = unicodedata.normalize("NFC", value or "")
    return base64.urlsafe_b64encode(normalized.encode("utf-8")).decode("ascii").rstrip("=")


def decode_segment(encoded: str) -> str:
    """Inverse of ``encode_segment``: base64url-decode back to the raw value.

    Re-pads the (unpadded) input before decoding. Used when enumerating stored
    connections to recover the original provider/server_path values.
    """
    padded = encoded + "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")


def user_principal(auth_method: str, user_id: str) -> str:
    """Build the per-user namespace prefix ``enc(auth_method)/enc(user_id)``.

    This is the vault prefix under which all of one principal's connections
    live. ``LIST`` / iterate under it enumerates the principal's connections.
    Two segments joined by ``/``; each segment is separator-free after encoding.
    """
    return f"{encode_segment(auth_method)}/{encode_segment(user_id)}"


def map_key(provider: str, server_path: str) -> str:
    """Build the per-connection map key ``enc(provider)|enc(server_path)``.

    Used by the one-secret-per-principal backends to key the JSON map. Encoding
    guarantees the delimiter never appears inside either segment.
    """
    return f"{encode_segment(provider)}{MAP_KEY_DELIMITER}{encode_segment(server_path)}"


def openbao_path(
    prefix: str,
    auth_method: str,
    user_id: str,
    provider: str,
    server_path: str,
) -> str:
    """Build the OpenBao KV-v2 logical path for one connection.

    ``{prefix}/{enc(auth_method)}/{enc(user_id)}/{enc(provider)}/{enc(server_path)}``
    -- ``/`` only ever appears between encoded segments.
    """
    return (
        f"{prefix.strip('/')}/"
        f"{encode_segment(auth_method)}/{encode_segment(user_id)}/"
        f"{encode_segment(provider)}/{encode_segment(server_path)}"
    )
