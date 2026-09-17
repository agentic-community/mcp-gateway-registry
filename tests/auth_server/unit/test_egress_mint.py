"""Auth-server egress mint-path tests: canonical auth_method + nginx marker.

These exercise the two auth_server-side guards:
- _canonical_auth_method: cookie path 'session_cookie' -> session record 'oauth2'.
- _attach_mcp_proxy_token: only mints the egress-capable token when the nginx
  marker matches (when configured); the auth_method claim is stamped.
"""

import jwt as pyjwt
import pytest

from auth_server import server


class _FakeHeaders(dict):
    """Case-insensitive-ish header stub: tests pass exact-case keys."""

    def get(self, key, default=""):
        return super().get(key, default)


class _FakeRequest:
    def __init__(self, headers: dict):
        self.headers = _FakeHeaders(headers)


class _FakeResponse:
    def __init__(self):
        self.headers: dict = {}


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing-only-do-not-use")


@pytest.mark.unit
class TestCanonicalAuthMethod:
    def test_cookie_maps_to_session_record_value(self):
        vr = {"method": "session_cookie", "data": {"auth_method": "oauth2"}}
        assert server._canonical_auth_method(vr) == "oauth2"

    def test_cookie_defaults_oauth2(self):
        assert server._canonical_auth_method({"method": "session_cookie", "data": {}}) == "oauth2"

    def test_idp_provider_methods_canonicalize_to_oauth2(self):
        # A bearer issued directly by the per-user IdP (what a DCR client like
        # Claude Code / Codex presents) reports the provider name as `method`. It
        # MUST fold into the same `oauth2` bucket the cookie-consent path wrote --
        # otherwise the DCR vend misses the vault and the user loops on consent.
        for method in ("keycloak", "entra", "cognito", "okta", "auth0", "pingfederate"):
            assert server._canonical_auth_method({"method": method}) == "oauth2", method

    def test_network_trusted_passthrough(self):
        # Non-per-user methods are NOT folded -- they pass through so the vend's
        # is_per_user_auth_method() check still rejects them.
        assert server._canonical_auth_method({"method": "network-trusted"}) == "network-trusted"
        assert server._canonical_auth_method({"method": "federation-static"}) == "federation-static"
        assert server._canonical_auth_method({"method": "future-unknown"}) == "future-unknown"

    def test_self_signed_maps_to_inner_auth_method_claim(self):
        # A self-signed JWT (UI 'generate token', or the egress OAuth-facade
        # /token mint) reports method='self_signed' (the FORMAT) but carries the
        # principal's auth_method as an inner claim. The vault keys on the
        # principal method, so this MUST canonicalize to the claim -- else a user
        # who consents via a cookie session (bucket 'oauth2') and vends with a
        # minted token (would-be bucket 'self_signed') loops on consent forever.
        vr = {"method": "self_signed", "data": {"auth_method": "oauth2"}}
        assert server._canonical_auth_method(vr) == "oauth2"

    def test_self_signed_defaults_oauth2_when_claim_absent(self):
        assert server._canonical_auth_method({"method": "self_signed", "data": {}}) == "oauth2"


@pytest.mark.unit
class TestCanonicalEgressUserPaths:
    """The per-user egress vault id must resolve identically on the consent-write
    (cookie) and vend (bearer) paths, or the vaulted token is written under one
    id and looked up under another (permanent vend miss). It keys on the OIDC
    ``sub``, which is present in both id_tokens and access tokens across providers.

    NOTE: previously this class was also named ``TestCanonicalEgressUser`` -- the
    same name as the class further down -- so it was shadowed at import time and
    silently NOT collected. Renamed so both suites run.
    """

    def test_bearer_uses_data_sub(self):
        # Vend path: verified bearer claims land in ``data``; the sub wins.
        vr = {"username": "alice@example.com", "data": {"sub": "00000000-sub-alice"}}
        assert server._canonical_egress_user(vr) == "00000000-sub-alice"

    def test_cookie_uses_persisted_subject(self):
        # Consent-write path: the session carries the sub persisted at login as
        # ``subject`` (create_session), NOT ``sub``. It must resolve the same value.
        vr = {
            "method": "session_cookie",
            "username": "alice@example.com",
            "data": {"subject": "00000000-sub-alice", "auth_method": "oauth2"},
        }
        assert server._canonical_egress_user(vr) == "00000000-sub-alice"

    def test_consent_and_vend_agree_for_entra_shaped_result(self):
        # Entra: the browser id_token has preferred_username (email) but the DCR
        # client's access token does not -- yet both carry the same sub. Keying on
        # sub makes the two paths agree even though ``username`` differs.
        cookie_vr = {
            "method": "session_cookie",
            "username": "alice@contoso.com",  # from preferred_username / email
            "data": {"subject": "entra-oid-sub-123", "auth_method": "oauth2"},
        }
        bearer_vr = {
            "username": "entra-oid-sub-123",  # access token lacks preferred_username
            "data": {"sub": "entra-oid-sub-123"},
        }
        assert server._canonical_egress_user(cookie_vr) == server._canonical_egress_user(bearer_vr)

    def test_falls_back_to_username_when_no_sub(self):
        # Non-OIDC callers (no sub anywhere) keep their pre-existing bucket.
        vr = {"username": "svc-account", "data": {}}
        assert server._canonical_egress_user(vr) == "svc-account"

    def test_empty_when_nothing_present(self):
        assert server._canonical_egress_user({}) == ""

    def test_egress_user_honored_only_for_self_signed(self):
        # The gateway stamps egress_user (the OIDC sub) onto its OWN self_signed
        # USER token; there it wins over the token's username sub.
        vr = {
            "method": server.AUTH_METHOD_SELF_SIGNED,
            "username": "alice@example.com",
            "data": {"egress_user": "00000000-sub-alice", "sub": "alice@example.com"},
        }
        assert server._canonical_egress_user(vr) == "00000000-sub-alice"

    def test_egress_user_ignored_for_non_self_signed_token(self):
        # Trust boundary: egress_user is an identity-keying claim. From any token
        # this gateway did NOT mint (e.g. a raw IdP jwt), it must be ignored so an
        # external issuer cannot inject the vault key and vend a victim's token.
        # Resolution falls through to the token's real subject instead.
        vr = {
            "method": "jwt",
            "username": "attacker@example.com",
            "data": {"egress_user": "00000000-sub-victim", "sub": "00000000-sub-attacker"},
        }
        assert server._canonical_egress_user(vr) == "00000000-sub-attacker"

    def test_egress_user_ignored_when_method_absent(self):
        # No method marker -> not provably gateway-minted -> egress_user ignored.
        vr = {
            "username": "attacker@example.com",
            "data": {"egress_user": "00000000-sub-victim", "sub": "00000000-sub-attacker"},
        }
        assert server._canonical_egress_user(vr) == "00000000-sub-attacker"


@pytest.mark.unit
class TestAuditIdentityDisplay:
    """`_audit_identity_display` resolves the HUMAN-READABLE audit identity
    (email -> preferred_username -> upn -> username/sub), the counterpart to
    `_canonical_egress_user` (which resolves the stable sub for vault keying).
    The two must NOT converge: audit prefers readability, vault prefers stability.
    """

    def test_prefers_email_from_data(self):
        vr = {
            "username": "00000000-sub-alice",
            "data": {
                "email": "alice@example.com",
                "preferred_username": "alice",
                "sub": "00000000-sub-alice",
            },
        }
        assert server._audit_identity_display(vr) == "alice@example.com"

    def test_prefers_top_level_email_when_no_data_email(self):
        vr = {"username": "00000000-sub-alice", "email": "alice@example.com", "data": {}}
        assert server._audit_identity_display(vr) == "alice@example.com"

    def test_falls_back_to_preferred_username(self):
        vr = {
            "username": "00000000-sub-alice",
            "data": {"preferred_username": "alice@contoso.com", "sub": "00000000-sub-alice"},
        }
        assert server._audit_identity_display(vr) == "alice@contoso.com"

    def test_falls_back_to_upn_for_entra_access_token(self):
        # Entra ACCESS token (the OBO ingress token) omits both `email` and
        # `preferred_username` but carries `upn`; the Entra provider resolves
        # `username` to the opaque `sub`. Without `upn` in the chain the audit
        # USER showed the opaque sub. It must resolve to the UPN.
        vr = {
            "username": "00000000-sub-alice",
            "data": {
                "upn": "alice@example.com",
                "oid": "user-object-id",
                "tid": "tenant-id",
                "name": "Example User",
                "sub": "00000000-sub-alice",
            },
        }
        assert server._audit_identity_display(vr) == "alice@example.com"

    def test_upn_wins_over_a_sub_valued_username(self):
        # On the Entra bearer path the provider resolves `username` to the opaque
        # `sub`, so upn must be consulted BEFORE username or the readable identity
        # loses to the opaque one.
        vr = {
            "username": "00000000-sub-alice",
            "data": {"upn": "alice@example.com", "sub": "00000000-sub-alice"},
        }
        assert server._audit_identity_display(vr) == "alice@example.com"

    def test_non_string_claim_is_skipped_not_returned(self):
        # A multivalued IdP mapper can emit `upn` as a JSON array. The record
        # model types this field as `str`, so returning the list would raise
        # ValidationError inside the best-effort audit emit and silently DROP the
        # record. It must fall through to the next usable candidate instead.
        vr = {
            "username": "alice@example.com",
            "data": {"upn": ["alice@example.com"], "sub": "00000000-sub-alice"},
        }
        assert server._audit_identity_display(vr) == "alice@example.com"
        assert isinstance(server._audit_identity_display(vr), str)

    def test_falls_back_to_username_when_no_email_or_pref(self):
        # Session path: username is already the human handle.
        vr = {"username": "alice@example.com", "data": {"sub": "00000000-sub-alice"}}
        assert server._audit_identity_display(vr) == "alice@example.com"

    def test_falls_back_to_sub_when_only_sub(self):
        # Self-signed token whose readable claims are absent: sub, not anonymous.
        vr = {"data": {"sub": "00000000-sub-alice"}}
        assert server._audit_identity_display(vr) == "00000000-sub-alice"

    def test_anonymous_when_nothing_present(self):
        assert server._audit_identity_display({}) == "anonymous"

    def test_diverges_from_canonical_egress_user(self):
        # REGRESSION: the same Entra-shaped result must yield the READABLE email
        # for audit but the STABLE sub for the vault key -- they must not collapse
        # into one value (that divergence is the whole point of the split).
        vr = {
            "method": "session_cookie",
            "username": "alice@contoso.com",
            "email": "alice@contoso.com",
            "data": {"subject": "entra-oid-sub-123", "auth_method": "oauth2"},
        }
        assert server._audit_identity_display(vr) == "alice@contoso.com"
        assert server._canonical_egress_user(vr) == "entra-oid-sub-123"
        assert server._audit_identity_display(vr) != server._canonical_egress_user(vr)


@pytest.mark.unit
class TestAuditIdentityClaims:
    """`_audit_identity_claims` enriches an audit record with the durable and
    IdP identity claims (sub, oid+tid canonical id, upn, appid/azp) so an
    operator can correlate an opaque `sub` back to a user. It must
    NOT feed auth/vault/OBO decisions -- those stay on `_canonical_egress_user`.
    """

    def test_entra_obo_access_token_full_claims(self):
        vr = {
            "client_id": "client-app-id",
            "data": {
                "sub": "00000000-sub-alice",
                "oid": "user-object-id",
                "tid": "tenant-id",
                "upn": "alice@example.com",
                "name": "Example User",
                "azp": "client-app-id",
                "appid": "client-app-id",
            },
        }
        claims = server._audit_identity_claims(vr)
        assert claims == {
            "subject": "00000000-sub-alice",
            "canonical_id": "user-object-id@tenant-id",
            "principal_name": "alice@example.com",
            "object_id": "user-object-id",
            "tenant_id": "tenant-id",
            "app_id": "client-app-id",
        }

    def test_full_name_claim_is_never_captured(self):
        # Data minimisation (GDPR Art. 25(2)): `principal_name` already makes the
        # actor contactable, so the `name` claim must not be persisted -- storing
        # it would add a PII category to every audit record.
        vr = {"data": {"sub": "s", "name": "Example User", "upn": "alice@example.com"}}
        claims = server._audit_identity_claims(vr)
        assert "display_name" not in claims
        assert "Example User" not in claims.values()

    def test_cookie_session_subject_populates_durable_identity(self):
        # Cookie sessions persist the OIDC sub under `subject`, not `sub` (see
        # resolve_session). Without that hop every browser-originated record
        # stored a null durable identity while _canonical_egress_user resolved it.
        vr = {"username": "alice@example.com", "data": {"subject": "oidc-sub-abc"}}
        claims = server._audit_identity_claims(vr)
        assert claims["subject"] == "oidc-sub-abc"
        assert claims["canonical_id"] == "oidc-sub-abc"
        # Must agree with the vault-keying resolver on WHICH id is the durable one.
        assert claims["subject"] == server._canonical_egress_user(vr)

    def test_self_signed_bearer_records_the_oidc_sub_not_the_login_username(self):
        # REGRESSION: a gateway-minted USER token (Cursor/Claude bearer minted from
        # a browser login) sets `sub` = the LOGIN USERNAME and stamps the real OIDC
        # sub into `egress_user`. Without that hop the durable fields stored a
        # username in a field documented as an opaque OIDC sub -- duplicating
        # `username`, and making an exact-match lookup by the real sub return
        # nothing for that whole auth path.
        vr = {
            "method": server.AUTH_METHOD_SELF_SIGNED,
            "username": "alice",
            "data": {"sub": "alice", "egress_user": "1a2b3c-oidc-sub-of-alice"},
        }
        claims = server._audit_identity_claims(vr)
        assert claims["subject"] == "1a2b3c-oidc-sub-of-alice"
        assert claims["canonical_id"] == "1a2b3c-oidc-sub-of-alice"
        # The audit record must resolve the SAME human as the egress vault key, or
        # a record cannot be joined to the consent record for that human.
        assert claims["subject"] == server._canonical_egress_user(vr)

    def test_egress_user_is_ignored_from_a_foreign_issuer(self):
        # TRUST BOUNDARY: `egress_user` is identity-keying. From any token this
        # gateway did not mint, an attacker could otherwise stamp a victim's id
        # into the durable identity of their own audit records.
        vr = {
            "method": "entra",
            "username": "attacker",
            "data": {"sub": "attacker-sub", "egress_user": "victim-oidc-sub"},
        }
        claims = server._audit_identity_claims(vr)
        assert claims["subject"] == "attacker-sub"
        assert claims["canonical_id"] == "attacker-sub"
        assert "victim-oidc-sub" not in claims.values()
        # Same gate as the vault key, so the two cannot disagree about trust.
        assert claims["subject"] == server._canonical_egress_user(vr)

    def test_principal_name_falls_back_to_email_for_an_entra_v2_token(self):
        # PRODUCTION-OBSERVED: an Entra v2.0 access token carries neither `upn`
        # nor `preferred_username`, only `email`. Without email in the chain
        # `principal_name` was null on every v2.0 deployment even though a
        # readable handle existed in v1.0.
        #
        # The sub is deliberately mixed-case base64url, the real v2.0 shape: it
        # documents why the audit filter matches opaque ids by equality with a
        # one-directional lowercase fold, so a v2.0 sub must be pasted verbatim
        # (see _identity_search_clause).
        vr = {
            "username": "Ab3Kx9QmMfE7bTn4pLsWzYcHu1JdRoAiSeXvNkGqBw0",
            "data": {
                "sub": "Ab3Kx9QmMfE7bTn4pLsWzYcHu1JdRoAiSeXvNkGqBw0",
                "oid": "11111111-2222-4333-8444-555555555555",
                "tid": "99999999-8888-4777-8666-777777777777",
                "email": "azure@example.com",
            },
        }
        claims = server._audit_identity_claims(vr)
        assert claims["principal_name"] == "azure@example.com"
        # The readable display and the durable ids are unaffected.
        assert server._audit_identity_display(vr) == "azure@example.com"
        assert claims["canonical_id"] == (
            "11111111-2222-4333-8444-555555555555@99999999-8888-4777-8666-777777777777"
        )

    def test_principal_name_prefers_upn_over_email(self):
        # `upn` IS the principal name on Entra; email is a contact address that
        # may be an alias, so it must lose to both stronger claims.
        vr = {
            "data": {
                "sub": "s",
                "upn": "alice@corp.example.com",
                "preferred_username": "alice",
                "email": "alias@personal.example.com",
            }
        }
        assert server._audit_identity_claims(vr)["principal_name"] == "alice@corp.example.com"

    def test_principal_name_prefers_preferred_username_over_email(self):
        vr = {"data": {"sub": "s", "preferred_username": "alice", "email": "a@x.com"}}
        assert server._audit_identity_claims(vr)["principal_name"] == "alice"

    def test_principal_name_reads_the_top_level_email_copy(self):
        # Providers surface `email` at the top level as well as under `data`.
        vr = {"email": "alice@example.com", "data": {"sub": "s"}}
        assert server._audit_identity_claims(vr)["principal_name"] == "alice@example.com"

    def test_canonical_id_falls_back_to_sub_without_oid_tid(self):
        # Non-Entra token: no oid/tid, so the durable id is the sub itself.
        vr = {"data": {"sub": "keycloak-sub-1", "preferred_username": "alice"}}
        claims = server._audit_identity_claims(vr)
        assert claims["canonical_id"] == "keycloak-sub-1"
        assert claims["subject"] == "keycloak-sub-1"
        assert claims["principal_name"] == "alice"
        assert claims["object_id"] is None
        assert claims["tenant_id"] is None

    def test_app_id_comes_only_from_the_token_never_the_resolved_client_id(self):
        # The self-signed validator resolves client_id to the literal
        # "user-generated" sentinel (and otherwise to the GATEWAY's own client
        # id), neither of which is the calling app. app_id must stay empty rather
        # than record a misleading value operators would query on.
        vr = {"client_id": "user-generated", "data": {"sub": "s"}}
        assert server._audit_identity_claims(vr)["app_id"] is None

    def test_non_string_claims_degrade_to_none(self):
        # Same silent-drop hazard as the display resolver: a non-string claim must
        # degrade ITS OWN field to None, leaving the rest of the record intact.
        vr = {"data": {"sub": "s", "oid": ["o1", "o2"], "tid": 42, "upn": {"bad": 1}}}
        claims = server._audit_identity_claims(vr)
        assert claims["subject"] == "s"
        assert claims["object_id"] == "o1"  # first usable member of a multivalued claim
        assert claims["tenant_id"] == "42"  # numeric directory id stringified
        assert claims["principal_name"] is None  # unusable shape -> absent, not fatal
        assert claims["canonical_id"] == "o1@42"

    def test_bool_claim_is_rejected_not_stringified(self):
        # `isinstance(True, int)` is True, so without the explicit bool rejection a
        # claim of `true` would be stored as the string "True": an identity value
        # that looks real and identifies nobody.
        claims = server._audit_identity_claims({"data": {"sub": True, "oid": False}})
        assert claims["subject"] is None
        assert claims["object_id"] is None

    def test_list_claim_with_no_usable_member_degrades_to_none(self):
        # A multivalued mapper can emit a list whose members are all unusable.
        # Exhausting it must yield None rather than raising inside the
        # best-effort audit emit, which would drop the whole record.
        claims = server._audit_identity_claims({"data": {"oid": [None, {}, "  "], "sub": "s"}})
        assert claims["object_id"] is None
        assert claims["subject"] == "s"

    def test_whitespace_only_claim_degrades_to_none(self):
        # `"   "` is truthy, so without a strip it would be STORED as the record's
        # principal name: a value that identifies nobody while looking like it
        # does. An operator filtering on it finds nothing and cannot tell why.
        vr = {"data": {"sub": "s", "upn": "   ", "oid": "\t\n ", "tid": ""}}
        claims = server._audit_identity_claims(vr)
        assert claims["principal_name"] is None
        assert claims["object_id"] is None
        assert claims["tenant_id"] is None
        # The usable claim on the same token is unaffected.
        assert claims["subject"] == "s"

    def test_surrounding_whitespace_is_trimmed_not_stored(self):
        # Whitespace is not part of any identifier, and an untrimmed copy would
        # fail the equality lookup the opaque fields are searched by.
        vr = {"data": {"sub": "  sub-123  ", "upn": " alice@contoso.com\n"}}
        claims = server._audit_identity_claims(vr)
        assert claims["subject"] == "sub-123"
        assert claims["principal_name"] == "alice@contoso.com"

    def test_all_none_when_no_claims(self):
        # Nothing to surface: every field is None (the record fields are optional,
        # so records for such tokens are unchanged).
        assert server._audit_identity_claims({}) == {
            "subject": None,
            "canonical_id": None,
            "principal_name": None,
            "object_id": None,
            "tenant_id": None,
            "app_id": None,
        }

    def test_splats_into_both_audit_record_models(self):
        # The enrichment dict must be accepted verbatim by BOTH record shapes:
        # nested on Identity (registry_api / mcp_access) and flat on the
        # token_mint record, which has no identity block.
        from registry.audit.models import Identity, TokenMintAuditRecord

        vr = {"data": {"sub": "s", "oid": "o", "tid": "t", "upn": "u@x.com"}}
        claims = server._audit_identity_claims(vr)

        ident = Identity(
            username="u@x.com",
            auth_method="entra",
            credential_type="bearer_token",
            **claims,
        )
        assert (ident.subject, ident.canonical_id, ident.object_id) == ("s", "o@t", "o")
        assert ident.principal_name == "u@x.com"
        assert ident.tenant_id == "t"

        mint = TokenMintAuditRecord(
            request_id="req-1",
            username_hash="user_deadbeef",
            auth_method="entra",
            internal_caller="mcp-proxy",
            token_kind="user",
            token_path="obo_exchange",  # nosec B106 - audit metadata label
            outcome="success",
            **claims,
        )
        assert (mint.subject, mint.canonical_id, mint.object_id) == ("s", "o@t", "o")

    def test_canonical_id_is_capped_like_every_other_field(self):
        # REGRESSION: canonical_id was composed with an f-string from two
        # already-capped halves, so it could reach 2 * _MAX_AUDIT_CLAIM_LEN + 1
        # while every sibling field was bounded. It rides in the response header
        # nginx copies, where an oversized value fails the request outright.
        vr = {"data": {"sub": "s", "oid": "o" * 300, "tid": "t" * 300}}
        claims = server._audit_identity_claims(vr)
        assert len(claims["object_id"]) == server._MAX_AUDIT_CLAIM_LEN
        assert len(claims["tenant_id"]) == server._MAX_AUDIT_CLAIM_LEN
        assert len(claims["canonical_id"]) == server._MAX_AUDIT_CLAIM_LEN


@pytest.mark.unit
class TestAuditProvider:
    """`_audit_provider` records WHICH IdP authenticated the caller.

    PRODUCTION-OBSERVED: provider validators return `method` (e.g. "entra"), not
    `provider`, so reading validation_result["provider"] alone left
    identity.provider null on every bearer-authenticated audit record.
    """

    def test_top_level_provider_wins(self):
        vr = {"provider": "keycloak", "data": {"provider": "entra"}}
        assert server._audit_provider(vr) == "keycloak"

    def test_falls_back_to_the_session_provider(self):
        # The cookie session persists the provider at login.
        vr = {"data": {"provider": "pingfederate"}}
        assert server._audit_provider(vr) == "pingfederate"

    def test_falls_back_to_the_configured_idp(self, monkeypatch):
        # The Entra bearer path: no provider anywhere in the result, but the
        # gateway's configured IdP is the truth for every token it accepts.
        monkeypatch.setattr(server.settings, "auth_provider", "entra")
        assert server._audit_provider({"data": {"sub": "s"}, "method": "entra"}) == "entra"

    def test_none_when_nothing_identifies_the_idp(self, monkeypatch):
        # Stays optional rather than inventing a value.
        monkeypatch.setattr(server.settings, "auth_provider", "")
        assert server._audit_provider({}) is None


@pytest.mark.unit
class TestAuditIdentityHopClaim:
    """The mcp-proxy hop must attribute its OBO mint record from the SIGNED
    `audit_identity` claim, never from the raw ingress header it also receives.

    nginx forwards the client's Authorization/X-Authorization to that hop, but
    /validate authenticates a session cookie FIRST and only falls through to a
    bearer when no valid cookie exists -- so a cookie-authenticated request's
    bearer header is never signature-verified. Reading identity there let a
    caller choose what the audit trail recorded.
    """

    def test_round_trips_display_and_claims(self):
        vr = {
            "username": "00000000-sub-alice",
            "data": {
                "sub": "00000000-sub-alice",
                "oid": "user-object-id",
                "tid": "tenant-id",
                "upn": "alice@example.com",
                "appid": "client-app-id",
            },
        }
        display, fields = server._audit_identity_from_token(
            {"audit_identity": server._audit_identity_token_claim(vr)}
        )
        assert display == "alice@example.com"
        assert fields == server._audit_identity_claims(vr)

    def test_claim_omits_empty_values(self):
        # The claim rides in a response header nginx copies; absent claims must
        # not pad it with nulls.
        claim = server._audit_identity_token_claim({"data": {"sub": "s"}})
        assert claim == {"display": "s", "subject": "s", "canonical_id": "s"}

    def test_absent_claim_degrades_to_empty_not_to_header_identity(self):
        # Rolling deploy: an in-flight token can predate the claim. The caller
        # then falls back to the verified `sub` principal -- opaque but true.
        assert server._audit_identity_from_token({}) == ("", {})
        assert server._audit_identity_from_token({"audit_identity": "not-a-dict"}) == ("", {})

    def test_malformed_claim_values_degrade_per_field(self):
        display, fields = server._audit_identity_from_token(
            {"audit_identity": {"display": ["alice@example.com"], "subject": {"bad": 1}}}
        )
        assert display == "alice@example.com"
        assert fields["subject"] is None
        # Unknown keys are dropped: only the canonical field set is accepted.
        assert set(fields) == set(server._audit_identity_claims({}))

    def test_only_canonical_keys_are_accepted(self):
        _, fields = server._audit_identity_from_token(
            {"audit_identity": {"display": "d", "is_admin": True, "groups": ["admin"]}}
        )
        assert "is_admin" not in fields
        assert "groups" not in fields


def _decode(token: str) -> dict:
    return pyjwt.decode(
        token,
        "test-secret-key-for-testing-only-do-not-use",
        algorithms=["HS256"],
        audience="mcp-proxy",
        issuer="mcp-auth-server",
    )


@pytest.mark.unit
class TestAttachMcpProxyTokenMarker:
    def test_no_upstream_does_not_mint(self):
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest({}), resp, subject="alice", scopes=[], server_name="github-mcp"
        )
        assert "X-Internal-Token" not in resp.headers

    def test_empty_marker_mints_unconditionally(self, monkeypatch):
        # Function-level fallback only: an empty marker is rejected at startup
        # (Settings.__init__), so this state is unreachable in a running server.
        # Kept to pin the helper's branch behavior.
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "")
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest({"X-Resolved-Upstream": "https://u/mcp"}),
            resp,
            subject="alice",
            scopes=["repo"],
            server_name="github-mcp",
            auth_method="oauth2",
        )
        claims = _decode(resp.headers["X-Internal-Token"])
        assert claims["sub"] == "alice"
        assert claims["auth_method"] == "oauth2"
        assert claims["upstream_url"] == "https://u/mcp"

    def test_marker_enabled_and_matching_mints(self, monkeypatch):
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "s3cret")
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest(
                {"X-Resolved-Upstream": "https://u/mcp", "X-Validate-Source-Secret": "s3cret"}
            ),
            resp,
            subject="alice",
            scopes=[],
            server_name="github-mcp",
            auth_method="oauth2",
        )
        assert "X-Internal-Token" in resp.headers

    def test_marker_enabled_and_missing_does_not_mint(self, monkeypatch):
        # Direct :8888 caller (no nginx marker) gets no egress-capable token
        # even with a forged X-Resolved-Upstream.
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "s3cret")
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest({"X-Resolved-Upstream": "https://attacker.example/mcp"}),
            resp,
            subject="alice",
            scopes=[],
            server_name="github-mcp",
            auth_method="oauth2",
        )
        assert "X-Internal-Token" not in resp.headers

    def test_marker_enabled_and_mismatch_does_not_mint(self, monkeypatch):
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "s3cret")
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest(
                {"X-Resolved-Upstream": "https://u/mcp", "X-Validate-Source-Secret": "wrong"}
            ),
            resp,
            subject="alice",
            scopes=[],
            server_name="github-mcp",
            auth_method="oauth2",
        )
        assert "X-Internal-Token" not in resp.headers

    def test_egress_user_claim_is_stamped(self, monkeypatch):
        # The vend path reads egress_user off this token to key the vault, so the
        # canonical per-user id must ride along even when it differs from subject.
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "")
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest({"X-Resolved-Upstream": "https://u/mcp"}),
            resp,
            subject="alice@example.com",
            scopes=[],
            server_name="github-mcp",
            auth_method="oauth2",
            egress_user="00000000-sub-alice",
        )
        claims = _decode(resp.headers["X-Internal-Token"])
        assert claims["egress_user"] == "00000000-sub-alice"

    def test_audit_identity_claim_is_signed_into_the_token(self, monkeypatch):
        # The mcp-proxy hop attributes its OBO mint record from this claim. It is
        # signed here precisely so that hop never has to read identity from the
        # raw ingress header, which /validate may not have verified at all.
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "")
        vr = {
            "username": "00000000-sub-alice",
            "data": {
                "sub": "00000000-sub-alice",
                "oid": "user-object-id",
                "tid": "tenant-id",
                "upn": "alice@example.com",
            },
        }
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest({"X-Resolved-Upstream": "https://u/mcp"}),
            resp,
            subject="00000000-sub-alice",
            scopes=[],
            server_name="github-mcp",
            auth_method="oauth2",
            audit_identity=server._audit_identity_token_claim(vr),
        )
        claims = _decode(resp.headers["X-Internal-Token"])
        display, fields = server._audit_identity_from_token(claims)
        assert display == "alice@example.com"
        assert fields["canonical_id"] == "user-object-id@tenant-id"

    def test_audit_identity_does_not_disturb_the_egress_vault_key(self, monkeypatch):
        # REGRESSION (3LO): the egress vend keys the per-user token vault on the
        # `egress_user` claim of THIS token. If adding the audit claim shifted or
        # dropped it, every user's already-consented upstream token would be
        # written under one id and looked up under another -- a permanent vend miss
        # presenting as "0 tools" and a re-consent prompt. Mint with and without
        # the audit claim and require the vault-keying claims to be identical.
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "")
        vr = {
            "method": server.AUTH_METHOD_SELF_SIGNED,
            "username": "alice",
            "data": {"sub": "alice", "egress_user": "00000000-sub-alice"},
        }
        minted = {}
        for label, audit_identity in (
            ("without", None),
            ("with", server._audit_identity_token_claim(vr)),
        ):
            resp = _FakeResponse()
            server._attach_mcp_proxy_token(
                _FakeRequest({"X-Resolved-Upstream": "https://u/mcp"}),
                resp,
                subject="alice",
                scopes=["repo"],
                server_name="github-mcp",
                auth_method="oauth2",
                egress_user=server._canonical_egress_user(vr),
                audit_identity=audit_identity,
            )
            minted[label] = _decode(resp.headers["X-Internal-Token"])

        # The vault bucket the registry vend resolves must not move.
        for claim in ("egress_user", "auth_method", "sub", "server", "upstream_url"):
            assert minted["without"][claim] == minted["with"][claim], claim
        assert minted["with"]["egress_user"] == "00000000-sub-alice"
        # The audit claim is additive only.
        assert set(minted["with"]) - set(minted["without"]) == {"audit_identity"}

    def test_no_audit_identity_leaves_the_claim_absent(self, monkeypatch):
        # Callers with nothing to attribute (federation-static, network-trusted)
        # must not pad the header with an empty claim.
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "")
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest({"X-Resolved-Upstream": "https://u/mcp"}),
            resp,
            subject="federation-peer",
            scopes=[],
            server_name="github-mcp",
            auth_method="oauth2",
        )
        assert "audit_identity" not in _decode(resp.headers["X-Internal-Token"])


@pytest.mark.unit
class TestCanonicalEgressUser:
    """The vault-keying id must agree between the consent-write and vend paths."""

    def test_explicit_egress_user_claim_wins_over_username_sub(self):
        # A gateway-issued self-signed USER token carries sub=username but stamps
        # the OIDC sub as egress_user. Without egress_user winning, a Cursor/Claude
        # bearer would key the vault on the username and miss the browser-consented
        # token (the "0 tools" bug). egress_user MUST take precedence over sub --
        # but only on a self_signed token (a claim minted by this gateway); see
        # test_egress_user_ignored_for_non_self_signed_token for the trust boundary.
        vr = {
            "method": server.AUTH_METHOD_SELF_SIGNED,
            "data": {"egress_user": "oidc-sub-xyz", "sub": "alice@example.com"},
        }
        assert server._canonical_egress_user(vr) == "oidc-sub-xyz"

    def test_cookie_session_subject_used_when_no_egress_user(self):
        # Cookie path: session_data has no sub/egress_user, only the persisted
        # OIDC subject -> that is the canonical id.
        vr = {"data": {"subject": "oidc-sub-abc"}, "username": "alice@example.com"}
        assert server._canonical_egress_user(vr) == "oidc-sub-abc"

    def test_bearer_sub_used_when_no_egress_user(self):
        # OIDC bearer path: verified claims expose sub directly.
        vr = {"data": {"sub": "oidc-sub-idp"}}
        assert server._canonical_egress_user(vr) == "oidc-sub-idp"

    def test_username_fallback_preserves_non_oidc_behavior(self):
        vr = {"data": {}, "username": "svc-account"}
        assert server._canonical_egress_user(vr) == "svc-account"
