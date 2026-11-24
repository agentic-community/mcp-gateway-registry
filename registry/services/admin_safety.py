"""Intra-admin safety guards for user-management operations.

These helpers protect the administrator population from two self-inflicted
lockout classes that admin authorization alone does not prevent:

- **Self-deletion.** An admin accidentally (or under coercion) deleting their
  own account.
- **Last-admin removal.** Deleting or demoting the final administrator, which
  would leave the deployment with no one able to manage users, groups, or
  scopes.

The admin population is derived with the request-time admin rule itself
(:func:`registry.auth.dependencies._user_is_admin`, i.e. a mutating action
granted to ``all``), so a group is "admin-conferring" here for exactly the same
reason a request bearing it is treated as ``is_admin``. The broader
privileged-WRITE predicate (``scope_service._import_touches_privileged_scope``)
is deliberately NOT used: it also matches read-only ``all`` grants such as
``list_service: ["all"]``, and counting those holders as admins would let the
last real administrator be deleted or demoted.

Fail-closed: if the admin population cannot be determined (IdP/DB error, empty
result), the caller must treat the operation as removing the last admin and
DENY. A guard that can be silently skipped is equivalent to no guard.
"""

from __future__ import annotations

import logging

from ..auth.dependencies import _user_is_admin
from ..services import scope_service
from ..utils.iam_manager import get_iam_manager

logger = logging.getLogger(__name__)


class AdminSafetyError(Exception):
    """Raised when an admin-safety guard cannot be satisfied.

    Carries an HTTP-appropriate status code and a human-readable reason so the
    route layer can translate it into an HTTPException without re-deriving the
    failure mode.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _normalize(value: str | None) -> str:
    """Lower-case + strip a name for case-insensitive comparison.

    IdP usernames and group names are compared case-insensitively elsewhere in
    this module family (see the case-insensitive dedup in the user list/delete
    handlers), so we mirror that here to avoid a bypass where ``Admin`` and
    ``admin`` are treated as different principals.
    """
    return (value or "").strip().casefold()


def is_self_target(actor_username: str | None, target_username: str) -> bool:
    """Return True when the actor is operating on their own account.

    Comparison is case-insensitive and whitespace-insensitive. An empty or
    missing actor username never matches a target (fail closed: we cannot prove
    it is *not* self, but an unauthenticated actor should already have been
    rejected upstream, and refusing to match here avoids blocking legitimate
    deletes of a genuinely different user when the actor identity is unknown —
    the upstream admin gate remains the primary control).
    """
    actor = _normalize(actor_username)
    if not actor:
        return False
    return actor == _normalize(target_username)


async def _list_scope_groups() -> dict:
    """Read the scope catalogue (``scope_service.list_groups``), failing closed.

    Raises:
        AdminSafetyError: 503 if the catalogue cannot be read or reports an error.
    """
    try:
        groups = await scope_service.list_groups()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("admin_safety: failed to list scope groups: %s", exc)
        raise AdminSafetyError(
            status_code=503,
            detail="Unable to verify administrator population; operation refused",
        ) from exc

    if not isinstance(groups, dict) or groups.get("error"):
        logger.error("admin_safety: scope group listing unavailable or errored: %r", groups)
        raise AdminSafetyError(
            status_code=503,
            detail="Unable to verify administrator population; operation refused",
        )
    return groups


def _scope_confers_admin(ui_permissions: object) -> bool:
    """Return True if a stored scope's ui_permissions confer admin at request time.

    Mirrors ``registry.auth.dependencies.get_ui_permissions_for_user``, which
    only reads list values: a bare string such as ``"all"`` (or ``"wall"``,
    where ``"all" in "wall"`` is True) grants nothing to the user, so it must
    not make the guard count anyone as an admin either.
    """
    if not isinstance(ui_permissions, dict):
        return False
    listed = {
        action: resources
        for action, resources in ui_permissions.items()
        if isinstance(action, str) and isinstance(resources, list)
    }
    return _user_is_admin(listed)


def _admin_names_from_groups(groups: dict) -> set[str]:
    """Return the case-folded scope names and mappings that confer admin."""
    admin_names: set[str] = set()
    for scope_name, meta in groups.items():
        if not isinstance(scope_name, str) or not isinstance(meta, dict):
            continue
        group_mappings = meta.get("mappings") or []
        if _scope_confers_admin(meta.get("ui_scopes")):
            admin_names.add(_normalize(scope_name))
            for mapped in group_mappings:
                if isinstance(mapped, str):
                    admin_names.add(_normalize(mapped))
    return admin_names


def _admin_names_or_fail(groups: dict) -> set[str]:
    """Return the admin-conferring names in ``groups``, failing closed on none.

    Raises:
        AdminSafetyError: 503 if no scope in ``groups`` confers admin.
    """
    admin_names = _admin_names_from_groups(groups)
    if not admin_names:
        logger.error(
            "admin_safety: scope catalogue yielded ZERO admin-conferring groups "
            "(%d groups scanned); refusing to derive admin population",
            len(groups),
        )
        raise AdminSafetyError(
            status_code=503,
            detail="Unable to verify administrator population; operation refused",
        )
    return admin_names


async def resolve_admin_group_names() -> set[str]:
    """Return the set of group/scope names that confer admin, case-folded.

    A group confers admin when its ui_permissions grant an admin-conferring
    (mutating) action to ``all`` resources: the same rule the request-time
    ``is_admin`` check applies (``_user_is_admin``). Read-only ``all`` grants
    (``list_service``, ``get_agent``, ``health_check_service``) do not count.

    Raises:
        AdminSafetyError: If the scope catalogue cannot be read or yields no
            admin-conferring scope. Fails closed so callers do not proceed on
            an incomplete admin picture.
    """
    groups = await _list_scope_groups()
    admin_names = _admin_names_or_fail(groups)
    logger.debug("admin_safety: resolved %d admin-conferring group names", len(admin_names))
    return admin_names


def groups_confer_admin(
    user_groups: list[str] | None,
    admin_group_names: set[str],
) -> bool:
    """Return True if any of the user's groups confers admin.

    Matching is case-insensitive even though the request-time mapping
    (``map_cognito_groups_to_scopes``) is exact. The guard reads group names
    from the IdP user listing, which is not always the token's identifier:
    Entra's listing returns display names while its tokens and mappings carry
    object IDs, so the guard can only match an Entra admin through the scope
    name, whose case the operator chose independently. Exact matching there
    would hide real admins from the count.

    Args:
        user_groups: The group names attached to the user (from the IdP listing).
        admin_group_names: Case-folded set from :func:`resolve_admin_group_names`.
    """
    for group in user_groups or []:
        if isinstance(group, str) and _normalize(group) in admin_group_names:
            return True
    return False


async def _m2m_admin_identities(admin_group_names: set[str]) -> list[frozenset[str]]:
    """Return one alias-set per admin M2M service account from ``idp_m2m_clients``.

    For non-Keycloak IdPs (Okta/Auth0/PingFederate) an M2M client's group
    membership lives ONLY in the MongoDB ``idp_m2m_clients`` collection, not in the
    IdP user listing. The user-management list handler merges this collection, so
    the admin count must too — otherwise an M2M-only admin is invisible and the
    last-admin guard both under-counts and (with the empty-set fail-closed) can
    wrongly refuse legitimate operations. Mirrors the merge in
    ``registry.api.management_routes.management_list_users``.

    Each admin is returned as a frozenset of ALL its identifiers (normalized
    ``name`` and ``client_id``). ``management_delete_user`` resolves an M2M client
    by ``{"$or": [{"client_id": ...}, {"name": ...}]}`` and the guard is called
    with the raw path param (either key), so the guard must recognise a delete by
    EITHER — but the two aliases are ONE admin, so they must count as one (a flat
    set of identifiers would double-count and re-open the last-admin fail-open).
    For Keycloak ``name == client_id`` so the set collapses to one element.

    Best-effort: a datastore error is logged and treated as "no M2M admins found"
    rather than raised, because the IdP listing is the primary source and the
    empty-population guard in :func:`list_admin_identities` still fails closed if
    the combined result is empty.

    Args:
        admin_group_names: Case-folded admin-conferring group set.

    Returns:
        One alias frozenset per M2M client whose groups confer admin.
    """
    from ..repositories.documentdb.client import get_documentdb_client

    m2m_admins: list[frozenset[str]] = []
    try:
        db = await get_documentdb_client()
        collection = db["idp_m2m_clients"]
        cursor = collection.find({})
        docs = await cursor.to_list(length=None)
    except Exception as exc:
        logger.warning(
            "admin_safety: could not read idp_m2m_clients for admin count "
            "(M2M admins not included): %s",
            exc,
        )
        return m2m_admins

    for doc in docs or []:
        if not isinstance(doc, dict) or not _record_enabled(doc):
            continue
        if not groups_confer_admin(doc.get("groups"), admin_group_names):
            continue
        aliases = {_normalize(key) for key in (doc.get("name"), doc.get("client_id")) if key}
        if aliases:
            m2m_admins.append(frozenset(aliases))

    logger.debug("admin_safety: %d M2M admin account(s) found", len(m2m_admins))
    return m2m_admins


def _record_enabled(doc: dict) -> bool:
    """Return True if a group-store record is active.

    Mirrors ``auth_server.mongodb_groups_enrichment._is_record_enabled``: only
    an absent ``enabled`` or the boolean ``True`` is active. The auth server
    never grants groups from a disabled record, so a disabled admin record is
    not an administrator and must not count as "another admin remains".
    """
    return doc.get("enabled", True) is True


def user_group_fallback_active() -> bool:
    """Return True if the auth server reads ``idp_user_groups`` for this IdP."""
    from ..core.config import settings

    provider = (settings.auth_provider or "").lower()
    return provider in settings.idp_user_group_fallback_enabled_providers


async def _user_group_admin_usernames(admin_group_names: set[str]) -> set[str]:
    """Return normalized usernames whose ``idp_user_groups`` record confers admin.

    The auth server uses this collection as the group source for IdPs whose
    tokens carry no groups claim (PingFederate by default), so on those
    deployments it is part of the admin population. Skipped when the
    fallback is not enabled for the active provider, because the auth server
    then never reads it and its groups confer nothing.

    Best-effort like :func:`_m2m_admin_identities`: a datastore error is logged
    and yields no admins; the empty-population check still fails closed.
    """
    if not user_group_fallback_active():
        return set()

    from ..repositories.documentdb.client import get_documentdb_client

    try:
        db = await get_documentdb_client()
        docs = await db["idp_user_groups"].find({}).to_list(length=None)
    except Exception as exc:
        logger.warning(
            "admin_safety: could not read idp_user_groups for admin count "
            "(fallback-store admins not included): %s",
            exc,
        )
        return set()

    usernames: set[str] = set()
    for doc in docs or []:
        if not isinstance(doc, dict) or not _record_enabled(doc):
            continue
        username = doc.get("username")
        if username and groups_confer_admin(doc.get("groups"), admin_group_names):
            usernames.add(_normalize(username))

    logger.debug("admin_safety: %d idp_user_groups admin account(s) found", len(usernames))
    return usernames


async def list_admin_identities() -> list[frozenset[str]]:
    """Return one alias-set per current administrator.

    Each element is the set of case-folded identifiers by which one admin can be
    addressed: a single username for an IdP user, or ``{name, client_id}`` for an
    M2M service account (which a delete may target by either). Returning one entry
    PER ADMIN — rather than a flat set of identifiers — is what lets the guards
    both (a) recognise the target by any of its aliases and (b) count distinct
    admins correctly (a flat identifier set would count a two-alias M2M admin
    twice and re-open the last-admin fail-open).

    Cross-references the IdP/DB user listing, the MongoDB ``idp_m2m_clients``
    store and (when the auth server uses it) the ``idp_user_groups`` fallback
    store against the admin-conferring group set, so the population matches what
    the user-management list handler shows (M2M-only admins included). Disabled
    store records are skipped. Fails closed on any error enumerating the primary
    IdP listing (an IdP with no listing at all is not an error), and on an empty
    population.

    Raises:
        AdminSafetyError: If users/groups cannot be enumerated, or the resolved
            population is empty (treated as unverifiable, not a genuine zero).
    """
    admin_group_names = await resolve_admin_group_names()
    admins = await _collect_admin_identities(admin_group_names)

    # Fail closed on an empty admin population.
    if not admins:
        logger.error(
            "admin_safety: no administrator accounts found despite %d "
            "admin-conferring group(s); refusing (user listing likely incomplete)",
            len(admin_group_names),
        )
        raise AdminSafetyError(
            status_code=503,
            detail="Unable to verify administrator population; operation refused",
        )

    logger.debug("admin_safety: %d administrator account(s) currently present", len(admins))
    return admins


async def _collect_admin_identities(admin_group_names: set[str]) -> list[frozenset[str]]:
    """Return one alias-set per principal whose groups are in ``admin_group_names``.

    The enumeration behind :func:`list_admin_identities`, without its
    empty-population check, so a caller can ask "who would still be admin"
    under a proposed admin-group set.

    Raises:
        AdminSafetyError: 503 if the IdP user listing fails.
    """
    iam = get_iam_manager()
    try:
        users = await iam.list_users(max_results=1000, include_groups=True)
    except NotImplementedError:
        # PingFederate has no user listing; its human admins live only in
        # idp_user_groups, merged below. Any other error still fails closed.
        logger.info("admin_safety: IdP has no user listing; using idp_user_groups only")
        users = []
    except Exception as exc:
        logger.error("admin_safety: failed to list users for admin count: %s", exc)
        raise AdminSafetyError(
            status_code=503,
            detail="Unable to verify administrator population; operation refused",
        ) from exc

    # A set, so a user listed by the IdP AND in idp_user_groups counts once.
    admin_usernames: set[str] = set()
    for user in users or []:
        if not isinstance(user, dict):
            continue
        username = user.get("username") or user.get("id")
        if not username:
            continue
        if groups_confer_admin(user.get("groups"), admin_group_names):
            admin_usernames.add(_normalize(username))
    admin_usernames |= await _user_group_admin_usernames(admin_group_names)
    admins: list[frozenset[str]] = [frozenset({name}) for name in sorted(admin_usernames)]

    # Merge M2M admins (stored only in idp_m2m_clients for non-Keycloak IdPs) so
    # the count matches management_list_users and an M2M-only admin is not
    # invisible to the last-admin guard.
    admins.extend(await _m2m_admin_identities(admin_group_names))
    return admins


def _would_empty_admins(
    admins: list[frozenset[str]],
    target: str,
) -> bool:
    """Return True if removing ``target`` leaves zero administrators.

    ``target`` is a normalized identifier. An admin is "the target" if that
    identifier is one of its aliases; the removal empties the population only if
    NO OTHER admin (a different alias-set) remains. Counting distinct admins —
    not identifiers — is what prevents a two-alias M2M admin from masking itself.
    """
    is_target_admin = any(target in aliases for aliases in admins)
    if not is_target_admin:
        return False
    remaining = [aliases for aliases in admins if target not in aliases]
    return not remaining


async def assert_not_last_admin(target_username: str) -> None:
    """Refuse an operation that would remove the last administrator.

    Call this immediately before deleting an admin account, or before a group
    update that would strip a user of every admin-conferring group. It is a
    no-op when the target is not currently an admin, or when at least one other
    admin remains.

    Args:
        target_username: The account being removed/demoted.

    Raises:
        AdminSafetyError: If the target is the only remaining administrator, or
            the admin population cannot be determined (fail closed).
    """
    admins = await list_admin_identities()
    target = _normalize(target_username)

    if _would_empty_admins(admins, target):
        raise AdminSafetyError(
            status_code=409,
            detail=(
                "Refusing to remove the last administrator. Grant admin to "
                "another account before removing or demoting this one."
            ),
        )


async def would_remove_last_admin_via_groups(
    target_username: str,
    desired_groups: list[str] | None,
) -> None:
    """Refuse a group update that demotes the last administrator.

    Determines whether ``desired_groups`` still confers admin on the target. If
    the target is currently the sole admin and the new group set would strip
    their admin status, the update is refused.

    Args:
        target_username: The account whose groups are being replaced.
        desired_groups: The complete new group set the update would apply.

    Raises:
        AdminSafetyError: If applying ``desired_groups`` would leave zero
            administrators, or the admin population cannot be determined.
    """
    admin_group_names = await resolve_admin_group_names()

    # If the new group set still confers admin, the target stays an admin and no
    # lockout is possible from this update.
    if groups_confer_admin(desired_groups, admin_group_names):
        return

    admins = await list_admin_identities()
    target = _normalize(target_username)

    if _would_empty_admins(admins, target):
        raise AdminSafetyError(
            status_code=409,
            detail=(
                "Refusing to demote the last administrator. Grant admin to "
                "another account before removing admin from this one."
            ),
        )


async def _refuse_if_no_admin_remains(
    current_names: set[str],
    new_names: set[str],
    refusal: str,
) -> None:
    """Refuse a catalogue change that would leave zero administrators.

    A change that loses no admin-conferring name cannot demote anyone, so it
    returns without reading the IdP. Otherwise the change is refused unless at
    least one principal is still an admin under ``new_names``.

    Args:
        current_names: Today's admin-conferring names (never empty).
        new_names: The admin-conferring names after the change.
        refusal: The 409 detail, naming the operation being refused.

    Raises:
        AdminSafetyError: 409 with ``refusal`` if no administrator would
            remain; 503 if today's population cannot be verified.
    """
    if current_names <= new_names:
        return

    # Fails closed (503) if today's population cannot be verified.
    await list_admin_identities()
    remaining = await _collect_admin_identities(new_names) if new_names else []
    if not remaining:
        raise AdminSafetyError(status_code=409, detail=refusal)


async def assert_scope_update_keeps_an_admin(
    scope_name: str,
    ui_permissions: dict | None,
    group_mappings: list[str] | None,
) -> None:
    """Refuse a scope/group write that would leave zero administrators.

    Covers every whole-document write of a scope: PATCH, create and import all
    replace the stored document (``import_group`` upserts), so each one can
    strip an existing scope's admin grant. Recomputes the admin-conferring
    group names with ``scope_name`` replaced by its new ``ui_permissions`` and
    ``group_mappings``; see :func:`_refuse_if_no_admin_remains`.

    Args:
        scope_name: The scope document being written.
        ui_permissions: The ui_permissions the write will store.
        group_mappings: The group_mappings the write will store.

    Raises:
        AdminSafetyError: 409 if the write would leave no administrator; 503
            if the scope catalogue or the admin population cannot be
            determined.
    """
    groups = await _list_scope_groups()
    current_names = _admin_names_or_fail(groups)
    proposed = dict(groups)
    proposed[scope_name] = {"ui_scopes": ui_permissions or {}, "mappings": group_mappings or []}
    new_names = _admin_names_from_groups(proposed)
    await _refuse_if_no_admin_remains(
        current_names,
        new_names,
        f"Refusing to update '{scope_name}': no administrator would remain. "
        "Grant admin through another group before removing it from this one.",
    )


async def assert_scope_delete_keeps_an_admin(
    scope_name: str,
    idp_group_removed: bool,
) -> None:
    """Refuse a scope/group delete that would leave zero administrators.

    Deleting the scope document drops its grant. Deleting the IdP group as
    well removes every member's ``scope_name`` membership, so that name stops
    conferring admin through any other scope that maps it.

    Args:
        scope_name: The scope document (and IdP group) being deleted.
        idp_group_removed: Whether the same request deletes the IdP group.

    Raises:
        AdminSafetyError: 409 if the delete would leave no administrator; 503
            if the scope catalogue or the admin population cannot be
            determined.
    """
    groups = await _list_scope_groups()
    current_names = _admin_names_or_fail(groups)
    proposed = {name: meta for name, meta in groups.items() if name != scope_name}
    new_names = _admin_names_from_groups(proposed)
    if idp_group_removed:
        new_names.discard(_normalize(scope_name))
    await _refuse_if_no_admin_remains(
        current_names,
        new_names,
        f"Refusing to delete '{scope_name}': no administrator would remain. "
        "Grant admin through another group before deleting this one.",
    )


async def assert_record_patch_keeps_admin(
    target: str,
    patch_fields: dict,
) -> None:
    """Refuse a group-store record PATCH that would demote the last admin.

    For the ``idp_m2m_clients`` and ``idp_user_groups`` PATCH routes, whose
    services write only the fields present in the body
    (``model_dump(exclude_unset=True)``). Writing ``enabled`` as anything but
    ``True`` disables the record, which strips every group just like
    ``groups: []``. A body touching neither field cannot change admin status.

    Args:
        target: The client_id or username the record is keyed on.
        patch_fields: ``payload.model_dump(exclude_unset=True)``.

    Raises:
        AdminSafetyError: If the patch would leave zero administrators, or the
            admin population cannot be determined.
    """
    if "enabled" in patch_fields and patch_fields["enabled"] is not True:
        await would_remove_last_admin_via_groups(target, [])
    elif "groups" in patch_fields:
        await would_remove_last_admin_via_groups(target, patch_fields["groups"] or [])


async def desired_groups_grant_admin(desired_groups: list[str] | None) -> bool:
    """Return True if the desired group set would confer admin on a user.

    Used to emit a distinct audit event when a group update elevates an account
    to admin-tier.
    """
    admin_group_names = await resolve_admin_group_names()
    return groups_confer_admin(desired_groups, admin_group_names)
