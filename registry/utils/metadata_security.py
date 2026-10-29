"""YARA scanning of registrant-supplied metadata, and what each read may see (#1811).

Registrants attach free-form ``metadata`` to servers, agents and skills. It can
carry text written to steer a model that reads it. This module:

1. Scans the whole metadata document with three of the YARA rule files that ship
   with cisco-ai-mcp-scanner (``prompt_injection``, ``tool_poisoning``,
   ``credential_harvesting``). No new patterns are written here.
2. Attributes each match to the metadata keys responsible, so one bad key does
   not cost the record its other metadata, and never disables the record.
3. Decides, per read surface, which keys may be returned:
   - model-facing reads (the mcpgw tools, semantic search, discovery) get only
     keys covered by a current clean verdict; flagged keys are removed even when
     the caller names them, and a missing, stale or failed verdict withholds all
     metadata (fail closed);
   - the search index gets the same treatment, from a fresh scan;
   - operator-facing REST reads get everything, plus the verdict.

Why the rules are compiled here instead of through ``mcpscanner.YaraAnalyzer``:
importing ``mcpscanner`` runs its package ``__init__``, which imports the
behavioural analyzer and with it LiteLLM. The registry deliberately does not
install LiteLLM and keeps mcp-scanner in a subprocess (issue #1826). The rule
files are located with ``importlib.util.find_spec``, which does not execute the
package, and compiled with yara-python, the same engine ``YaraAnalyzer`` uses.

The scanned bytes are the compact JSON serialization of the document followed
by its keys and string values as plain text, one per line. The plain-text
copy exists because JSON escapes control characters: an instruction split
across a newline is ``ignore\\nprevious`` in JSON, which a rule written with
``\\s+`` cannot match. Matched text is never logged or stored, only rule names
and key paths.
"""

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import ValidationError

from ..schemas.metadata_security import (
    MetadataKeyPath,
    MetadataScanFinding,
    MetadataScanVerdict,
    MetadataSecurityView,
)

logger = logging.getLogger(__name__)

EntityType = Literal["server", "agent", "skill"]

# The scanner whose rule files are used, and where they live inside it.
# Rules ship with cisco-ai-mcp-scanner; see
# https://github.com/cisco-ai-defense/mcp-scanner/tree/main/mcpscanner/data/yara_rules
SCANNER_DISTRIBUTION: str = "cisco-ai-mcp-scanner"
SCANNER_PACKAGE: str = "mcpscanner"
RULES_SUBDIRECTORY: tuple[str, ...] = ("data", "yara_rules")
RULE_NAMES: tuple[str, ...] = (
    "prompt_injection",
    "tool_poisoning",
    "credential_harvesting",
)

# mcp-scanner's YARA_THREAT_MAPPING (mcpscanner/threats/threats.py, 4.8.6)
# rates all three threat types HIGH. Copied rather than imported, because
# importing that module runs mcpscanner's package __init__ (see module docstring).
SEVERITY_BY_THREAT_TYPE: dict[str, str] = {
    "PROMPT INJECTION": "HIGH",
    "TOOL POISONING": "HIGH",
    "CREDENTIAL HARVESTING": "HIGH",
}
UNKNOWN_SEVERITY: str = "UNKNOWN"

# yara-python match timeout. Metadata is at most 10,240 bytes (PR 1 limit), so
# this only bounds a pathological rule; a timeout is a failed verdict.
MATCH_TIMEOUT_SECONDS: int = 5

# Fixed error codes stored on a failed verdict. Scanner exception text is
# never stored, since it could echo input.
ERROR_SCANNER_UNAVAILABLE: str = "scanner_unavailable"
ERROR_SCAN_TIMEOUT: str = "scan_timeout"
ERROR_SCAN_ERROR: str = "scan_error"
ERROR_INVALID_METADATA: str = "invalid_metadata"

UNAVAILABLE_RULESET: str = "unavailable"

# Statuses under which every key may reach a model.
_FULLY_VISIBLE_STATUSES: frozenset[str] = frozenset({"clean", "disabled"})


class MetadataScannerUnavailableError(RuntimeError):
    """Raised when the packaged YARA rules cannot be located or compiled."""


class _CompiledRules(NamedTuple):
    rules: Any  # yara.Rules
    ruleset: str


def _scanning_enabled() -> bool:
    """Read METADATA_SCAN_ENABLED.

    Imported lazily so the scan core (``scan_metadata``) can run without the
    registry's full configuration, e.g. from scripts/evaluate_metadata_scanner.py.
    """
    from ..core.config import settings

    return settings.metadata_scan_enabled


def _yara_module() -> Any:
    """Import yara-python on first use.

    Imported lazily because the auth server imports the repositories that call
    this module and does not install yara-python; it never scans.

    Raises:
        MetadataScannerUnavailableError: If yara-python is not installed.
    """
    try:
        import yara
    except ImportError as e:
        raise MetadataScannerUnavailableError(
            "yara-python is not installed; install it with `uv sync` to enable "
            "metadata scanning, or set METADATA_SCAN_ENABLED=false"
        ) from e
    return yara


def _rules_directory() -> Path:
    """Locate mcp-scanner's packaged rule directory without importing the package."""
    spec = importlib.util.find_spec(SCANNER_PACKAGE)
    if spec is None or not spec.submodule_search_locations:
        raise MetadataScannerUnavailableError(
            f"{SCANNER_DISTRIBUTION} is not installed; install it with `uv sync` "
            "to enable metadata scanning, or set METADATA_SCAN_ENABLED=false"
        )
    return Path(next(iter(spec.submodule_search_locations))).joinpath(*RULES_SUBDIRECTORY)


@lru_cache(maxsize=1)
def _load_rules() -> _CompiledRules:
    """Compile the three rule files once per process.

    Raises:
        MetadataScannerUnavailableError: If a rule file is missing or does not compile.
    """
    yara = _yara_module()
    directory = _rules_directory()
    paths = {name: directory / f"{name}.yara" for name in RULE_NAMES}
    try:
        digest = hashlib.sha256()
        for name in RULE_NAMES:
            digest.update(paths[name].read_bytes())
        rules = yara.compile(filepaths={name: str(path) for name, path in paths.items()})
        version = importlib.metadata.version(SCANNER_DISTRIBUTION)
    except (OSError, yara.Error, importlib.metadata.PackageNotFoundError) as e:
        raise MetadataScannerUnavailableError(
            f"Could not load metadata YARA rules from {directory}: {type(e).__name__}"
        ) from e
    ruleset = f"{SCANNER_DISTRIBUTION}=={version};sha256:{digest.hexdigest()[:16]}"
    logger.info(f"Loaded metadata YARA rules {list(RULE_NAMES)} ({ruleset})")
    return _CompiledRules(rules=rules, ruleset=ruleset)


def _iter_text(
    document: Any,
) -> Iterator[str]:
    """Yield every key and string value in a document, depth first, iteratively."""
    stack: list[Any] = [document]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, child in value.items():
                yield str(key)
                stack.append(child)
        elif isinstance(value, list | tuple):
            stack.extend(value)


def _scan_bytes(
    document: Any,
) -> bytes:
    """Return the bytes the rules are matched against (see module docstring)."""
    serialized = json.dumps(document, ensure_ascii=False, separators=(",", ":"), default=str)
    text = "\n".join(_iter_text(document))
    # errors="replace": an unpaired surrogate cannot be stored (PR 1 rejects it),
    # but a legacy record must still be scannable rather than crash the read.
    return f"{serialized}\n{text}".encode("utf-8", errors="replace")


def _matched_rules(
    rules: Any,
    document: Any,
) -> dict[str, str]:
    """Return {rule name: threat_type} for every rule that matches the document."""
    matches = rules.match(data=_scan_bytes(document), timeout=MATCH_TIMEOUT_SECONDS)
    return {match.rule: str(match.meta.get("threat_type", "UNKNOWN")) for match in matches}


def _attribution_units(
    metadata: dict[str, Any],
    entity_type: EntityType,
) -> list[tuple[MetadataKeyPath, Any]]:
    """Split metadata into the units a match can be attributed to.

    Top-level keys for servers and agents. For skills, ``author`` and ``version``
    plus each key under ``extra``, since ``extra`` holds all the free-form keys.
    """
    units: list[tuple[MetadataKeyPath, Any]] = []
    for key, value in metadata.items():
        if entity_type == "skill" and key == "extra" and isinstance(value, dict) and value:
            units.extend(([key, str(sub_key)], sub_value) for sub_key, sub_value in value.items())
        else:
            units.append(([str(key)], value))
    return units


def _attribute_findings(
    rules: Any,
    metadata: dict[str, Any],
    entity_type: EntityType,
    whole_document_matches: dict[str, str],
) -> list[MetadataScanFinding]:
    """Attribute each rule that matched the whole document to the keys responsible.

    Each unit is rescanned on its own, wrapped with its key so a key name can be
    the match. A rule that matched the whole document but no single unit spans
    keys, so every unit is flagged for it.
    """
    units = _attribution_units(metadata, entity_type)
    unit_matches = [(path, _matched_rules(rules, {path[-1]: value})) for path, value in units]
    findings = []
    for rule, threat_type in sorted(whole_document_matches.items()):
        keys = [path for path, matched in unit_matches if rule in matched]
        if not keys:
            keys = [path for path, _ in units]
        findings.append(
            MetadataScanFinding(
                rule=rule,
                threat_type=threat_type,
                severity=SEVERITY_BY_THREAT_TYPE.get(threat_type, UNKNOWN_SEVERITY),
                keys=keys,
            )
        )
    return findings


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _failed_verdict(
    digest: str,
    error: str,
    ruleset: str,
) -> MetadataScanVerdict:
    return MetadataScanVerdict(
        status="failed",
        scanned_at=_now_iso(),
        metadata_digest=digest,
        ruleset=ruleset,
        error=error,
    )


# The fixed slots of a stored skill metadata document. Their names are set by
# the registry, so an empty slot carries no registrant text; every key under
# ``extra`` is registrant text.
_SKILL_SCALAR_SLOTS: tuple[str, ...] = ("author", "version")


def _has_content(
    document: dict[str, Any] | None,
    entity_type: EntityType,
) -> bool:
    """True if the document holds any registrant-controlled text to scan.

    Key names are registrant text too, so for servers and agents any key at
    all is content, whatever its value: ``{"<injected instruction>": None}``
    must be scanned. A skill document always has the registry's own
    ``author``/``version``/``extra`` keys, so it has content when a slot is
    filled or ``extra`` has any key.
    """
    if not document:
        return False
    if entity_type != "skill":
        return True
    if any(document.get(slot) not in (None, "") for slot in _SKILL_SCALAR_SLOTS):
        return True
    extra_keys = set(document) - set(_SKILL_SCALAR_SLOTS) - {"extra"}
    return bool(document.get("extra")) or bool(extra_keys)


def _canonical_skill_metadata(
    metadata: Any,
) -> dict[str, Any] | None:
    """Return skill metadata in its stored {author, version, extra} shape.

    Mirrors the skill repository's legacy-flat-dict normalization without
    mutating the caller's dict, then validates through SkillMetadata so a write
    and a later read of the same metadata produce the same digest.
    """
    from ..schemas.skill_models import SkillMetadata

    if metadata is None:
        return None
    raw = metadata.model_dump(mode="json") if hasattr(metadata, "model_dump") else metadata
    if not isinstance(raw, dict):
        raise ValueError("skill metadata must be an object")
    if not raw:
        return None
    raw = dict(raw)
    if "extra" not in raw:
        raw = {"author": raw.pop("author", None), "version": raw.pop("version", None), "extra": raw}
    return SkillMetadata(**raw).model_dump(mode="json")


def _without_paths(
    document: dict[str, Any],
    paths: list[MetadataKeyPath],
) -> dict[str, Any]:
    """Return a copy of ``document`` with each key path removed (missing paths ignored)."""
    result = copy.deepcopy(document)
    for path in paths:
        node: Any = result
        for segment in path[:-1]:
            node = node.get(segment) if isinstance(node, dict) else None
        if isinstance(node, dict) and path:
            node.pop(path[-1], None)
    return result


def _parse_stored_verdict(
    stored: Any,
) -> MetadataScanVerdict | None:
    """Parse a stored verdict, returning None when it is missing or malformed."""
    if stored is None:
        return None
    if isinstance(stored, MetadataScanVerdict):
        return stored
    if not isinstance(stored, dict):
        return None
    try:
        return MetadataScanVerdict(**stored)
    except (ValidationError, TypeError):
        return None


def canonical_metadata(
    metadata: Any,
    entity_type: EntityType,
) -> dict[str, Any] | None:
    """Return the metadata document a verdict is computed over.

    Args:
        metadata: The record's metadata (a dict, or SkillMetadata for skills).
        entity_type: Which kind of record it belongs to.

    Returns:
        The canonical document, or None when the record has no metadata.

    Raises:
        ValueError: If the metadata is not an object (or not valid skill metadata).
    """
    if entity_type == "skill":
        try:
            return _canonical_skill_metadata(metadata)
        except ValidationError as e:
            raise ValueError("skill metadata does not match {author, version, extra}") from e
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    return metadata


def metadata_digest(
    document: Any,
) -> str:
    """Return the sha256 digest a verdict is bound to.

    Computed over sorted-key, ASCII-escaped compact JSON so it is stable across
    dict ordering and a round trip through MongoDB.
    """
    canonical = json.dumps(
        document if document is not None else {},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()


def current_ruleset() -> str:
    """Return the identifier of the loaded rule set, or 'unavailable'."""
    try:
        return _load_rules().ruleset
    except MetadataScannerUnavailableError:
        return UNAVAILABLE_RULESET


def scan_metadata(
    metadata: Any,
    entity_type: EntityType,
) -> MetadataScanVerdict:
    """Scan one record's metadata and return the verdict to store.

    Never raises: an unreadable document or a scanner problem produces a
    ``failed`` verdict, which model-facing reads treat as "withhold everything".

    Args:
        metadata: The record's metadata as it will be stored.
        entity_type: "server", "agent" or "skill".

    Returns:
        The verdict, bound to the digest of the canonical metadata.
    """
    try:
        document = canonical_metadata(metadata, entity_type)
    except ValueError:
        return _failed_verdict(metadata_digest(metadata), ERROR_INVALID_METADATA, current_ruleset())
    digest = metadata_digest(document)
    try:
        compiled = _load_rules()
    except MetadataScannerUnavailableError as e:
        logger.error(f"Metadata scan not run for {entity_type}: {e}")
        return _failed_verdict(digest, ERROR_SCANNER_UNAVAILABLE, UNAVAILABLE_RULESET)
    if not _has_content(document, entity_type):
        return MetadataScanVerdict(
            status="clean", scanned_at=_now_iso(), metadata_digest=digest, ruleset=compiled.ruleset
        )
    return _scan_document(compiled, document, entity_type, digest)


def _scan_document(
    compiled: _CompiledRules,
    document: dict[str, Any],
    entity_type: EntityType,
    digest: str,
) -> MetadataScanVerdict:
    """Run the rules over a non-empty canonical document."""
    yara = _yara_module()
    try:
        whole = _matched_rules(compiled.rules, document)
        findings = (
            _attribute_findings(compiled.rules, document, entity_type, whole) if whole else []
        )
    except yara.TimeoutError:
        logger.error(f"Metadata scan timed out for {entity_type}")
        return _failed_verdict(digest, ERROR_SCAN_TIMEOUT, compiled.ruleset)
    except (yara.Error, TypeError, ValueError, RecursionError) as e:
        logger.error(f"Metadata scan failed for {entity_type}: {type(e).__name__}")
        return _failed_verdict(digest, ERROR_SCAN_ERROR, compiled.ruleset)
    flagged = sorted({tuple(path) for finding in findings for path in finding.keys})
    if findings:
        logger.warning(
            f"Metadata scan flagged {len(flagged)} key(s) on a {entity_type}: "
            f"rules={[finding.rule for finding in findings]}"
        )
    return MetadataScanVerdict(
        status="flagged" if findings else "clean",
        flagged_keys=[list(path) for path in flagged],
        findings=findings,
        scanned_at=_now_iso(),
        metadata_digest=digest,
        ruleset=compiled.ruleset,
    )


def stamp_metadata_scan(
    document: dict[str, Any],
    entity_type: EntityType,
) -> None:
    """Set or clear ``metadata_scan`` on a document about to be written.

    Called by the server, agent and skill repositories on every write. A write
    that carries ``metadata`` gets a freshly computed verdict, replacing any
    supplied value; a write that does not carry ``metadata`` has any supplied
    ``metadata_scan`` removed, so a verdict can never be written on its own.

    Args:
        document: The document (or ``$set`` body) being written, modified in place.
        entity_type: "server", "agent" or "skill".
    """
    if "metadata" not in document or not _scanning_enabled():
        document.pop("metadata_scan", None)
        return
    document["metadata_scan"] = scan_metadata(document["metadata"], entity_type).model_dump(
        mode="json"
    )


def metadata_field_update(
    field: str,
    value: Any,
    entity_type: EntityType,
) -> dict[str, Any]:
    """Return the ``$set`` body for a single-field repository update.

    ``update_field("metadata", ...)`` writes the verdict alongside, so the two
    never disagree. ``metadata_scan`` cannot be written on its own.

    Args:
        field: The field being set.
        value: Its new value.
        entity_type: "server", "agent" or "skill".

    Returns:
        The ``$set`` body.

    Raises:
        ValueError: If ``field`` is ``metadata_scan``.
    """
    if field == "metadata_scan":
        raise ValueError("metadata_scan is computed by the registry and cannot be set directly")
    body: dict[str, Any] = {field: value}
    if field == "metadata":
        stamp_metadata_scan(body, entity_type)
    return body


def metadata_field_unset(
    field: str,
) -> dict[str, str]:
    """Return the ``$unset`` body for a single-field repository update.

    Unsetting ``metadata`` also unsets its verdict, so no verdict outlives the
    metadata it was computed for. ``metadata_scan`` cannot be unset on its own.

    Args:
        field: The field being removed.

    Returns:
        The ``$unset`` body.

    Raises:
        ValueError: If ``field`` is ``metadata_scan``.
    """
    if field == "metadata_scan":
        raise ValueError("metadata_scan is computed by the registry and cannot be set directly")
    body = {field: ""}
    if field == "metadata":
        body["metadata_scan"] = ""
    return body


def evaluate_metadata_security(
    metadata: Any,
    stored_verdict: Any,
    entity_type: EntityType,
) -> MetadataSecurityView:
    """Work out a record's metadata security state from its stored verdict.

    Args:
        metadata: The record's current metadata.
        stored_verdict: The record's ``metadata_scan`` value, possibly missing
            or malformed.
        entity_type: "server", "agent" or "skill".

    Returns:
        The view. A verdict whose digest does not match the current metadata is
        ``stale``; a missing one is ``unscanned``; a malformed one is ``failed``.
    """
    if not _scanning_enabled():
        return MetadataSecurityView(status="disabled")
    try:
        document = canonical_metadata(metadata, entity_type)
    except ValueError:
        return MetadataSecurityView(status="failed")
    if not _has_content(document, entity_type):
        return MetadataSecurityView(status="clean")
    if stored_verdict is None:
        return MetadataSecurityView(status="unscanned")
    verdict = _parse_stored_verdict(stored_verdict)
    if verdict is None:
        return MetadataSecurityView(status="failed")
    view = MetadataSecurityView(
        status=verdict.status,
        flagged_keys=verdict.flagged_keys,
        findings=verdict.findings,
        scanned_at=verdict.scanned_at,
        ruleset=verdict.ruleset,
    )
    if verdict.metadata_digest != metadata_digest(document):
        view.status = "stale"
    return view


def metadata_for_model(
    metadata: dict[str, Any] | None,
    view: MetadataSecurityView,
) -> dict[str, Any] | None:
    """Return the part of ``metadata`` a model may see under ``view``.

    ``metadata`` may already be narrowed by a ``metadata_fields`` projection;
    flagged keys are removed from whatever is left.

    Args:
        metadata: Metadata about to be returned to a model-facing caller.
        view: The record's view from ``evaluate_metadata_security``.

    Returns:
        The metadata unchanged (clean or scanning disabled), without its flagged
        keys (flagged), or None (unscanned, stale or failed: withhold all).
    """
    if metadata is None:
        return None
    if view.status in _FULLY_VISIBLE_STATUSES:
        return metadata
    if view.status == "flagged":
        return _without_paths(metadata, view.flagged_keys)
    return None


def metadata_for_index(
    metadata: Any,
    entity_type: EntityType,
) -> dict[str, Any] | None:
    """Return the metadata that may enter the search index, from a fresh scan.

    Scanning again rather than reading the stored verdict means the index can
    never hold metadata a verdict was not computed for, whatever path wrote it.

    Args:
        metadata: The record's metadata as stored.
        entity_type: "server", "agent" or "skill".

    Returns:
        The canonical metadata without flagged keys, or None when it must be
        excluded entirely (scan failed) or there is none.
    """
    if not _scanning_enabled():
        try:
            return canonical_metadata(metadata, entity_type)
        except ValueError:
            return None
    verdict = scan_metadata(metadata, entity_type)
    try:
        document = canonical_metadata(metadata, entity_type)
    except ValueError:
        return None
    view = MetadataSecurityView(status=verdict.status, flagged_keys=verdict.flagged_keys)
    return metadata_for_model(document, view)


def model_facing_metadata(
    metadata_to_return: dict[str, Any] | None,
    stored_metadata: Any,
    stored_verdict: Any,
    entity_type: EntityType,
) -> dict[str, Any] | None:
    """Filter metadata for a model-facing response.

    Args:
        metadata_to_return: What the response would carry, possibly already
            narrowed by a metadata_fields projection.
        stored_metadata: The record's full stored metadata, which the verdict
            digest is checked against.
        stored_verdict: The record's ``metadata_scan``.
        entity_type: "server", "agent" or "skill".

    Returns:
        ``metadata_to_return`` with flagged keys removed, or None when the
        record has no current verdict.
    """
    view = evaluate_metadata_security(stored_metadata, stored_verdict, entity_type)
    return metadata_for_model(metadata_to_return, view)


def sanitize_agent_card_for_model(
    card: dict[str, Any],
    stored_metadata: Any,
    stored_verdict: Any,
) -> None:
    """Make an agent card dict safe for a model-facing response, in place.

    Replaces ``metadata`` with its model-visible part (``{}`` when withheld)
    and removes the stored verdict, which is for operators.

    Args:
        card: The card about to be returned; its ``metadata`` may already be
            narrowed by a metadata_fields projection.
        stored_metadata: The agent's full stored metadata.
        stored_verdict: The agent's ``metadata_scan``.
    """
    card.pop("metadata_scan", None)
    card.pop("metadata_security", None)
    if "metadata" in card:
        card["metadata"] = (
            model_facing_metadata(card.get("metadata"), stored_metadata, stored_verdict, "agent")
            or {}
        )


def operator_metadata_security(
    stored_metadata: Any,
    stored_verdict: Any,
    entity_type: EntityType,
) -> dict[str, Any]:
    """Return the ``metadata_security`` annotation for operator-facing reads."""
    return evaluate_metadata_security(stored_metadata, stored_verdict, entity_type).model_dump(
        mode="json"
    )


def needs_rescan(
    metadata: Any,
    stored_verdict: Any,
    entity_type: EntityType,
) -> bool:
    """True if a record's stored verdict should be recomputed.

    A verdict needs recomputing when it is missing, malformed, failed, stale, or
    was produced by a different rule set than the one now loaded.
    """
    view = evaluate_metadata_security(metadata, stored_verdict, entity_type)
    if view.status == "disabled":
        return False
    if view.status in ("unscanned", "stale", "failed"):
        return True
    parsed = _parse_stored_verdict(stored_verdict)
    if parsed is None:
        try:
            return _has_content(canonical_metadata(metadata, entity_type), entity_type)
        except ValueError:
            return True
    return parsed.ruleset != current_ruleset()
