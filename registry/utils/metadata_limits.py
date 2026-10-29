"""Size and nesting limits for registrant-supplied metadata (issue #1811).

Registrants attach free-form ``metadata`` to servers, agents and skills, and that
document reaches a model's context through the discovery tools. These limits
bound it at every write path so one record cannot flood a listing, an embedding
or a model context.

The two limits:

- ``MAX_METADATA_BYTES``: the metadata serialized as compact JSON, encoded as
  UTF-8, must be at most this many bytes. This is the 10 KB cap already used for
  registry card metadata in ``registry/schemas/registry_card.py``.
- ``MAX_METADATA_DEPTH``: the metadata may nest at most this many containers
  deep. The top-level object is depth 1, so ``{"a": {"b": 1}}`` has depth 2.
  Lists count as a level, the same as objects.

There is deliberately no key-count limit: the byte limit already bounds it.

Size is measured on compact JSON (``separators=(",", ":")``) with
``ensure_ascii=False``, so a non-ASCII character counts as its real UTF-8 width
(an accented letter is 2 bytes, most emoji 4) rather than as a 6-byte ``\\uXXXX``
escape. Whitespace a client happens to send does not count against the limit.

Two malformed inputs are reported as limit errors rather than crashing the
caller:

- Text that is not valid Unicode. JSON lets a client send an unpaired UTF-16
  surrogate (``"\\ud800"``), which Python decodes into a string that cannot be
  encoded as UTF-8, so it can be neither measured nor stored.
- A JSON string nested so deeply that the decoder gives up with
  ``RecursionError`` before any depth check can run (on Python versions with
  that decoder limit). ``parse_metadata_json`` turns this into the depth error;
  if decoding succeeds, ``validate_metadata_limits`` enforces the same cap.
"""

import json
from typing import Any

MAX_METADATA_BYTES: int = 10240
MAX_METADATA_DEPTH: int = 8


class MetadataLimitError(ValueError):
    """Raised when metadata breaks the size or the nesting limit.

    Subclasses ``ValueError`` so a raise inside a Pydantic validator becomes an
    ordinary validation error, which FastAPI returns as a 422.
    """


def _container_children(
    value: Any,
) -> list[Any]:
    """Return the nested values of a dict or list, or [] for a scalar."""
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, list | tuple):
        return list(value)
    return []


def metadata_serialized_size(
    metadata: Any,
) -> int:
    """Return the size in bytes of ``metadata`` as compact UTF-8 JSON.

    Args:
        metadata: The metadata document. None counts as empty.

    Returns:
        Number of bytes the compact JSON encoding takes.

    Raises:
        UnicodeEncodeError: If a key or value holds an unpaired surrogate.
            ``validate_metadata_limits`` reports this as a MetadataLimitError.

    Example:
        >>> metadata_serialized_size({"team": "é"})
        13
    """
    if metadata is None:
        return 0
    encoded = json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return len(encoded.encode("utf-8"))


def metadata_depth(
    metadata: Any,
) -> int:
    """Return how many containers deep ``metadata`` nests.

    A scalar is depth 0, ``{}`` and ``{"a": 1}`` are depth 1, and
    ``{"a": [{"b": 1}]}`` is depth 3. Walks iteratively so a hostile document
    cannot exhaust the Python stack.

    Args:
        metadata: The metadata document.

    Returns:
        The maximum nesting depth.
    """
    max_depth = 0
    stack: list[tuple[Any, int]] = [(metadata, 1)]
    while stack:
        value, depth = stack.pop()
        if not isinstance(value, dict | list | tuple):
            continue
        max_depth = max(max_depth, depth)
        stack.extend((child, depth + 1) for child in _container_children(value))
    return max_depth


def parse_metadata_json(
    raw: str,
) -> Any:
    """Decode a metadata JSON string, reporting runaway nesting as a limit error.

    The form-based registration routes receive metadata as a JSON string.
    ``json.loads`` recurses, so a document nested tens of thousands of levels
    deep raises ``RecursionError`` inside the decoder, before the depth limit
    could be checked. That is still a depth violation, so it is reported as one.

    Args:
        raw: The metadata JSON text.

    Returns:
        The decoded value. Callers still run ``validate_metadata_limits`` on it.

    Raises:
        json.JSONDecodeError: If ``raw`` is not valid JSON, unchanged so callers
            keep their existing invalid-JSON response.
        MetadataLimitError: If ``raw`` nests too deeply for the decoder to read.
    """
    try:
        return json.loads(raw)
    except RecursionError as e:
        raise MetadataLimitError(
            f"metadata nests too deeply to decode, over the limit of "
            f"{MAX_METADATA_DEPTH}. Flatten the nested objects or lists."
        ) from e


def validate_metadata_limits(
    metadata: Any,
    *,
    subject: str = "metadata",
) -> None:
    """Raise if ``metadata`` breaks the size or the nesting limit.

    The depth check runs first: ``json.dumps`` recurses, so measuring the size
    of a document nested past the interpreter's recursion limit would raise
    RecursionError instead of a clear limit error.

    Args:
        metadata: The metadata document. None and empty documents always pass.
        subject: How to name the document in the error message.

    Raises:
        MetadataLimitError: Naming the limit that was broken, the limit and the
            measured value, or saying the text is not valid Unicode. The message
            never echoes the offending text.
    """
    if not metadata:
        return

    depth = metadata_depth(metadata)
    if depth > MAX_METADATA_DEPTH:
        raise MetadataLimitError(
            f"{subject} nests {depth} levels deep, over the limit of "
            f"{MAX_METADATA_DEPTH}. Flatten the nested objects or lists."
        )

    try:
        size = metadata_serialized_size(metadata)
    except UnicodeEncodeError as e:
        raise MetadataLimitError(
            f"{subject} contains text that is not valid Unicode (an unpaired "
            "UTF-16 surrogate such as \\ud800). Remove or correct it."
        ) from e
    if size > MAX_METADATA_BYTES:
        raise MetadataLimitError(
            f"{subject} is {size} bytes serialized as compact UTF-8 JSON, over the "
            f"limit of {MAX_METADATA_BYTES} bytes. Remove or shorten keys, or store "
            "large values elsewhere and reference them by URL."
        )
