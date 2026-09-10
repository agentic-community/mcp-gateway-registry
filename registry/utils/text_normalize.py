"""Text normalization helpers for similarity queries.

Used by the registration duplicate-check advisory
(:mod:`registry.services.duplicate_check_service`) to strip catalog
boilerplate out of a query before it is embedded.

Registry names share a structural suffix that carries no capability
signal. Sentence-embedding models weight tokens roughly in proportion
to how much of the string they fill, so on a short name that suffix
dominates: ``topic-mcp-server`` and ``category-mcp-server`` measure
0.76 cosine on ``all-MiniLM-L6-v2`` while sharing nothing but
``-mcp-server``, and 0.47 once it is gone (issue #1696).

Stripping happens on the query side only. Stored document embeddings
keep their raw text, so nothing has to be re-indexed; measurement on
the shipped model showed query-only stripping separates unrelated pairs
(max 0.3717) from true duplicates (min 0.7074) as cleanly as stripping
both sides would.
"""

import re

# Tokens that describe what kind of thing an entry is, not what it does.
# Every registry carries them, so they add similarity between entries
# that have nothing in common.
#
# `gateway` and `registry` are deliberately absent: they carry meaning in
# this catalog. Stripping them dropped two genuine near-duplicates from
# 0.6230 and 0.6810 to 0.4633 and 0.4518, turning both into misses.
BOILERPLATE_TOKENS: frozenset[str] = frozenset(
    {
        "mcp",
        "server",
        "servers",
        "tool",
        "tools",
        "skill",
        "skills",
        "agent",
        "agents",
        "api",
        "apis",
        "service",
        "services",
    }
)

# Split on any run of non-word characters or underscores, so
# 'topic-mcp-server', 'topic_mcp_server', and 'topic mcp server' all
# tokenize identically.
_TOKEN_SPLIT_PATTERN = re.compile(r"[\W_]+")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, dropping empty fragments."""
    return [token for token in _TOKEN_SPLIT_PATTERN.split(text.lower()) if token]


def strip_boilerplate_tokens(
    text: str,
    tokens: frozenset[str] = BOILERPLATE_TOKENS,
) -> str:
    """Drop catalog-boilerplate tokens from ``text``.

    Returns the surviving tokens joined by single spaces, lowercased.
    Punctuation is not preserved: the result is fed to an embedder, not
    shown to a user.

    Returns an empty string when every token is boilerplate, which the
    caller must treat as "no signal to compare" rather than as a query.
    """
    if not text:
        return ""
    return " ".join(token for token in _tokenize(text) if token not in tokens)


def distinct_token_count(text: str) -> int:
    """Count distinct word tokens in ``text``.

    The duplicate-check advisory uses this to refuse a query too thin to
    carry meaning. Distinct rather than total, so a name repeated in its
    own description does not read as two pieces of evidence.
    """
    return len(set(_tokenize(text)))
