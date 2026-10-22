"""Embedding text kept as parts, so a failed embedding can be retried trimmed (#1917).

The indexers join a name, a description, tags, every tool (or A2A skill) name
and description, and metadata into one string with no length limit. A server
with many tools can exceed the model's input limit, the provider rejects it,
and the entry never gets a vector.

EmbeddingText holds the same text as three groups: header parts, (name,
description) items, and trailing parts. full() rebuilds today's string byte
for byte, so content hashes do not change and nothing re-embeds. trimmed() is
used only after a failed embedding call, for a retry: it keeps every item name
and shortens the longest descriptions first.

This module lives in registry/utils, not registry/embeddings, on purpose: the
registry.embeddings package imports numpy, and the auth server imports
search_repository without having numpy installed. Keep it free of heavy imports.
"""

from dataclasses import (
    dataclass,
    field,
)

# Each step shortens the description cap by 10 percent. Geometric steps reach
# a fitting cap in a few dozen joins even for thousands of tools, and land
# close to the largest cap that fits.
CAP_STEP_RATIO: float = 0.9


def _cut_at_word(
    text: str,
    limit: int,
) -> str:
    """Return text cut to at most limit characters, at the last space when there is one."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_space = cut.rfind(" ")
    if last_space > 0:
        return cut[:last_space]
    return cut


@dataclass
class EmbeddingText:
    """The text an indexer embeds, kept as parts so it can be trimmed.

    Attributes:
        head: Parts that always come first (name, description, tags).
        items: (name, description) pairs for tools or skills. Names are always kept.
            An item with an empty name contributes only its description.
        tail: Parts that come last (metadata, a list of tool names).
    """

    head: list[str] = field(default_factory=list)
    items: list[tuple[str, str]] = field(default_factory=list)
    tail: list[str] = field(default_factory=list)

    def _join(
        self,
        description_cap: int | None,
    ) -> str:
        """Join the parts, with every item description cut to description_cap if set."""
        parts: list[str] = list(self.head)
        for name, description in self.items:
            parts.append(name)
            if description_cap is None:
                parts.append(description)
            else:
                parts.append(_cut_at_word(description, description_cap))
        parts.extend(self.tail)
        return " ".join(part for part in parts if part)

    def full(self) -> str:
        """Return the text exactly as the indexers built it before #1917."""
        return self._join(None)

    def trimmed(
        self,
        max_chars: int,
    ) -> str:
        """Return the text shortened to at most max_chars characters.

        1. If the full text fits, return it unchanged.
        2. Otherwise lower one cap on every item description, longest first,
           cutting at word boundaries, until the text fits. Short descriptions
           stay whole; every item name is kept.
        3. If even names alone do not fit, cut the whole text at a word boundary.
        """
        text = self.full()
        if len(text) <= max_chars:
            return text
        cap = max((len(description) for _name, description in self.items), default=0)
        while cap > 0:
            cap = int(cap * CAP_STEP_RATIO)
            text = self._join(cap)
            if len(text) <= max_chars:
                return text
        return _cut_at_word(self._join(0), max_chars)
