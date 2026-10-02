#!/usr/bin/env python3
"""Scan prose for the phrase-level LLM tells banned by the writing skill.

The writing skill (.claude/skills/writing/SKILL.md) lists a revision pass. A
prose rule you have to remember is a rule you skip under deadline, so the
phrase-level half of that pass lives here as a command. The shape-level tells
(parallel structure, paragraph pinning, summary beats, decorative bolding) still
need a human read; this script does not try to find them.

Every hit is a candidate, not a verdict: "robust" is fine inside a quotation and
"ultimately," is sometimes load-bearing. Read each one and decide. What you may
not do is leave a hit unexamined.

Usage:
    uv run python scripts/prose-scan.py FILE [FILE ...]
    uv run python scripts/prose-scan.py --strict FILE    # exit 1 on any hit

Exit codes:
    0  no hits, or hits found in advisory (default) mode
    1  hits found in --strict mode, or a file could not be read
"""

import argparse
import os
import re
import stat
import sys

# Bound untrusted hook input and reporting to prevent local resource exhaustion.
MAX_FILE_BYTES: int = 5 * 1024 * 1024
MAX_REPORTED_HITS_PER_FILE: int = 200
CONTEXT_BEFORE: int = 70
CONTEXT_AFTER: int = 50

# Phrase-level tells from the writing skill's ban lists. Keep this in sync with
# the "Sentence-shape tells" and "Extra rules for LLM prose" sections.
TELLS: dict[str, str] = {
    "significance flag": (
        r"\b(matters more than it looks|worth (noticing|knowing)|the part worth\b"
        r"|this is the important bit|importantly|it is not obvious|not a small thing)\b"
    ),
    "stage direction": (
        r"(^|\. )(Now look at|Watch what happens|Here is the thing|Consider "
        r"|Notice that|Let us walk through)"
    ),
    "closing flourish": (
        r"\b(that is the whole \w+|the rest is variations|end of story"
        r"|in conclusion|ultimately,)\b"
    ),
    "antithesis": (
        r"(is not (a|an|the) \w+, (it|that) is\b|, not something\b|, not what\b"
        r"|, and neither\b)"
    ),
    "corrective negation": (
        r"\b(is ?n[o']t (just |really )?about\b|is not being\b|are not being\b"
        r"|Not a \w+ (in|here),? but\b)"
    ),
    "contrasting pair": (
        r"\b(not just \w+,? but|less \w+,? more \w+"
        r"|did ?n[o']t \w+,? (we|they|it) \w+ed)\b"
    ),
    "reassurance clause": r", and (all (two|three|four|five)|each one|both) \w+",
    "connective padding": (
        r"\b(that said|with that in mind|having said that|it is also worth adding)\b"
    ),
    "hedge stack": r"\b(may possibly|might potentially|could perhaps|probably likely)\b",
    # Both the literal character and the HTML entity forms. An entity slips past
    # a character check but renders as the banned dash, and `&mdash;` inside an
    # inline SVG also breaks XML parsing, since only XML's five entities exist there.
    "em-dash": r"(—|&mdash;|&#8212;|&#x2014;)",
    "double hyphen dash": r"\w--\w",
    "throat clearing": (
        r"\b(it is important to note|it should be noted|in terms of|the fact that)\b"
    ),
    "praise": r"\b(great question|powerful and flexible|robust|seamless|game changer)\b",
    # Teaser openers. Only the dependency-reveal half is reliably catchable; a
    # bare count can be legitimate signposting, so this matches the count only
    # when it opens a paragraph with nothing after it but a comma or full stop.
    "teaser opener": (
        r"(and the (second|latter|other|first) \w*\s*(exists|is there|comes) because"
        r"|(?:^|<p>)(Two|Three|Four|Five) (things|reasons|parts|facts)[,.]"
        r"|There are (two|three|four|five) (things|reasons|parts)\b)"
    ),
}

_COMPILED: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE | re.MULTILINE) for name, pattern in TELLS.items()
}


def _line_of(
    text: str,
    offset: int,
) -> int:
    """Return the 1-indexed line number containing a character offset."""
    return text.count("\n", 0, offset) + 1


def _fragment(
    text: str,
    start: int,
    end: int,
) -> str:
    """Return a single-line excerpt around a match, for display."""
    left = max(0, start - CONTEXT_BEFORE)
    right = min(len(text), end + CONTEXT_AFTER)
    return " ".join(text[left:right].split())


def _terminal_safe(value: str) -> str:
    """Return an ASCII-only representation safe to print in a terminal."""
    return value.encode("unicode_escape").decode("ascii")


def _scan_text(
    text: str,
) -> list[tuple[int, str, str]]:
    """Return (line, tell_name, fragment) for every tell found in text."""
    hits: list[tuple[int, str, str]] = []
    for name, pattern in _COMPILED.items():
        for match in pattern.finditer(text):
            if len(hits) >= MAX_REPORTED_HITS_PER_FILE:
                hits.sort(key=lambda hit: hit[0])
                return hits
            hits.append(
                (
                    _line_of(text, match.start()),
                    name,
                    _fragment(text, match.start(), match.end()),
                )
            )
    hits.sort(key=lambda hit: hit[0])
    return hits


def _scan_file(
    path: str,
) -> list[tuple[int, str, str]] | None:
    """Scan one file. Returns None when the file cannot be read."""
    try:
        file_stat = os.lstat(path)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OSError("refusing to scan a non-regular file")
        if file_stat.st_size > MAX_FILE_BYTES:
            raise OSError(f"refusing to scan files larger than {MAX_FILE_BYTES} bytes")
        with open(path, encoding="utf-8") as handle:
            return _scan_text(handle.read())
    except (OSError, UnicodeDecodeError) as exc:
        print(
            f"{_terminal_safe(path)}: could not read ({_terminal_safe(str(exc))})", file=sys.stderr
        )
        return None


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan prose for the phrase-level LLM tells the writing skill bans.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Advisory: print hits, always exit 0 (what the pre-commit hook does)
    uv run python scripts/prose-scan.py docs/my-doc.md

    # Strict: exit 1 on any hit, for a generated document that must be clean
    uv run python scripts/prose-scan.py --strict .scratchpad/pr-1831/explainer.md
""",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Files to scan",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any tell is found (default: report and exit 0)",
    )
    return parser.parse_args()


def main() -> int:
    """Scan every file given and report the tells found."""
    args = _parse_args()

    total = 0
    unreadable = False
    for path in args.files:
        hits = _scan_file(path)
        if hits is None:
            unreadable = True
            continue
        for line, name, fragment in hits:
            print(
                f"{_terminal_safe(path)}:{line}: {_terminal_safe(name)}\n    ...{_terminal_safe(fragment)}..."
            )
        total += len(hits)

    if total:
        label = "must be fixed" if args.strict else "advisory"
        print(f"\n{total} hit(s) ({label})")
    else:
        print("no prose tells found")

    if unreadable:
        return 1
    return 1 if (args.strict and total) else 0


if __name__ == "__main__":
    sys.exit(main())
