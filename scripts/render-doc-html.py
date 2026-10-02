#!/usr/bin/env python3
"""Render a markdown document into a self-contained, styled HTML page.

The explainer skill hand-authors its HTML, which works for a short three-level
document with one bespoke diagram. A low-level design or a multi-persona review
runs to tens of thousands of words and dozens of tables, so hand-porting it
guarantees the two copies drift. This script renders the markdown instead, using
the explainer template's stylesheet so every generated document looks the same.

The output is self-contained by design: all CSS inline, no webfonts, no CDN, no
JavaScript, so it renders from file:// with no network. SVG diagrams are inlined
from files rather than linked, for the same reason.

Trust model: the markdown body, the byline derived from it, and every inlined SVG
are treated as untrusted. Raw HTML in the markdown is disabled, the byline is
escaped with an href scheme allowlist, and an SVG carrying a script, an event
handler, or an external reference aborts the render. This matters because the
pr-review and new-feature-design skills summarize GitHub-fetched PR and issue
bodies into these documents, and the generated page may be served from
127.0.0.1, an origin that can reach a locally running registry.

``--byline-html`` and ``--footer-html`` are the two deliberate exceptions: they
are inserted verbatim and are for first-party provenance text only. Never pass
issue, PR, or any other fetched metadata through them.

Diagrams: a fenced block tagged ``svg:<key>`` is replaced by the contents of
``<diagrams-dir>/<key>.svg``, wrapped in a figure. Any text on the fence line
after the key becomes the caption. The markdown keeps an ASCII version inside the
fence, so the document still reads in a terminal and a missing .svg file falls
back to that ASCII rather than losing the diagram.

Usage:
    uv run python scripts/render-doc-html.py DOC.md
    uv run python scripts/render-doc-html.py DOC.md -o out.html --title "My Design"
    uv run python scripts/render-doc-html.py DOC.md --byline-html '<em>draft</em>'

Exit codes:
    0  rendered successfully
    1  input, template, or diagram could not be read
"""

import argparse
import html
import logging
import pathlib
import re
import sys

from markdown_it import MarkdownIt

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE: pathlib.Path = REPO_ROOT / ".claude/skills/explainer/assets/template.html"
DEFAULT_DIAGRAMS_DIRNAME: str = "diagrams"
MAX_INPUT_BYTES: int = 8 * 1024 * 1024
MAX_TEMPLATE_BYTES: int = 2 * 1024 * 1024
MAX_SVG_BYTES: int = 1024 * 1024
MAX_SVG_TOTAL_BYTES: int = 8 * 1024 * 1024

# Schemes permitted on a byline href. Everything else renders as inert text, per
# the project's one-shared-URL-scheme-guard rule (see frontend/src/utils/safeUrl.ts
# and the "Frontend" invariant in AGENTS.md). A relative or fragment target has no
# scheme and is allowed.
_SAFE_URL_SCHEMES: frozenset[str] = frozenset({"http", "https", "mailto"})

# Constructs that make an inlined SVG executable or able to reach off-document.
# The guard is a denylist on purpose: it has to run over SVG that a human authored
# by hand, and an element allowlist strict enough to be safe would reject ordinary
# diagrams. Provenance is the primary control (diagrams are written beside the
# markdown by the document's own author); this is the backstop for a file that
# arrived some other way. Fail closed: a hit aborts the render.
_SVG_DENY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"<\s*script", "a script element"),
    (r"<\s*foreignObject", "a foreignObject element"),
    (r"<\s*(iframe|embed|object|audio|video|animate|set)\b", "an active embedded element"),
    (r"<!\s*(DOCTYPE|ENTITY)", "a DOCTYPE or ENTITY declaration"),
    (r"\son[a-z]+\s*=", "an inline event handler attribute"),
    (r"(?:javascript|data|vbscript)\s*:", "a script-capable URL scheme"),
    (r"(?:xlink:)?href\s*=\s*[\"'](?!#)", "an external reference (only #fragment is allowed)"),
)

# The template inverts `pre` blocks (dark on a light page), which suits an explainer
# where code is occasional. A design document is mostly code, so "match" points
# --pre-bg at the same surface as inline `code` and keeps one palette across both.
# Reusing the existing variables rather than hardcoding hex means dark mode follows.
_CODE_STYLE_MATCH: str = """<style>
  :root { --pre-bg: var(--code-bg); --pre-fg: var(--fg); }
  pre { border: 1px solid var(--rule); }
  pre code { color: var(--fg); }
</style>
"""


def _slug(
    text: str,
) -> str:
    """Build a GitHub-style heading anchor so in-document links keep working."""
    cleaned = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_]+", "-", cleaned).strip("-")


def _strip_inline_markdown(
    text: str,
) -> str:
    """Reduce inline markdown to plain text, for a heading used in the nav."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[*_]", "", text)


def _is_safe_url(
    url: str,
) -> bool:
    """Return True if a URL is safe to place in an href.

    Allows http, https, mailto, and any scheme-less (relative or fragment)
    target. Control characters and whitespace are stripped before the check so
    ``java\\tscript:`` cannot smuggle a scheme past it. Fails closed on anything
    unparseable.
    """
    cleaned = re.sub(r"[\x00-\x20\x7f]", "", url)
    if not cleaned:
        return False
    match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", cleaned)
    if match is None:
        return True  # relative path or #fragment, no scheme to abuse
    return match.group(1).lower() in _SAFE_URL_SCHEMES


def _byline_link(
    label: str,
    url: str,
) -> str:
    """Build one escaped anchor, or inert escaped text for an unsafe URL."""
    if not _is_safe_url(url):
        logger.warning("byline link to a disallowed URL scheme rendered as text")
        return html.escape(f"[{label}]({url})")
    return f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'


def _inline_markdown_to_html(
    text: str,
) -> str:
    """Convert the inline markdown a byline uses into escaped, safe HTML.

    The byline renders inside a paragraph the template owns, so it must be HTML
    rather than markdown. Everything is HTML-escaped and only three constructs
    are then re-introduced: a markdown link, a bare URL, and inline code.

    This has to escape rather than interpolate. A review byline carries metadata
    fetched from GitHub (PR title, author, branch), so a PR titled with an
    ``<img src=x onerror=...>`` would otherwise become live HTML in a page the
    maintainer opens, and the skill offers to serve that page from 127.0.0.1,
    which is an origin that can reach a locally running registry.
    """
    pattern = re.compile(
        r"\[(?P<label>[^\]]+)\]\((?P<url>[^)\s]+)\)"  # markdown link
        r"|(?P<bare>https?://[^\s<>\"')]+)"  # bare URL
        r"|`(?P<code>[^`]+)`"  # inline code
    )
    out: list[str] = []
    position = 0
    for match in pattern.finditer(text):
        out.append(html.escape(text[position : match.start()]))
        if match.group("label") is not None:
            out.append(_byline_link(match.group("label"), match.group("url")))
        elif match.group("bare") is not None:
            out.append(_byline_link(match.group("bare"), match.group("bare")))
        else:
            out.append(f"<code>{html.escape(match.group('code'))}</code>")
        position = match.end()
    out.append(html.escape(text[position:]))
    # Emphasis markers are dropped rather than converted: the byline is already
    # styled by the template, and leaving them unescaped is a needless sink.
    rendered = re.sub(r"(?<!\\)[*_]", "", "".join(out))
    # Each metadata line arrives newline-separated; join them only now, after
    # escaping, so the separator entity reaches the page intact.
    return rendered.replace("\n", " &middot; ")


def _is_italic_line(
    stripped: str,
) -> bool:
    """Return True for a line that is entirely one italic span."""
    if not (stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 2):
        return False
    return not stripped.startswith("**")


def _split_document(
    markdown_text: str,
) -> tuple[str, str, str]:
    """Split the source into (title, byline markdown, body markdown).

    The template renders the H1 and the byline itself, so both are stripped from
    the body. The byline is the run of italic metadata lines directly below the
    H1, joined: the explainer convention is one such line, but a design document
    carries several (created, author, status, source issue), and keeping only the
    first would lose the rest.

    Everything after that run stays in the body, including a lead paragraph, a
    horizontal rule, or a bold-style metadata block. Skipping ahead to the first
    H2 instead would silently delete real content: several existing reviews put a
    paste-ready PR comment or their metadata above the first H2.
    """
    lines = markdown_text.splitlines()
    title = ""
    byline_parts: list[str] = []
    cursor = 0

    while cursor < len(lines) and not title:
        stripped = lines[cursor].strip()
        cursor += 1
        if stripped.startswith("# "):
            title = stripped[2:].strip()

    if not title:
        # No H1 anywhere, so the search above walked off the end. Rewind rather
        # than treat the whole document as consumed: a file whose first line is
        # malformed must still render its content, with the title falling back to
        # the filename. Returning an empty body here would publish a blank page.
        return "", "", markdown_text

    while cursor < len(lines):
        stripped = lines[cursor].strip()
        if not stripped:
            cursor += 1
            continue
        if not _is_italic_line(stripped):
            break
        byline_parts.append(stripped.strip("*").strip())
        cursor += 1

    # Joined with a newline, not the HTML separator: the converter escapes its
    # input, so an entity inserted here would render as literal "&middot;" text.
    return title, "\n".join(byline_parts), "\n".join(lines[cursor:])


def _assert_svg_inert(
    path: pathlib.Path,
    svg: str,
) -> None:
    """Raise if an SVG carries anything executable or externally referencing.

    The file is inlined verbatim into the page, so a script element, an event
    handler, or an off-document reference would execute in the reader's browser.
    Fail closed: abort the render rather than drop the diagram, because silently
    falling back to the ASCII version would hide a tampered file.
    """
    for pattern, description in _SVG_DENY_PATTERNS:
        if re.search(pattern, svg, re.IGNORECASE):
            raise ValueError(f"{path} contains {description} and will not be inlined")


def _load_diagrams(
    diagrams_dir: pathlib.Path | None,
) -> dict[str, str]:
    """Load, size-check, and vet every <key>.svg in the diagrams directory."""
    if diagrams_dir is None or not diagrams_dir.is_dir():
        return {}
    diagrams: dict[str, str] = {}
    total = 0
    for path in sorted(diagrams_dir.glob("*.svg")):
        size = path.stat().st_size
        if size > MAX_SVG_BYTES:
            raise ValueError(f"{path} is {size} bytes, over the {MAX_SVG_BYTES} per-file limit")
        total += size
        if total > MAX_SVG_TOTAL_BYTES:
            raise ValueError(f"{diagrams_dir} exceeds the {MAX_SVG_TOTAL_BYTES} total byte limit")
        svg = path.read_text(encoding="utf-8").strip()
        _assert_svg_inert(path, svg)
        diagrams[path.stem] = svg
    logger.info("loaded %d vetted diagram(s) from %s", len(diagrams), diagrams_dir)
    return diagrams


def _install_svg_fence(
    parser: MarkdownIt,
    diagrams: dict[str, str],
    missing: list[str],
) -> None:
    """Replace an ``svg:<key>`` fence with the matching inlined SVG figure."""
    default_fence = parser.renderer.rules.get("fence")

    def _fence(
        tokens: list,
        idx: int,
        options: dict,
        env: dict,
    ) -> str:
        info = tokens[idx].info.strip()
        if info.startswith("svg:"):
            tag, _, caption = info.partition(" ")
            key = tag[len("svg:") :]
            svg = diagrams.get(key)
            if svg is None:
                missing.append(key)
            else:
                figure = f"<figure>\n{svg}\n"
                if caption.strip():
                    figure += f"  <figcaption>{html.escape(caption.strip())}</figcaption>\n"
                return figure + "</figure>\n"
        return default_fence(tokens, idx, options, env)

    parser.renderer.rules["fence"] = _fence


def _add_heading_ids(
    body_html: str,
) -> str:
    """Give every h2, h3 and h4 an id matching its GitHub anchor.

    Entities are unescaped before slugging. markdown-it renders an ampersand as
    ``&amp;``, so slugging the escaped text turns "Authentication & Security" into
    "authentication-amp-security" and every table-of-contents link to that heading
    breaks. h4 is included because a review's persona sections are often that deep.
    """

    def _replace(
        match: re.Match,
    ) -> str:
        level, inner = match.group(1), match.group(2)
        plain = html.unescape(re.sub(r"<[^>]+>", "", inner))
        return f'<h{level} id="{_slug(_strip_inline_markdown(plain))}">{inner}</h{level}>'

    return re.sub(r"<h([234])>(.*?)</h\1>", _replace, body_html, flags=re.S)


def _build_nav(
    markdown_text: str,
) -> str:
    """Build a section nav from the H2 headings, replacing the template's nav."""
    items: list[str] = []
    for line in markdown_text.splitlines():
        if line.startswith("## ") and "table of contents" not in line.lower():
            text = _strip_inline_markdown(html.unescape(line[3:].strip()))
            items.append(f'    <a href="#{_slug(text)}">{html.escape(text)}</a>')
    if not items:
        return ""
    return '  <nav class="levels">\n' + "\n".join(items) + "\n  </nav>"


def _replace_template_nav(
    page: str,
    nav: str,
) -> str:
    """Swap the template's three-level nav for the section nav, or drop it."""
    start = page.find('<nav class="levels">')
    if start == -1:
        return page
    line_start = page.rfind("\n", 0, start) + 1
    end = page.index("</nav>", start) + len("</nav>")
    return page[:line_start] + nav + page[end:]


def _read_source(
    path: pathlib.Path,
) -> str:
    """Read the markdown source, refusing anything implausibly large."""
    size = path.stat().st_size
    if size > MAX_INPUT_BYTES:
        raise ValueError(f"{path} is {size} bytes, over the {MAX_INPUT_BYTES} limit")
    return path.read_text(encoding="utf-8")


def _render_page(
    source: pathlib.Path,
    template_path: pathlib.Path,
    diagrams_dir: pathlib.Path | None,
    title_override: str | None,
    byline_override: str | None,
    footer_html: str | None,
    code_style: str,
    want_nav: bool,
) -> tuple[str, dict[str, int], list[str]]:
    """Render the markdown into a full HTML page plus a few sanity counts."""
    markdown_text = _read_source(source)
    template_size = template_path.stat().st_size
    if template_size > MAX_TEMPLATE_BYTES:
        raise ValueError(
            f"{template_path} is {template_size} bytes, over the {MAX_TEMPLATE_BYTES} limit"
        )
    template = template_path.read_text(encoding="utf-8")
    title, byline_md, body_md = _split_document(markdown_text)

    diagrams = _load_diagrams(diagrams_dir)
    missing: list[str] = []
    # html=False is a security control, not a style choice. The pr-review and
    # new-feature-design skills summarize GitHub-fetched PR and issue bodies into
    # these documents, so a <script> or an <img onerror> in a PR body would
    # otherwise become live HTML in a page the maintainer opens. With it off, raw
    # HTML renders as visible, inert text. Diagrams do not need it: the svg: fence
    # injects through this module's own vetted rule.
    parser = MarkdownIt("commonmark", {"html": False}).enable("table").enable("strikethrough")
    _install_svg_fence(parser, diagrams, missing)
    body = _add_heading_ids(parser.render(body_md))

    if code_style == "match":
        template = template.replace("</head>", _CODE_STYLE_MATCH + "</head>", 1)

    resolved_title = title_override or title or source.stem
    resolved_byline = byline_override or _inline_markdown_to_html(byline_md)
    page = template.replace("{{TITLE}}", html.escape(resolved_title))
    page = page.replace("{{BYLINE}}", resolved_byline)
    page = page.replace("{{BODY}}", body)
    page = page.replace("{{FOOTER}}", footer_html or "")
    page = _replace_template_nav(page, _build_nav(markdown_text) if want_nav else "")

    counts = {
        "tables": page.count("<table>"),
        "diagrams": page.count("<svg "),
        "headings": len(re.findall(r'<h[234] id="', page)),
    }
    return page, counts, missing


def _report(
    output: pathlib.Path,
    page: str,
    counts: dict[str, int],
    missing: list[str],
) -> None:
    """Log what was written and anything the author should look at."""
    logger.info(
        "wrote %s (%d bytes, %d tables, %d diagrams, %d anchored headings)",
        output,
        len(page),
        counts["tables"],
        counts["diagrams"],
        counts["headings"],
    )
    for key in missing:
        logger.warning("no diagram file for 'svg:%s'; the ASCII fallback was rendered instead", key)
    leftover = [
        token for token in ("{{TITLE}}", "{{BYLINE}}", "{{BODY}}", "{{FOOTER}}") if token in page
    ]
    if leftover:
        logger.warning("template placeholders left unfilled: %s", ", ".join(leftover))
    broken = _broken_anchors(page)
    if broken:
        logger.warning("in-page links with no matching heading: %s", ", ".join(broken))


def _broken_anchors(
    page: str,
) -> list[str]:
    """Return in-page link targets that no heading id satisfies."""
    ids = set(re.findall(r'<h[234] id="([^"]+)"', page))
    targets = set(re.findall(r'href="#([^"]+)"', page))
    return sorted(target for target in targets if target not in ids)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Render a markdown document into self-contained styled HTML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Simplest form: title and byline come from the H1 and the italic line under it
    uv run python scripts/render-doc-html.py .scratchpad/issue-123/lld.md

    # Explicit output path and title
    uv run python scripts/render-doc-html.py doc.md -o doc.html --title "Design"

    # Inline SVG diagrams from a sibling directory (the default when it exists)
    uv run python scripts/render-doc-html.py doc.md --diagrams .scratchpad/issue-123/diagrams

Diagrams: fence a block as ```svg:my-key Optional caption text and the renderer
inlines diagrams/my-key.svg in its place. Keep the ASCII inside the fence as the
terminal-readable fallback.
""",
    )
    parser.add_argument("source", help="Path to the markdown file to render")
    parser.add_argument(
        "-o",
        "--output",
        help="Output HTML path (default: the source path with a .html suffix)",
    )
    parser.add_argument("--title", help="Page title (default: the document's H1)")
    parser.add_argument(
        "--byline-html",
        dest="byline_html",
        help=(
            "TRUSTED HTML, inserted verbatim. First-party text only, never fetched "
            "metadata. Default: the italic lines below the H1, escaped and converted."
        ),
    )
    parser.add_argument(
        "--footer-html",
        dest="footer_html",
        help=(
            "TRUSTED HTML, inserted verbatim. First-party provenance text only, never "
            "fetched metadata. Default: empty."
        ),
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help=f"HTML template with placeholders (default: {DEFAULT_TEMPLATE})",
    )
    parser.add_argument(
        "--diagrams",
        help=(
            "Directory of <key>.svg files for svg: fences "
            f"(default: a '{DEFAULT_DIAGRAMS_DIRNAME}' directory beside the source)"
        ),
    )
    parser.add_argument(
        "--code-style",
        choices=("match", "invert"),
        default="match",
        help="Code blocks follow the page surface (match) or the template's dark block (invert)",
    )
    parser.add_argument(
        "--no-nav",
        action="store_true",
        help="Omit the section navigation built from the H2 headings",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    """Parse arguments, render the document, write it, and report."""
    args = _parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    source = pathlib.Path(args.source)
    output = pathlib.Path(args.output) if args.output else source.with_suffix(".html")
    template_path = pathlib.Path(args.template)
    if args.diagrams:
        diagrams_dir: pathlib.Path | None = pathlib.Path(args.diagrams)
    else:
        candidate = source.parent / DEFAULT_DIAGRAMS_DIRNAME
        diagrams_dir = candidate if candidate.is_dir() else None

    try:
        page, counts, missing = _render_page(
            source=source,
            template_path=template_path,
            diagrams_dir=diagrams_dir,
            title_override=args.title,
            byline_override=args.byline_html,
            footer_html=args.footer_html,
            code_style=args.code_style,
            want_nav=not args.no_nav,
        )
    except (OSError, ValueError) as exc:
        logger.error("could not render %s: %s", source, exc)
        return 1

    output.write_text(page, encoding="utf-8")
    _report(output, page, counts, missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
