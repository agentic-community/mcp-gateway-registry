"""Generate the standalone one-page landing site from README.md.

The landing page is a single self-contained HTML file built from the repository README,
so the README stays the single source of truth (and keeps fork-safe relative links). Only
the *generated* HTML gets absolute links baked in, since the published site is org-specific.

Usage:
    uv run python scripts/build_landing_page.py [--out site/index.html]
"""

import argparse
import re
from pathlib import Path


REPO = "agentic-community/mcp-gateway-registry"
GH_BLOB = f"https://github.com/{REPO}/blob/main/"
GH_RAW = f"https://raw.githubusercontent.com/{REPO}/main/"


def _rewrite_links(
    markdown_text: str,
) -> str:
    """Rewrite README-relative links/images to absolute URLs for the standalone page.

    The README uses repo-relative links (fork-safe on GitHub). In the generated HTML those
    must resolve absolutely: docs pages and repo files point at GitHub blob URLs; images
    point at raw.githubusercontent.com.
    """

    def fix_link(match: re.Match) -> str:
        text, url = match.group(1), match.group(2)
        if url.startswith(("http", "#", "mailto:")):
            return match.group(0)
        return f"[{text}]({GH_BLOB}{url.lstrip('./')})"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", fix_link, markdown_text)
    text = text.replace('src="docs/img/', f'src="{GH_RAW}docs/img/')
    return text


def _split_hero(
    markdown_text: str,
) -> tuple[str, str, str]:
    """Split the README into (logo_src, tagline, body).

    The README opens with a centered hero block (logo, bold tagline, shields.io badges, and a
    quick-links bar) wrapped in a raw ``<div align="center">`` that Markdown renders as literal
    text, and which duplicates the page header/sidebar. We drop that whole block (everything up
    to the first ``---`` rule), extract the logo and tagline to render a clean hero, and return
    the remaining body for normal Markdown rendering.
    """
    text = re.sub(r"^<!--.*?-->\n", "", markdown_text, flags=re.DOTALL, count=1)

    parts = text.split("\n---\n", 1)
    hero, body = (parts[0], parts[1]) if len(parts) == 2 else ("", text)

    logo_match = re.search(r'<img[^>]*src="([^"]+)"', hero)
    logo_src = logo_match.group(1) if logo_match else ""
    if logo_src.startswith("docs/img/"):
        logo_src = GH_RAW + logo_src

    tag_match = re.search(r"\*\*(.+?)\*\*", hero)
    tagline = tag_match.group(1).strip() if tag_match else ""

    return logo_src, tagline, body.lstrip()


def _build_sidebar(
    html_body: str,
) -> str:
    """Build sidebar anchor links from the rendered h2 headings."""
    heads = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', html_body)
    return "\n".join(
        f'    <a href="#{hid}">{re.sub("<.*?>", "", txt)}</a>' for hid, txt in heads
    )


def _render(
    readme_path: Path,
    template_path: Path,
) -> str:
    """Render the full landing page HTML from README + template."""
    import markdown

    logo_src, tagline, body = _split_hero(readme_path.read_text())
    body = _rewrite_links(body)
    html_body = markdown.markdown(
        body,
        extensions=["extra", "toc", "sane_lists", "nl2br"],
    )

    hero = ""
    if logo_src:
        hero += f'<img class="hero-logo" src="{logo_src}" alt="MCP Gateway &amp; Registry">\n'
    if tagline:
        hero += f'<p class="hero-tagline">{tagline}</p>\n'

    template = template_path.read_text()
    return (
        template.replace("{{SIDEBAR}}", _build_sidebar(html_body))
        .replace("{{HERO}}", hero)
        .replace("{{CONTENT}}", html_body)
    )


def main() -> None:
    """Parse args and write the generated landing page."""
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build the landing page from README.md")
    parser.add_argument(
        "--out",
        default=str(repo_root / "site" / "index.html"),
        help="Output HTML path (default: site/index.html)",
    )
    args = parser.parse_args()

    html = _render(
        readme_path=repo_root / "README.md",
        template_path=repo_root / "landing" / "template.html",
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Wrote landing page: {out_path} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
