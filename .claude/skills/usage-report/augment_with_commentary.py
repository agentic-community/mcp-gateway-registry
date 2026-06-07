"""Augment a deterministic usage report with LLM-generated commentary blocks.

This script does two things:

  1. `extract`: find every `<!-- COMMENTARY:name -->` marker in the rendered
     markdown and emit a JSON manifest of section names + the surrounding
     text + the data files relevant to each section. The skill's Step 8
     hands this manifest to the LLM, which generates one prose paragraph
     per section.

  2. `apply`: take a JSON map of `{section_name: commentary_text}` and
     replace each `<!-- COMMENTARY:name -->` marker with the corresponding
     commentary paragraph. Markers without an entry in the JSON map are
     replaced with an empty string (silently dropped).

The split keeps the LLM's job narrow: produce a fixed-shape JSON of short
prose paragraphs. It cannot modify any number, table, chart reference, or
other section. The augmenter is the only path that mutates the markdown.

Usage:
    # Step 1: extract sections needing commentary (after render_report.py)
    augment_with_commentary.py extract \\
        --md $DATE_DIR/ai-registry-usage-report-YYYY-MM-DD.md \\
        --output $DATE_DIR/commentary-manifest.json

    # Step 2 (LLM produces commentary.json): see SKILL.md Step 8.

    # Step 3: apply the LLM's commentary back into the markdown
    augment_with_commentary.py apply \\
        --md $DATE_DIR/ai-registry-usage-report-YYYY-MM-DD.md \\
        --commentary $DATE_DIR/commentary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

MARKER_RE = re.compile(r"<!--\s*COMMENTARY:([\w_-]+)\s*-->")


def _split_sections(
    md_text: str,
) -> list[dict]:
    """Split markdown by `## ` headings and return [{name, level, text, start, end}, ...]."""
    section_starts: list[int] = [m.start() for m in re.finditer(r"^##[^#]", md_text, re.MULTILINE)]
    if not section_starts:
        return []
    sections = []
    for i, start in enumerate(section_starts):
        end = section_starts[i + 1] if i + 1 < len(section_starts) else len(md_text)
        chunk = md_text[start:end]
        first_line_end = chunk.find("\n")
        heading = chunk[:first_line_end].lstrip("# ").strip()
        sections.append({"name": heading, "start": start, "end": end, "text": chunk})
    return sections


def _enclosing_section(
    md_text: str,
    marker_pos: int,
) -> str | None:
    """Find the markdown section containing the marker at marker_pos."""
    sections = _split_sections(md_text)
    for sec in sections:
        if sec["start"] <= marker_pos < sec["end"]:
            return sec["text"]
    return None


def _marker_positions(
    md_text: str,
) -> list[dict]:
    """Find every COMMENTARY marker and the section it sits in."""
    out: list[dict] = []
    for m in MARKER_RE.finditer(md_text):
        name = m.group(1)
        sec_text = _enclosing_section(md_text, m.start())
        out.append(
            {
                "section_id": name,
                "marker_position": m.start(),
                "section_text": sec_text or "",
            }
        )
    return out


def _extract(
    args: argparse.Namespace,
) -> None:
    """Build a JSON manifest of every COMMENTARY anchor and its section context."""
    md_path = Path(args.md)
    if not md_path.exists():
        logger.error(f"Markdown file not found: {md_path}")
        raise SystemExit(1)

    md_text = md_path.read_text()
    markers = _marker_positions(md_text)

    if not markers:
        logger.warning(f"No COMMENTARY markers found in {md_path}")

    manifest = {
        "report_path": str(md_path),
        "report_date": args.date,
        "sections_needing_commentary": [
            {
                "section_id": m["section_id"],
                "section_text": m["section_text"],
            }
            for m in markers
        ],
        "instructions_for_llm": (
            "For each entry in `sections_needing_commentary`, read the "
            "`section_text` and produce a 2-4 sentence paragraph of analyst "
            "commentary. Output a JSON object mapping `section_id` to the "
            "commentary paragraph. Constraints:\n"
            " 1. Use only numbers, names, and patterns that appear in the "
            "`section_text` itself or in the broader report data. Do NOT "
            "invent statistics, customer IDs, or version numbers.\n"
            " 2. Synthesize meaning: explain what the numbers imply, compare "
            "to prior reports, identify shifts, name patterns. Do not just "
            "restate the table.\n"
            " 3. Keep prose tight: 2-4 sentences per section. No filler.\n"
            " 4. No em-dashes. No emojis. No headings or markdown subheadings "
            "in the commentary text itself - the augmenter will wrap it.\n"
            " 5. If a section has nothing meaningful to add (e.g. unchanged "
            "metrics), output an empty string for that section_id."
        ),
        "output_path": args.output,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        f"Wrote manifest with {len(markers)} sections needing commentary to {output_path}"
    )


def _apply(
    args: argparse.Namespace,
) -> None:
    """Replace COMMENTARY markers with commentary text from a JSON map."""
    md_path = Path(args.md)
    commentary_path = Path(args.commentary)

    if not md_path.exists():
        logger.error(f"Markdown file not found: {md_path}")
        raise SystemExit(1)
    if not commentary_path.exists():
        logger.error(f"Commentary JSON not found: {commentary_path}")
        raise SystemExit(1)

    md_text = md_path.read_text()
    raw = commentary_path.read_text().strip()
    # Allow either {section_id: text, ...} or {"commentary": {...}}
    parsed = json.loads(raw)
    if isinstance(parsed, dict) and "commentary" in parsed and isinstance(parsed["commentary"], dict):
        commentary = parsed["commentary"]
    elif isinstance(parsed, dict):
        commentary = parsed
    else:
        logger.error("Commentary JSON must be an object mapping section_id -> text")
        raise SystemExit(1)

    replaced = 0
    missing: list[str] = []

    def _sub(match: re.Match) -> str:
        nonlocal replaced
        section_id = match.group(1)
        text = (commentary.get(section_id) or "").strip()
        if not text:
            missing.append(section_id)
            return ""  # drop the marker silently
        replaced += 1
        # Format as italicized analyst commentary block
        return f"_Commentary: {text}_"

    new_md = MARKER_RE.sub(_sub, md_text)

    output_path = args.output if args.output else str(md_path)
    Path(output_path).write_text(new_md)
    logger.info(
        f"Applied {replaced} commentary blocks; {len(missing)} markers had empty/missing entries: {missing}"
    )
    logger.info(f"Wrote augmented markdown to {output_path}")


def main() -> None:
    """CLI dispatcher."""
    parser = argparse.ArgumentParser(
        description="Extract or apply LLM commentary blocks for the usage report"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract_p = sub.add_parser("extract", help="Build a manifest of sections needing commentary")
    extract_p.add_argument("--md", required=True, help="Path to the rendered report markdown")
    extract_p.add_argument("--date", required=True, help="Report date YYYY-MM-DD (for the manifest)")
    extract_p.add_argument("--output", required=True, help="Path to write the JSON manifest")

    apply_p = sub.add_parser("apply", help="Insert commentary into the markdown using a JSON map")
    apply_p.add_argument("--md", required=True, help="Path to the rendered report markdown")
    apply_p.add_argument("--commentary", required=True, help="Path to commentary.json from the LLM")
    apply_p.add_argument("--output", default=None, help="Output path (defaults to in-place edit of --md)")

    args = parser.parse_args()
    if args.cmd == "extract":
        _extract(args)
    elif args.cmd == "apply":
        _apply(args)
    else:
        parser.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
