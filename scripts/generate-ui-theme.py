#!/usr/bin/env python3
"""Generate a runtime branding theme.css (and optionally a manifest.json skeleton).

The registry UI reads its colors from CSS variables declared in
frontend/src/index.css. An operator reskins the UI by dropping a theme.css that
overrides those variables into the branding folder (see the branding contract).
Writing eleven RGB triplets per ramp by hand is tedious and easy to get wrong, so
this script derives each ramp from a single color.

How a ramp is built from one color:
    1. Convert the color to OKLCH and keep its hue.
    2. Anchor it at the step (50..950) whose reference lightness is closest.
    3. Remap the reference lightness curve (modeled on Tailwind's default
       palettes) so it passes through the input color at the anchor.
    4. Scale chroma along a curve that tapers toward the light and dark ends.
    5. Gamut-map each step back to sRGB by reducing chroma until it fits.

After generating, the script prints a WCAG 2.x contrast report for the pairs
the UI actually renders (accent button, accent text, soft pills, status pills).

Exit codes:
    0  files written (or CSS printed) and, in --strict mode, every check passed
    1  bad input, refused overwrite, or a contrast failure in --strict mode
"""

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

RGB = tuple[int, int, int]

STEPS: tuple[int, ...] = (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950)

# OKLCH lightness per step, averaged over Tailwind's default red, blue, green and
# violet palettes. The generated ramp is this curve, bent to pass through the input.
REFERENCE_LIGHTNESS: tuple[float, ...] = (
    0.973,
    0.943,
    0.897,
    0.825,
    0.726,
    0.647,
    0.573,
    0.503,
    0.437,
    0.387,
    0.272,
)

# Relative chroma per step (peak = 1.0), from the same palettes: near-neutral tints
# at 50, full color around 600, and a gentler fall-off toward 950.
CHROMA_TAPER: tuple[float, ...] = (
    0.06,
    0.12,
    0.23,
    0.41,
    0.68,
    0.89,
    1.0,
    0.96,
    0.81,
    0.64,
    0.44,
)

# Chroma below this is treated as achromatic (hue is meaningless).
ACHROMATIC_CHROMA: float = 1e-4
GAMUT_EPSILON: float = 1e-6
GAMUT_SEARCH_ITERATIONS: int = 32

HEX_PATTERN: re.Pattern[str] = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
THEME_ID_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
RESERVED_THEME_IDS: frozenset[str] = frozenset({"default", "light", "dark", "high-contrast"})
# A named theme becomes the class `theme-<id>` on <html> (see
# frontend/src/branding/manifest.ts THEME_CLASS_PREFIX), so ids never collide
# with Tailwind utilities.
THEME_CLASS_PREFIX: str = "theme-"
MAX_THEME_LABEL_LENGTH: int = 40

THEME_CSS_NAME: str = "theme.css"
MANIFEST_NAME: str = "manifest.json"
MANIFEST_VERSION: int = 1

# Optional ramps, in output order. Status ramps also get pill contrast checks.
STATUS_RAMPS: tuple[str, ...] = ("success", "warning", "danger", "info")
ENTITY_RAMPS: tuple[str, ...] = ("server", "agent", "skill", "virtual", "custom")
OPTIONAL_RAMPS: tuple[str, ...] = STATUS_RAMPS + ENTITY_RAMPS

WHITE: RGB = (255, 255, 255)
# Dark-mode surface (--gray-800 in index.css).
DARK_SURFACE: RGB = (29, 41, 57)
# index.css builds the dark soft fills as solid blends of step 900 over gray-800,
# so alpha modifiers on them stay valid. 0.45 reproduces its default (43 39 82).
DARK_SOFT_BLEND_ALPHA: float = 0.45

AA_TEXT_CONTRAST: float = 4.5

# Linear sRGB <-> LMS and LMS' <-> OKLab matrices (Bjorn Ottosson, 2020).
_LINEAR_SRGB_TO_LMS: tuple[tuple[float, float, float], ...] = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)
_LMS_TO_OKLAB: tuple[tuple[float, float, float], ...] = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)
_OKLAB_TO_LMS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.3963377774, 0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
)
_LMS_TO_LINEAR_SRGB: tuple[tuple[float, float, float], ...] = (
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
)


@dataclass(frozen=True)
class ContrastCheck:
    """One foreground/background pair from the contrast report."""

    label: str
    foreground: RGB
    background: RGB
    ratio: float
    threshold: float

    @property
    def passed(self) -> bool:
        """Whether the pair meets its threshold."""
        return self.ratio >= self.threshold


def parse_hex(
    value: str,
) -> RGB:
    """Parse ``#rgb`` or ``#rrggbb`` (leading ``#`` optional) into an RGB tuple.

    Args:
        value: The hex color string.

    Returns:
        The color as 0-255 integer channels.

    Raises:
        ValueError: If the string is not a 3- or 6-digit hex color.
    """
    match = HEX_PATTERN.match(value.strip())
    if not match:
        raise ValueError(f"'{value}' is not a hex color; use #rrggbb or #rgb, e.g. #0d6efd")
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))


def validate_theme_id(
    theme_id: str,
) -> str:
    """Check a theme id against the branding contract.

    Args:
        theme_id: Candidate id for ``html.theme-<id>`` scoping and ``manifest.themes[].id``.

    Returns:
        The id, unchanged.

    Raises:
        ValueError: If the id is malformed or reserved.
    """
    if not THEME_ID_PATTERN.match(theme_id):
        raise ValueError(
            f"theme id '{theme_id}' must match {THEME_ID_PATTERN.pattern}: start with a "
            "lowercase letter, then up to 31 lowercase letters, digits or hyphens"
        )
    if theme_id in RESERVED_THEME_IDS:
        raise ValueError(
            f"theme id '{theme_id}' is reserved; pick a name other than "
            f"{', '.join(sorted(RESERVED_THEME_IDS))}"
        )
    return theme_id


def _hex_arg(
    value: str,
) -> RGB:
    """argparse type wrapper around parse_hex."""
    try:
        return parse_hex(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _theme_id_arg(
    value: str,
) -> str:
    """argparse type wrapper around validate_theme_id."""
    try:
        return validate_theme_id(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _mat_vec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Multiply a 3x3 matrix by a 3-vector."""
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _srgb_to_linear(
    channel: float,
) -> float:
    """Decode one gamma-encoded sRGB channel in [0, 1] to linear light."""
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(
    channel: float,
) -> float:
    """Encode one linear-light channel in [0, 1] to gamma-encoded sRGB."""
    if channel <= 0.0031308:
        return channel * 12.92
    return 1.055 * channel ** (1 / 2.4) - 0.055


def rgb_to_oklch(
    rgb: RGB,
) -> tuple[float, float, float]:
    """Convert an sRGB color to OKLCH.

    Args:
        rgb: 0-255 integer channels.

    Returns:
        ``(L, C, H)`` with L in [0, 1], C >= 0 and H in degrees [0, 360).
    """
    linear = tuple(_srgb_to_linear(c / 255) for c in rgb)
    lms = _mat_vec(_LINEAR_SRGB_TO_LMS, linear)
    lms_cbrt = tuple(math.copysign(abs(v) ** (1 / 3), v) for v in lms)
    lightness, a, b = _mat_vec(_LMS_TO_OKLAB, lms_cbrt)
    chroma = math.hypot(a, b)
    hue = math.degrees(math.atan2(b, a)) % 360 if chroma > ACHROMATIC_CHROMA else 0.0
    return (lightness, chroma, hue)


def _oklch_to_linear_srgb(
    lightness: float,
    chroma: float,
    hue: float,
) -> tuple[float, float, float]:
    """Convert OKLCH to linear sRGB without clamping (may fall outside [0, 1])."""
    a = chroma * math.cos(math.radians(hue))
    b = chroma * math.sin(math.radians(hue))
    lms_cbrt = _mat_vec(_OKLAB_TO_LMS, (lightness, a, b))
    lms = tuple(v**3 for v in lms_cbrt)
    return _mat_vec(_LMS_TO_LINEAR_SRGB, lms)


def _in_gamut(
    linear: tuple[float, float, float],
) -> bool:
    """Whether a linear sRGB color fits the sRGB gamut."""
    return all(-GAMUT_EPSILON <= c <= 1 + GAMUT_EPSILON for c in linear)


def _linear_to_rgb(
    linear: tuple[float, float, float],
) -> RGB:
    """Clamp, gamma-encode and round a linear sRGB color to 0-255 channels."""
    encoded = (_linear_to_srgb(min(1.0, max(0.0, c))) for c in linear)
    channels = tuple(int(math.floor(c * 255 + 0.5)) for c in encoded)
    return (channels[0], channels[1], channels[2])


def oklch_to_rgb(
    lightness: float,
    chroma: float,
    hue: float,
) -> RGB:
    """Convert OKLCH to sRGB, reducing chroma until the color is in gamut.

    Lightness and hue are preserved; only chroma gives way, so a ramp keeps its
    lightness curve even where the requested saturation cannot be displayed.

    Args:
        lightness: OKLCH L in [0, 1].
        chroma: OKLCH C (>= 0).
        hue: OKLCH H in degrees.

    Returns:
        0-255 integer channels.
    """
    lightness = min(1.0, max(0.0, lightness))
    linear = _oklch_to_linear_srgb(lightness, chroma, hue)
    if _in_gamut(linear):
        return _linear_to_rgb(linear)
    low, high = 0.0, chroma
    for _ in range(GAMUT_SEARCH_ITERATIONS):
        mid = (low + high) / 2
        if _in_gamut(_oklch_to_linear_srgb(lightness, mid, hue)):
            low = mid
        else:
            high = mid
    return _linear_to_rgb(_oklch_to_linear_srgb(lightness, low, hue))


def anchor_index(
    lightness: float,
) -> int:
    """Index into STEPS of the step whose reference lightness is closest."""
    return min(
        range(len(REFERENCE_LIGHTNESS)),
        key=lambda i: abs(REFERENCE_LIGHTNESS[i] - lightness),
    )


def _ramp_lightness(
    lightness: float,
    anchor: int,
) -> list[float]:
    """Bend the reference lightness curve so it passes through ``lightness`` at ``anchor``.

    Each side of the anchor is remapped linearly from the reference range onto
    the range between the curve's end and the input, so the ends stay put and
    the curve stays strictly decreasing.
    """
    first, last = REFERENCE_LIGHTNESS[0], REFERENCE_LIGHTNESS[-1]
    anchor_ref = REFERENCE_LIGHTNESS[anchor]
    result: list[float] = []
    for index, ref in enumerate(REFERENCE_LIGHTNESS):
        if index == anchor:
            result.append(lightness)
        elif index < anchor:
            scale = (lightness - first) / (anchor_ref - first)
            result.append(first + (ref - first) * scale)
        else:
            scale = (lightness - last) / (anchor_ref - last)
            result.append(last + (ref - last) * scale)
    return result


def generate_ramp(
    rgb: RGB,
) -> dict[int, RGB]:
    """Build an 11-step ramp (50..950) around one color.

    Args:
        rgb: The seed color; it appears unchanged at its anchored step.

    Returns:
        Mapping of step number to 0-255 RGB channels.
    """
    lightness, chroma, hue = rgb_to_oklch(rgb)
    anchor = anchor_index(lightness)
    peak_chroma = chroma / CHROMA_TAPER[anchor]
    lightness_curve = _ramp_lightness(lightness, anchor)
    logger.debug(
        f"Seed {_format_hex(rgb)}: L={lightness:.4f} C={chroma:.4f} H={hue:.1f}, "
        f"anchored at {STEPS[anchor]}"
    )
    ramp: dict[int, RGB] = {}
    for index, step in enumerate(STEPS):
        if index == anchor:
            ramp[step] = rgb
            continue
        ramp[step] = oklch_to_rgb(lightness_curve[index], peak_chroma * CHROMA_TAPER[index], hue)
    return ramp


def relative_luminance(
    rgb: RGB,
) -> float:
    """WCAG 2.x relative luminance of an sRGB color."""
    red, green, blue = (_srgb_to_linear(c / 255) for c in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(
    foreground: RGB,
    background: RGB,
) -> float:
    """WCAG 2.x contrast ratio between two colors (1.0 to 21.0)."""
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _blend(
    top: RGB,
    bottom: RGB,
    alpha: float,
) -> RGB:
    """Composite ``top`` at ``alpha`` over an opaque ``bottom`` (gamma-space, like CSS)."""
    pairs = zip(top, bottom, strict=True)
    channels = tuple(int(math.floor(b + (t - b) * alpha + 0.5)) for t, b in pairs)
    return (channels[0], channels[1], channels[2])


def _format_hex(
    rgb: RGB,
) -> str:
    """Format RGB channels as ``#rrggbb``."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _triplet(
    rgb: RGB,
) -> str:
    """Format RGB channels as a CSS variable value (``13 110 253``)."""
    return f"{rgb[0]} {rgb[1]} {rgb[2]}"


def _check(
    label: str,
    foreground: RGB,
    background: RGB,
) -> ContrastCheck:
    """Build one AA text contrast check."""
    return ContrastCheck(
        label=label,
        foreground=foreground,
        background=background,
        ratio=contrast_ratio(foreground, background),
        threshold=AA_TEXT_CONTRAST,
    )


def build_contrast_checks(
    ramps: dict[str, dict[int, RGB]],
) -> list[ContrastCheck]:
    """List the contrast pairs the UI renders with the generated ramps.

    Args:
        ramps: Generated ramps keyed by name (``primary`` always present).

    Returns:
        Checks in report order.
    """
    primary = ramps["primary"]
    checks = [
        _check("white on primary-600 (accent button, light)", WHITE, primary[600]),
        _check("white on primary-600 (accent button, dark)", WHITE, primary[600]),
        _check("primary-600 on white (accent text, light)", primary[600], WHITE),
        _check("primary-300 on gray-800 (accent text, dark)", primary[300], DARK_SURFACE),
        _check("primary-700 on primary-50 (soft pill, light)", primary[700], primary[50]),
    ]
    for name in OPTIONAL_RAMPS:
        ramp = ramps.get(name)
        if ramp is None:
            continue
        checks.append(_check(f"{name}-700 on {name}-50 (pill, light)", ramp[700], ramp[50]))
        checks.append(_check(f"{name}-300 on gray-800 (pill, dark)", ramp[300], DARK_SURFACE))
    return checks


def format_report(
    checks: list[ContrastCheck],
) -> str:
    """Render the contrast checks as a fixed-width table."""
    label_width = max(len("Pair"), *(len(c.label) for c in checks))
    header = f"{'Pair':<{label_width}}  {'FG':<7}  {'BG':<7}  {'Ratio':>6}  {'Min':>4}  Result"
    lines = [header, "-" * len(header)]
    for check in checks:
        result = "pass" if check.passed else "FAIL"
        lines.append(
            f"{check.label:<{label_width}}  {_format_hex(check.foreground)}  "
            f"{_format_hex(check.background)}  {check.ratio:>6.2f}  {check.threshold:>4.1f}  "
            f"{result}"
        )
    failures = sum(1 for c in checks if not c.passed)
    lines.append("")
    lines.append(
        f"{len(checks) - failures} of {len(checks)} checks pass WCAG AA "
        f"({AA_TEXT_CONTRAST}:1 for text)." + (f" {failures} FAILED." if failures else "")
    )
    return "\n".join(lines)


def _selectors(
    theme_id: str | None,
) -> tuple[str, str]:
    """Return the (light/shared, dark) selectors for the output scope."""
    if theme_id is None:
        return (":root", "html.dark")
    theme_class = f"{THEME_CLASS_PREFIX}{theme_id}"
    return (f"html.{theme_class}", f"html.{theme_class}.dark")


def render_css(
    ramps: dict[str, dict[int, RGB]],
    seeds: dict[str, RGB],
    theme_id: str | None,
) -> str:
    """Render theme.css for the generated ramps.

    Light/shared block: every generated ramp, plus the danger semantic tokens
    when a danger ramp is given (index.css hard-codes those). Dark block: the
    soft fills index.css hard-codes as blends of the default palette.

    Args:
        ramps: Generated ramps keyed by name, ``primary`` first.
        seeds: Seed color per ramp name, for the comments.
        theme_id: Named theme to scope under, or None for the default theme.

    Returns:
        The stylesheet text.
    """
    light_selector, dark_selector = _selectors(theme_id)
    seed_summary = " ".join(f"{name}={_format_hex(rgb)}" for name, rgb in seeds.items())
    lines = [
        "/*",
        " * Generated by scripts/generate-ui-theme.py. Re-run it rather than editing by hand.",
        f" * Seeds: {seed_summary}",
        " * Values are RGB channel triplets, used by the UI as rgb(var(--name)).",
        " */",
        f"{light_selector} {{",
    ]
    for name, ramp in ramps.items():
        anchor_step = STEPS[anchor_index(rgb_to_oklch(seeds[name])[0])]
        lines.append(f"  /* {name}: seed {_format_hex(seeds[name])} at step {anchor_step} */")
        lines.extend(f"  --{name}-{step}: {_triplet(rgb)};" for step, rgb in ramp.items())
    if "danger" in ramps:
        lines.extend(
            [
                "  /* Danger semantics follow the danger ramp. */",
                "  --color-danger: var(--danger-600);",
                "  --color-danger-hover: var(--danger-700);",
                "  --color-danger-soft: var(--danger-50);",
                "  --color-danger-soft-fg: var(--danger-700);",
            ]
        )
    lines.append("}")
    lines.append("")
    lines.append(f"{dark_selector} {{")
    lines.append("  /* Solid blends of step 900 over gray-800, as in index.css. */")
    accent_soft = _blend(ramps["primary"][900], DARK_SURFACE, DARK_SOFT_BLEND_ALPHA)
    lines.append(f"  --color-accent-soft: {_triplet(accent_soft)};")
    if "danger" in ramps:
        danger_soft = _blend(ramps["danger"][900], DARK_SURFACE, DARK_SOFT_BLEND_ALPHA)
        lines.extend(
            [
                "  --color-danger: var(--danger-500);",
                "  --color-danger-hover: var(--danger-400);",
                f"  --color-danger-soft: {_triplet(danger_soft)};",
                "  --color-danger-soft-fg: var(--danger-300);",
            ]
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _theme_label(
    theme_id: str,
) -> str:
    """Derive a display label from a theme id (``acme-dark-blue`` -> ``Acme Dark Blue``)."""
    return theme_id.replace("-", " ").title()[:MAX_THEME_LABEL_LENGTH]


def build_manifest(
    theme_id: str | None,
) -> dict:
    """Build a manifest.json skeleton per the branding contract.

    Logo and favicon keys are left out: they must name files that exist in the
    branding folder, and this script does not create any.

    Args:
        theme_id: Named theme to list, or None when theme.css targets the default theme.

    Returns:
        The manifest as a JSON-serializable dict.
    """
    themes = [] if theme_id is None else [{"id": theme_id, "label": _theme_label(theme_id)}]
    return {
        "version": MANIFEST_VERSION,
        "defaultMode": "system",
        "lockMode": None,
        "themes": themes,
        "defaultTheme": "default",
    }


def write_outputs(
    output_dir: Path,
    files: dict[str, str],
    force: bool,
) -> list[Path]:
    """Write files into ``output_dir``, refusing to overwrite unless ``force``.

    Every target is checked before anything is written, so a refusal leaves the
    folder untouched.

    Args:
        output_dir: Branding folder; created if missing.
        files: File name to content.
        force: Overwrite existing files.

    Returns:
        Paths written.

    Raises:
        FileExistsError: If a target exists and ``force`` is False.
        NotADirectoryError: If ``output_dir`` exists and is not a directory.
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"--output-dir {output_dir} exists and is not a directory")
    targets = [output_dir / name for name in files]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"refusing to overwrite {', '.join(existing)}; pass --force to replace"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in zip(targets, files.values(), strict=True):
        path.write_text(content, encoding="utf-8")
        logger.info(f"Wrote {path}")
    return targets


def _collect_seeds(
    args: argparse.Namespace,
) -> dict[str, RGB]:
    """Seed colors by ramp name, primary first, in output order."""
    seeds: dict[str, RGB] = {"primary": args.brand}
    for name in OPTIONAL_RAMPS:
        value = getattr(args, name)
        if value is not None:
            seeds[name] = value
    return seeds


def _log_checks(
    checks: list[ContrastCheck],
) -> None:
    """Log each check (failures as warnings) and a summary."""
    for check in checks:
        message = (
            f"Contrast {check.label}: {_format_hex(check.foreground)} on "
            f"{_format_hex(check.background)} = {check.ratio:.2f} (min {check.threshold})"
        )
        if check.passed:
            logger.debug(message)
        else:
            logger.warning(f"{message} FAILS")
    failures = sum(1 for c in checks if not c.passed)
    logger.info(f"Contrast report: {len(checks) - failures} pass, {failures} fail")


def _parse_args(
    argv: list[str] | None,
) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    # Print a theme.css for a blue brand to stdout (report goes to stderr)
    uv run python scripts/generate-ui-theme.py --brand '#0d6efd'

    # Write theme.css and manifest.json into the branding folder
    uv run python scripts/generate-ui-theme.py --brand '#0d6efd' \\
        --output-dir ./branding --manifest

    # A named theme "acme" with its own status colors, failing on any contrast miss
    uv run python scripts/generate-ui-theme.py --brand '#e4572e' --theme-id acme \\
        --success '#2e7d32' --danger '#c62828' --output-dir ./branding --manifest --strict

    # Regenerate over an existing theme.css
    uv run python scripts/generate-ui-theme.py --brand '#0d6efd' --output-dir ./branding --force
""",
    )
    parser.add_argument(
        "--brand",
        required=True,
        type=_hex_arg,
        help="Brand color (#rrggbb or #rgb). Generates the --primary-* ramp.",
    )
    for name in OPTIONAL_RAMPS:
        extra = " Defaults to the brand ramp in index.css." if name == "server" else ""
        parser.add_argument(
            f"--{name}",
            type=_hex_arg,
            default=None,
            help=f"Seed color for the --{name}-* ramp. Omitted ramps keep the app default.{extra}",
        )
    parser.add_argument(
        "--theme-id",
        type=_theme_id_arg,
        default=None,
        help=(
            "Scope output under html.theme-<id> and html.theme-<id>.dark (a named theme) instead of "
            ":root and html.dark. Must match ^[a-z][a-z0-9-]{0,31}$ and not be "
            "default, light, dark or high-contrast."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Branding folder to write theme.css into. Without it, CSS goes to stdout.",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Also write a manifest.json skeleton (requires --output-dir).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing theme.css / manifest.json.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 without writing anything if any contrast check fails.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)
    if args.manifest and args.output_dir is None:
        parser.error("--manifest requires --output-dir")
    return args


def main(
    argv: list[str] | None = None,
) -> int:
    """Generate the theme, report contrast, and write or print the output."""
    args = _parse_args(argv)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    seeds = _collect_seeds(args)
    ramps = {name: generate_ramp(rgb) for name, rgb in seeds.items()}
    checks = build_contrast_checks(ramps)
    css = render_css(ramps, seeds, args.theme_id)

    report_stream = sys.stdout if args.output_dir is not None else sys.stderr
    print(format_report(checks), file=report_stream)
    _log_checks(checks)

    if args.strict and not all(c.passed for c in checks):
        logger.error("Contrast failures in --strict mode; nothing written")
        return 1

    if args.output_dir is None:
        sys.stdout.write(css)
        return 0

    files = {THEME_CSS_NAME: css}
    if args.manifest:
        files[MANIFEST_NAME] = json.dumps(build_manifest(args.theme_id), indent=2) + "\n"
    try:
        write_outputs(args.output_dir, files, args.force)
    except (FileExistsError, NotADirectoryError) as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
