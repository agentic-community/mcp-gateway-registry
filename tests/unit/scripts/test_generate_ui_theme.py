"""Unit tests for scripts/generate-ui-theme.py.

The generator turns one seed color into an 11-step ramp and writes a branding
theme.css (plus an optional manifest.json skeleton). The module is loaded by
path because the filename uses hyphens (not importable as a package).
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "generate-ui-theme.py"
_INDEX_CSS = _REPO_ROOT / "frontend" / "src" / "index.css"

_ALL_RAMP_FLAGS: list[str] = [
    "--success",
    "#059669",
    "--warning",
    "#dc6803",
    "--danger",
    "#d92d20",
    "--info",
    "#2563eb",
    "--server",
    "#7c3aed",
    "--agent",
    "#0891b2",
    "--skill",
    "#d97706",
    "--virtual",
    "#0d9488",
    "--custom",
    "#3b82f6",
]

# Contract rules for manifest.json (branding contract, "manifest.json" section).
_FILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_IMAGE_EXTENSIONS = (".svg", ".png", ".webp", ".jpg", ".jpeg", ".ico")
_THEME_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_RESERVED_THEME_IDS = {"default", "light", "dark", "high-contrast"}
_MANIFEST_KEYS = {
    "version",
    "logo",
    "favicon",
    "themeColor",
    "defaultMode",
    "lockMode",
    "themes",
    "defaultTheme",
}


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_ui_theme", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_module()


def _declared_variables() -> set[str]:
    """Every custom property declared anywhere in index.css."""
    return set(re.findall(r"(--[a-z0-9-]+)\s*:", _INDEX_CSS.read_text(encoding="utf-8")))


def _emitted_variables(css: str) -> set[str]:
    return set(re.findall(r"^\s*(--[a-z0-9-]+)\s*:", css, flags=re.MULTILINE))


def _assert_manifest_valid(manifest: dict) -> None:
    """Check a manifest against the branding contract rules."""
    assert set(manifest) <= _MANIFEST_KEYS
    if "version" in manifest:
        assert manifest["version"] == 1
    for ref in [*manifest.get("logo", {}).values(), manifest.get("favicon")]:
        if ref is None:
            continue
        assert _FILE_NAME_PATTERN.match(ref) and ref.lower().endswith(_IMAGE_EXTENSIONS)
    assert manifest.get("defaultMode", "system") in {"light", "dark", "system"}
    assert manifest.get("lockMode") in {None, "light", "dark"}
    themes = manifest.get("themes", [])
    assert len(themes) <= 8
    ids = []
    for theme in themes:
        assert set(theme) <= {"id", "label"}
        assert _THEME_ID_PATTERN.match(theme["id"])
        assert theme["id"] not in _RESERVED_THEME_IDS
        assert isinstance(theme.get("label", ""), str) and len(theme.get("label", "")) <= 40
        ids.append(theme["id"])
    assert len(ids) == len(set(ids))
    assert manifest.get("defaultTheme", "default") in {"default", *ids}


@pytest.mark.unit
class TestColorMath:
    @pytest.mark.parametrize(
        ("hex_color", "expected"),
        [
            ("#ffffff", (1.0, 0.0)),
            ("#000000", (0.0, 0.0)),
            # Reference values from the CSS Color 4 OKLCH conversion.
            ("#ff0000", (0.62796, 0.25768, 29.2339)),
            ("#00ff00", (0.86644, 0.29483, 142.4953)),
            ("#0000ff", (0.45201, 0.31321, 264.0520)),
        ],
    )
    def test_oklch_known_values(self, gen, hex_color, expected):
        lightness, chroma, hue = gen.rgb_to_oklch(gen.parse_hex(hex_color))
        assert lightness == pytest.approx(expected[0], abs=1e-4)
        assert chroma == pytest.approx(expected[1], abs=1e-4)
        if len(expected) == 3:
            assert hue == pytest.approx(expected[2], abs=0.05)

    @pytest.mark.parametrize(
        "hex_color",
        ["#0d6efd", "#6b46c1", "#d92d20", "#059669", "#fafafa", "#101828", "#ffcc00", "#808080"],
    )
    def test_oklch_round_trip(self, gen, hex_color):
        rgb = gen.parse_hex(hex_color)
        assert gen.oklch_to_rgb(*gen.rgb_to_oklch(rgb)) == rgb

    def test_out_of_gamut_keeps_lightness_and_hue(self, gen):
        # Chroma 0.5 at this lightness is far outside sRGB; only chroma may give way.
        rgb = gen.oklch_to_rgb(0.7, 0.5, 145.0)
        assert all(0 <= c <= 255 for c in rgb)
        lightness, chroma, hue = gen.rgb_to_oklch(rgb)
        assert lightness == pytest.approx(0.7, abs=0.005)
        assert hue == pytest.approx(145.0, abs=1.0)
        assert chroma < 0.5

    @pytest.mark.parametrize(
        ("foreground", "background", "expected"),
        [
            ("#ffffff", "#000000", 21.0),
            ("#ffffff", "#767676", 4.54),
            ("#000000", "#000000", 1.0),
        ],
    )
    def test_contrast_ratio_known_pairs(self, gen, foreground, background, expected):
        ratio = gen.contrast_ratio(gen.parse_hex(foreground), gen.parse_hex(background))
        assert ratio == pytest.approx(expected, abs=0.01)
        # Order of the pair does not matter.
        assert gen.contrast_ratio(gen.parse_hex(background), gen.parse_hex(foreground)) == ratio

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("0d6efd", (13, 110, 253)), ("#0D6EFD", (13, 110, 253)), ("#06f", (0, 102, 255))],
    )
    def test_parse_hex_accepts_common_forms(self, gen, value, expected):
        assert gen.parse_hex(value) == expected

    @pytest.mark.parametrize("value", ["", "#12345", "#gggggg", "rgb(1,2,3)", "#0d6efd00"])
    def test_parse_hex_rejects_malformed(self, gen, value):
        with pytest.raises(ValueError):
            gen.parse_hex(value)


@pytest.mark.unit
class TestRamp:
    @pytest.mark.parametrize(
        "hex_color",
        # Mid tones, a light tint, a near-black, an achromatic gray, pure white and black.
        ["#0d6efd", "#6b46c1", "#ffcc00", "#e0f2fe", "#0c111d", "#808080", "#ffffff", "#000000"],
    )
    def test_ramp_is_monotonic_and_contains_seed(self, gen, hex_color):
        rgb = gen.parse_hex(hex_color)
        ramp = gen.generate_ramp(rgb)
        assert list(ramp) == list(gen.STEPS)

        lightness = [gen.rgb_to_oklch(ramp[step])[0] for step in gen.STEPS]
        assert all(a > b for a, b in zip(lightness, lightness[1:], strict=False)), lightness

        anchor_step = gen.STEPS[gen.anchor_index(gen.rgb_to_oklch(rgb)[0])]
        assert ramp[anchor_step] == rgb

    def test_seed_anchors_at_nearest_reference_step(self, gen):
        # Bootstrap blue sits at L ~0.58, closest to the 600 reference lightness.
        assert gen.STEPS[gen.anchor_index(gen.rgb_to_oklch(gen.parse_hex("#0d6efd"))[0])] == 600

    def test_ramp_keeps_hue(self, gen):
        _, _, seed_hue = gen.rgb_to_oklch(gen.parse_hex("#0d6efd"))
        ramp = gen.generate_ramp(gen.parse_hex("#0d6efd"))
        # The palest and darkest steps have little chroma, so 8-bit rounding moves
        # their hue most; the mid steps carry the hue.
        for step in (200, 300, 400, 500, 700, 800, 900):
            _, _, hue = gen.rgb_to_oklch(ramp[step])
            assert hue == pytest.approx(seed_hue, abs=3.0), step

    def test_chroma_tapers_at_ends(self, gen):
        ramp = gen.generate_ramp(gen.parse_hex("#0d6efd"))
        chroma = {step: gen.rgb_to_oklch(rgb)[1] for step, rgb in ramp.items()}
        assert chroma[50] < chroma[300] < chroma[600]
        assert chroma[950] < chroma[700]


@pytest.mark.unit
class TestCss:
    def test_emitted_variables_exist_in_index_css(self, gen, capsys):
        assert gen.main(["--brand", "#0d6efd", *_ALL_RAMP_FLAGS]) == 0
        emitted = _emitted_variables(capsys.readouterr().out)
        assert emitted
        missing = emitted - _declared_variables()
        assert not missing, f"generator emits variables index.css does not declare: {missing}"

    def test_brand_only_emits_primary_and_not_server(self, gen, capsys):
        assert gen.main(["--brand", "#0d6efd"]) == 0
        emitted = _emitted_variables(capsys.readouterr().out)
        assert {f"--primary-{step}" for step in gen.STEPS} <= emitted
        assert not any(
            name.startswith(("--server-", "--success-", "--danger-")) for name in emitted
        )

    def test_default_scope_uses_root_and_html_dark(self, gen):
        css = gen.render_css(
            {"primary": gen.generate_ramp((13, 110, 253))}, {"primary": (13, 110, 253)}, None
        )
        assert re.search(r"^:root \{", css, flags=re.MULTILINE)
        assert re.search(r"^html\.dark \{", css, flags=re.MULTILINE)
        assert "--primary-600: 13 110 253;" in css

    def test_theme_id_scopes_output(self, gen, capsys):
        assert gen.main(["--brand", "#0d6efd", "--theme-id", "acme"]) == 0
        css = capsys.readouterr().out
        assert re.search(r"^html\.theme-acme \{", css, flags=re.MULTILINE)
        assert re.search(r"^html\.theme-acme\.dark \{", css, flags=re.MULTILINE)
        assert ":root" not in css
        assert not re.search(r"^html\.dark \{", css, flags=re.MULTILINE)

    def test_default_dark_soft_blend_matches_index_css(self, gen):
        # index.css: --color-accent-soft: 43 39 82 = primary-900 (59 36 112) over gray-800.
        assert gen._blend((59, 36, 112), gen.DARK_SURFACE, gen.DARK_SOFT_BLEND_ALPHA) == (
            43,
            39,
            82,
        )


@pytest.mark.unit
class TestThemeId:
    @pytest.mark.parametrize("theme_id", ["acme", "a", "acme-2", "a" * 32])
    def test_accepts_valid_ids(self, gen, theme_id):
        assert gen.validate_theme_id(theme_id) == theme_id

    @pytest.mark.parametrize("theme_id", ["default", "light", "dark", "high-contrast"])
    def test_rejects_reserved_ids(self, gen, theme_id):
        with pytest.raises(ValueError, match="reserved"):
            gen.validate_theme_id(theme_id)

    @pytest.mark.parametrize(
        "theme_id", ["", "Acme", "1acme", "-acme", "acme_corp", "acme.dark", "a" * 33, "acme corp"]
    )
    def test_rejects_malformed_ids(self, gen, theme_id):
        with pytest.raises(ValueError, match="must match"):
            gen.validate_theme_id(theme_id)

    def test_cli_rejects_reserved_id(self, gen):
        with pytest.raises(SystemExit) as exc:
            gen.main(["--brand", "#0d6efd", "--theme-id", "dark"])
        assert exc.value.code == 2


@pytest.mark.unit
class TestContrastReport:
    def test_reports_requested_pairs(self, gen):
        ramps = {
            "primary": gen.generate_ramp((13, 110, 253)),
            "success": gen.generate_ramp((5, 150, 105)),
        }
        labels = [check.label for check in gen.build_contrast_checks(ramps)]
        assert len(labels) == 7
        assert any(label.startswith("success-700 on success-50") for label in labels)
        assert any(label.startswith("success-300 on gray-800") for label in labels)

    def test_marks_failures(self, gen):
        # A pale brand cannot carry white button text.
        checks = gen.build_contrast_checks({"primary": gen.generate_ramp((255, 230, 0))})
        report = gen.format_report(checks)
        assert any(not check.passed for check in checks)
        assert "FAIL" in report

    def test_strict_exits_nonzero_and_writes_nothing(self, gen, tmp_path):
        out = tmp_path / "branding"
        assert gen.main(["--brand", "#ffe600", "--output-dir", str(out), "--strict"]) == 1
        assert not (out / "theme.css").exists()

    def test_strict_passes_for_accessible_brand(self, gen, tmp_path):
        assert gen.main(["--brand", "#0d6efd", "--output-dir", str(tmp_path), "--strict"]) == 0
        assert (tmp_path / "theme.css").exists()


@pytest.mark.unit
class TestOutput:
    def test_refuses_overwrite_without_force(self, gen, tmp_path):
        existing = tmp_path / "theme.css"
        existing.write_text("/* operator edits */\n")
        assert gen.main(["--brand", "#0d6efd", "--output-dir", str(tmp_path), "--manifest"]) == 1
        assert existing.read_text() == "/* operator edits */\n"
        # The refusal happens before any write, so the manifest is not created either.
        assert not (tmp_path / "manifest.json").exists()

    def test_force_overwrites(self, gen, tmp_path):
        (tmp_path / "theme.css").write_text("/* old */\n")
        assert gen.main(["--brand", "#0d6efd", "--output-dir", str(tmp_path), "--force"]) == 0
        assert "--primary-600: 13 110 253;" in (tmp_path / "theme.css").read_text()

    def test_manifest_requires_output_dir(self, gen):
        with pytest.raises(SystemExit) as exc:
            gen.main(["--brand", "#0d6efd", "--manifest"])
        assert exc.value.code == 2

    def test_manifest_without_theme_is_valid(self, gen, tmp_path):
        assert gen.main(["--brand", "#0d6efd", "--output-dir", str(tmp_path), "--manifest"]) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        _assert_manifest_valid(manifest)
        assert manifest["themes"] == []

    def test_manifest_lists_named_theme(self, gen, tmp_path):
        argv = ["--brand", "#0d6efd", "--theme-id", "acme-blue", "--output-dir", str(tmp_path)]
        assert gen.main([*argv, "--manifest"]) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        _assert_manifest_valid(manifest)
        assert manifest["themes"] == [{"id": "acme-blue", "label": "Acme Blue"}]
