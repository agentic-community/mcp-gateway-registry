# UI branding

You can reskin the registry UI for a deployment without rebuilding the image. You supply a folder of files, the registry serves them at `{ROOT_PATH}/branding/`, and the UI picks them up at runtime. The folder can set brand colors, light and dark logos, a favicon, a default or locked light/dark mode, extra named themes in the Appearance menu, and a custom font.

The UI title is a separate setting, `UI_TITLE`. The branding folder does not change it.

## What goes in the folder

The folder is flat: no subdirectories. Every file is optional; a deployment with no branding gets the default look.

| File | Purpose |
|---|---|
| `manifest.json` | Logos, favicon, browser theme color, mode behavior and the list of named themes. |
| `theme.css` | CSS variable overrides: colors, corner radius, fonts. |
| Images (`.svg`, `.png`, `.webp`, `.jpg`, `.ico`) | Logos and favicon referenced from `manifest.json`. |
| Fonts (`.woff2`) | Referenced from `@font-face` rules in `theme.css`. |

File names must start with a letter or digit and contain only letters, digits, `.`, `_` and `-`, up to 64 characters. The registry refuses other names, other extensions and files larger than 2 MiB.

## manifest.json

```json
{
  "version": 1,
  "logo": { "light": "logo-light.svg", "dark": "logo-dark.svg", "alt": "Acme" },
  "favicon": "favicon.svg",
  "themeColor": { "light": "#ffffff", "dark": "#1d2939" },
  "defaultMode": "system",
  "lockMode": null,
  "themes": [{ "id": "acme-night", "label": "Acme Night" }],
  "defaultTheme": "default"
}
```

| Key | Meaning |
|---|---|
| `logo.light`, `logo.dark` | Header logo for each mode, as a file name in the folder. If you give only one, the UI uses it in both modes. A logo can be a wordmark: the header keeps its aspect ratio, 36px tall and at most 160px wide. |
| `logo.alt` | Alt text for the logo. Leave it out when the logo only repeats the UI title next to it. |
| `favicon` | Browser tab icon, as a file name in the folder. |
| `themeColor` | Hex color of the browser's address bar and title bar on mobile, per mode. |
| `defaultMode` | `light`, `dark` or `system`. Applies until a user picks a mode in the Appearance menu. |
| `lockMode` | `light` or `dark` forces that mode and hides the mode choices. `null` (the default) leaves the choice to the user. |
| `themes` | Up to 8 named themes the Appearance menu offers next to Default. An `id` uses lowercase letters, digits and `-`, starts with a letter, is at most 32 characters, and is not `default`, `light`, `dark` or `high-contrast`. A `label` is at most 40 characters. |
| `defaultTheme` | `default`, or one of the theme ids. Applies until a user picks a theme. |

Logo and favicon values are file names, never URLs: the UI loads them from the branding folder only. The UI ignores any key that fails these rules and logs a browser console warning that names it.

## theme.css

`theme.css` overrides the CSS variables that `frontend/src/index.css` declares. Color values are space-separated RGB channels, not hex, so Tailwind opacity modifiers keep working:

```css
/* Light mode, plus the color scales, which both modes share. */
:root {
  --primary-50: 239 246 255;
  --primary-600: 13 110 253;
  --primary-700: 10 88 202;
  --radius-card: 0.5rem;
}

/* Dark mode only. */
html.dark {
  --color-accent-text: 110 168 254;
}

/* A named theme from manifest.json: in both modes, light only, dark only. */
html.acme-night { --primary-600: 79 70 229; }
html.acme-night:not(.dark) { --color-canvas: 245 246 255; }
html.acme-night.dark { --color-surface: 17 24 39; }
```

These variables are supported:

| Group | Variables |
|---|---|
| Brand | `--primary-50` through `--primary-950`. They color buttons, links, focus rings, active tabs and selected rows. |
| Neutrals | `--gray-25`, `--gray-50` through `--gray-950`. |
| Status | `--success-*`, `--warning-*`, `--danger-*`, `--info-*`, steps 50 to 950. `--info-*` also colors the links, tags and toggles on ordinary server cards. |
| Entity types | `--server-*` (Anthropic registry server cards; follows the brand unless you set it), `--agent-*`, `--skill-*`, `--virtual-*`, `--custom-*` (custom entity types), steps 50 to 950. |
| Roles that differ per mode | `--color-canvas`, `--color-surface`, `--color-surface-muted`, `--color-surface-hover`, `--color-surface-overlay`, `--color-line`, `--color-line-strong`, `--color-fg`, `--color-fg-muted`, `--color-fg-subtle`, `--color-accent`, `--color-accent-hover`, `--color-accent-fg`, `--color-accent-text`, `--color-accent-soft`, `--color-accent-soft-fg`, `--color-focus`, `--color-danger`, `--color-danger-hover`, `--color-danger-soft`, `--color-danger-soft-fg`, `--shadow-color`. |
| Shape and type | `--radius-control` and `--radius-card` (CSS lengths), `--font-sans` and `--font-mono` (font-family lists), `--focus-ring-width`. |

The project treats this list as a public contract: renaming a variable is a breaking change and goes in the release notes. Rules that target the UI's own class names or HTML structure are not supported and can break on any upgrade.

### How the overrides apply

The UI marks the active state with classes on the `<html>` element: `dark` in dark mode, the theme id when a named theme is active, and `high-contrast` when the user turns that on. `index.css` declares each variable once for light mode under `:root`. It declares a second value under `html.dark` for the variables that change in dark mode (the role variables in the table above), and a third set for high contrast.

The registry loads `theme.css` after `index.css`. When two rules set the same variable, the browser uses the one with the more specific selector, and between equally specific selectors it uses the later one, which is yours. That gives these results:

| Your selector | Where it applies |
|---|---|
| `:root` | Light mode. Also dark mode for any variable `index.css` does not redefine under `html.dark`, which covers every color scale (`--primary-*`, `--gray-*` and the rest), the corner radii and the fonts. A role variable such as `--color-canvas` set here changes light mode only, because the default `html.dark` rule is more specific. |
| `html.dark` | Dark mode, for every variable. |
| `html.<theme-id>` | That theme in both modes. It outranks the default `html.dark` values too, so a role variable set here also replaces the dark-mode default. |
| `html.<theme-id>:not(.dark)` | That theme in light mode only. |
| `html.<theme-id>.dark` | That theme in dark mode only. |

Set a role variable for both modes explicitly: once under `:root` (or `html.<theme-id>:not(.dark)`) and once under `html.dark` (or `html.<theme-id>.dark`). Setting it in only one place is the most common reason an override seems to have no effect in the other mode.

### High contrast

High contrast is a user setting that combines with any theme. `index.css` implements it with two rules, `html.high-contrast:not(.dark)` and `html.high-contrast.dark`. They replace the middle of the gray scale (`--gray-200` to `--gray-600` in light mode, `--gray-300` to `--gray-600` in dark mode) with darker or lighter values, and repoint `--color-fg-muted`, `--color-fg-subtle`, `--color-line` and `--color-line-strong` at those steps.

Two things follow from that:

- If you replace the gray scale, high contrast still swaps in the default cool grays for those steps, so your neutrals change hue when a user turns it on. To keep your grays, add the same two selectors to `theme.css` with high-contrast values taken from your own scale: in light mode, each step two or three steps darker than normal; in dark mode, two steps lighter.
- A named theme's `html.<theme-id>:not(.dark)` and `html.<theme-id>.dark` rules are as specific as the high-contrast rules and load later, so they win. If one of them sets `--color-fg-muted`, `--color-fg-subtle`, `--color-line`, `--color-line-strong` or the middle grays, that theme loses part of high contrast. Either leave those variables alone in the theme, or add `html.<theme-id>.high-contrast:not(.dark)` and `html.<theme-id>.high-contrast.dark` rules with stronger values.

### Values that do not follow the color scales

Most role variables point at a step of a scale (`--color-accent` is `var(--primary-600)`, `--color-line` is `var(--gray-200)` in light mode), so changing the scale changes them. These hold fixed values in `index.css` and do not change when you change a scale:

| Variable | Fixed in | What it is | When to set it |
|---|---|---|---|
| `--color-surface`, `--color-surface-overlay` | light mode | White card and menu backgrounds. | When you want tinted cards in light mode. In dark mode they follow `--gray-800`. |
| `--color-accent-fg` | both modes | White text on brand buttons. | When your brand color is too light for white text. |
| `--color-accent-soft` | dark mode | Background of selected items and brand pills. | Whenever you change `--primary-*` or `--gray-800`. |
| `--color-danger`, `--color-danger-hover`, `--color-danger-soft`, `--color-danger-soft-fg` | both modes | Destructive buttons and error banners. | Whenever you change `--danger-*`. Changing the `--danger-*` scale alone recolors danger badges and text but not these. |
| `--shadow-color` | both modes | Shadow tint. | Rarely. |

The two dark-mode soft backgrounds are solid colors rather than transparent ones so that opacity modifiers keep working on them. Each is step 900 of its scale mixed over the dark card color, `--gray-800`, at 45 percent. For each of the red, green and blue channels, compute `round(0.45 × step900 + 0.55 × gray800)`. With a brand step 900 of `117 23 66` and the default `--gray-800` of `29 41 57`, that gives `69 33 61`.

```css
html.dark {
  --color-accent-soft: 69 33 61;
}
```

### Corners and fonts

`--radius-control` sets the corners of buttons, text inputs and dropdowns. `--radius-card` sets the corners of panels drawn with the shared card style, such as the audit log filter bar and grid, the Settings panels and the Settings navigation. Other elements keep fixed corners whatever you set: the dashboard's server, agent, skill and virtual server cards, most modal dialogs, badges and tags.

`--font-sans` sets the typeface of all UI text, including the audit grid. `--font-mono` sets the typeface of paths, code samples and other monospaced text. Text inside an SVG logo uses the font named in the SVG itself.

To use your own font, put the `.woff2` file in the folder and refer to it with a relative URL, which resolves against `theme.css` and so also works under a `ROOT_PATH`:

```css
@font-face {
  font-family: 'Acme Sans';
  src: url('acme-sans.woff2') format('woff2');
  font-weight: 100 900;
  font-display: swap;
}
:root { --font-sans: 'Acme Sans', ui-sans-serif, system-ui, sans-serif; }
```

### Generating a theme from one color

`scripts/generate-ui-theme.py` builds the 11-step ramps from a single brand color and checks the text contrast that results:

```bash
uv run python scripts/generate-ui-theme.py --brand '#0d6efd' --output-dir ./branding --manifest
uv run python scripts/generate-ui-theme.py --brand '#0d6efd' --success '#198754' --strict
uv run python scripts/generate-ui-theme.py --brand '#6f42c1' --theme-id acme-night --output-dir ./branding --force
```

It writes `theme.css` (and with `--manifest` a `manifest.json` skeleton) and prints a WCAG contrast table for the pairs the UI relies on, such as white text on the brand button and brand-colored links on a dark card. `--strict` exits non-zero and writes nothing when a pair fails AA. Run `--help` for every option.

The generator covers the color scales and the values that depend on them: `--color-accent-soft` for dark mode, and the four `--color-danger*` variables when you pass `--danger`. It leaves the rest to you:

- It does not generate grays. It computes the dark-mode soft backgrounds against the default `--gray-800`. If you also replace the gray scale, recompute those values with the formula in [Values that do not follow the color scales](#values-that-do-not-follow-the-color-scales) and override them after the generated block.
- It does not write high-contrast rules. Add them by hand if you replace the grays (see [High contrast](#high-contrast)).
- It does not set corners, fonts or role variables such as `--color-canvas`. Add those by hand below the generated block, so a later re-run with `--force` is easy to merge back.

## Delivering the folder

The registry reads the folder from `UI_BRANDING_DIR`, which defaults to `/app/branding` in the container. The registry reads the files on every request and never caches them, so a changed file shows up without a restart. Browsers cache branding files for 5 minutes (`Cache-Control: max-age=300`).

### Kubernetes (Helm)

Put the files in a ConfigMap, using `binaryData` for images such as PNG and ICO, and point the registry chart at it:

```bash
kubectl create configmap acme-branding --from-file=./branding/
```

```yaml
registry:
  branding:
    enabled: true
    existingConfigMap: acme-branding
```

The chart mounts the whole ConfigMap read-only as a directory and sets `UI_BRANDING_DIR` to it. Editing the ConfigMap updates the running pods within a minute or two, once the kubelet syncs the volume, with no restart. A ConfigMap holds at most 1 MiB, which is plenty for CSS, a manifest and SVG logos. Use `registry.branding.*` on the stack chart and `branding.*` on the registry chart installed on its own.

### Docker Compose

The compose files mount `./branding` from the repository root into the registry container, read-only. Put your files there, or set `MCP_BRANDING_DIR` to another folder before starting the stack. The repository ships the folder empty and git ignores anything you add to it.

### Amazon ECS (Terraform)

ECS has no ConfigMaps, so you pass the folder as a base64-encoded `.tar.gz` in the Terraform variable `ui_branding_archive_b64`. The registry container receives it as `UI_BRANDING_ARCHIVE_B64` and unpacks it into `UI_BRANDING_DIR` at startup:

```bash
tar -czf - -C ./branding . | base64 | tr -d '\n'
```

The archive must contain only regular files at its top level, at most 32 of them and at most 2 MiB uncompressed. macOS metadata entries (`._*`, `.DS_Store`) are skipped. Any other problem stops the container at startup with a message that says what to fix. Changing the archive takes a new task deployment. You can set `UI_BRANDING_ARCHIVE_B64` on other surfaces too, but not together with a mounted folder: the container refuses to start when both are present.

## Path-based routing

Under a `ROOT_PATH` such as `/registry`, the UI requests `/registry/branding/...`. That path sits under the registry's own prefix, so the ingress rule that already routes `/registry` to the registry covers it and no extra rule is needed. Keep URLs inside `theme.css` relative, as in the font example, so they resolve under the prefix too.

## Security

Only the people who deploy the registry should supply the branding folder. It gets the same trust as the RUM snippet (`docs/frontend-rum.md`): CSS can load images from other origins, and an attacker who controls `theme.css` could use that to leak what pages a user visits. The registry adds these limits:

- It serves only allowed file names and extensions from inside the folder, and refuses anything that resolves outside it.
- It sends `X-Content-Type-Options: nosniff` on every file.
- It sends a restrictive `Content-Security-Policy` with `sandbox` on SVG files, so script inside an SVG opened directly in a tab does not run.
- The UI shows logos with `<img>`, which never runs script inside an SVG.
- `branding` is a reserved path: you cannot register a server or agent at `/branding`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| A manifest setting has no effect | Open the browser console. The UI logs `Branding manifest: ignored invalid or unknown keys: ...` naming each key it dropped. |
| Colors do not change | Check that `{ROOT_PATH}/branding/theme.css` returns your file and that values are RGB channels (`13 110 253`), not hex. |
| An override works in one mode but not the other | Role variables need a value for each mode: one under `:root` and one under `html.dark` (see [How the overrides apply](#how-the-overrides-apply)). |
| Selected items or error banners keep the old color in dark mode | `--color-accent-soft` and `--color-danger*` hold fixed values. Recompute them (see [Values that do not follow the color scales](#values-that-do-not-follow-the-color-scales)). |
| Grays change hue when a user turns on high contrast | High contrast swaps in the default grays. Add your own `html.high-contrast:not(.dark)` and `html.high-contrast.dark` rules (see [High contrast](#high-contrast)). |
| Changing `--radius-card` does not round the dashboard cards | Those cards and most dialogs keep fixed corners (see [Corners and fonts](#corners-and-fonts)). |
| A logo shows as a broken image | The file name in `manifest.json` must match a file in the folder exactly, with an image extension. |
| The registry container exits at startup with a branding error | `UI_BRANDING_ARCHIVE_B64` failed validation, or it is set while a branding folder is also mounted. The message names the file or setting to fix. |
| A changed ConfigMap does not show up | Kubernetes updates the mounted folder within a minute or two; then the browser cache adds up to 5 minutes. Hard-reload the page. |
