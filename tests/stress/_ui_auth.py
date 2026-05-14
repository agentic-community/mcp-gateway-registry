"""Playwright login fixture for the registry UI.

Drives the OAuth/Keycloak login flow once, saves the browser storage state
(cookies + localStorage) to `.oauth-tokens/playwright-storage-state.json`,
and reuses it on subsequent calls. Re-login when the saved state is
missing, malformed, or `force_relogin=True`.

The registry SPA at `/` redirects unauthenticated users to `/login`,
where Login.tsx renders an OAuth-provider button per the
`/api/auth/providers` response. Clicking the Keycloak provider triggers
`window.location.href = ${authUrl}/oauth2/login/keycloak?redirect_uri=...`,
which redirects through the auth-server to Keycloak's login form. Once
the user submits the form, Keycloak redirects back through the auth-server
to the SPA, with the session cookie set.

Default credentials come from .env (`INITIAL_ADMIN_PASSWORD` for the
`admin` user). Most local setups copy the placeholder password from
`.env.example` verbatim, in which case Keycloak rejects the login --
the fixture saves a diagnostic screenshot under
`tests/stress/data/.cache/ui-login-stuck.png` so the operator can see
what's happening. The cure: set `STRESS_UI_USERNAME` and
`STRESS_UI_PASSWORD` env vars to a user that actually exists in the
`mcp-gateway` realm.

Smoke-test the fixture standalone:

    export STRESS_UI_USERNAME=admin
    export STRESS_UI_PASSWORD=<your real password>
    uv run python -m tests.stress._ui_auth --base-url http://localhost
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from playwright.async_api import Page, async_playwright

from tests.stress.config import default_base_url, project_root

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_BASENAME: str = "playwright-storage-state.json"


def default_storage_path() -> Path:
    return project_root() / ".oauth-tokens" / DEFAULT_STORAGE_BASENAME


def resolve_ui_credentials() -> tuple[str, str]:
    """Pick UI login credentials from env, falling back to .env defaults."""
    username = os.getenv("STRESS_UI_USERNAME", "admin")
    password = (
        os.getenv("STRESS_UI_PASSWORD")
        or os.getenv("INITIAL_ADMIN_PASSWORD")
        or "changeme"
    )
    return username, password


def _is_state_valid(path: Path) -> bool:
    """Return True iff the storage state file exists and has cookies/origins."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(data.get("cookies")) or bool(data.get("origins"))


async def _wait_for_authed(page: Page, base_url: str, timeout_ms: int = 30_000) -> None:
    """Poll until `page.url` is back on the app (no `/login`, no Keycloak)."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_ms / 1000
    app_origin = base_url.rstrip("/")
    while loop.time() < deadline:
        cur = page.url
        bad = "/login" in cur or "/realms/" in cur or "/oauth2/login" in cur
        if cur.startswith(app_origin) and not bad:
            return
        await asyncio.sleep(0.5)

    # Capture a diagnostic screenshot before failing so the operator can see
    # what Keycloak is showing (e.g. an MFA prompt, required-action page,
    # error banner from bad creds, etc.).
    diag = project_root() / "tests" / "stress" / "data" / ".cache" / "ui-login-stuck.png"
    diag.parent.mkdir(parents=True, exist_ok=True)
    try:
        await page.screenshot(path=str(diag), full_page=True)
        logger.error("Saved login-failure screenshot to %s", diag)
    except Exception as exc:
        logger.error("Could not save diagnostic screenshot: %s", exc)
    raise TimeoutError(
        f"Did not land back at the app after login. Current URL: {page.url}. "
        f"Check {diag} for what Keycloak is showing. If the screenshot shows "
        f"'Invalid username or password.', set STRESS_UI_USERNAME and "
        f"STRESS_UI_PASSWORD to a Keycloak user that exists in the "
        f"`mcp-gateway` realm."
    )


async def _drive_login(
    base_url: str,
    username: str,
    password: str,
    headless: bool,
) -> dict:
    """Click through the OAuth+Keycloak login flow and return the storage state."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()

        login_url = f"{base_url.rstrip('/')}/login"
        logger.info("Navigating to %s", login_url)
        await page.goto(login_url, wait_until="networkidle")

        # The OAuth provider button is rendered after /api/auth/providers
        # resolves. The visible text contains the provider name; we look for
        # any of "keycloak" (case-insensitive) on a button.
        keycloak_button = page.get_by_role("button").filter(has_text="keycloak")
        if await keycloak_button.count() == 0:
            keycloak_button = page.get_by_role("button").filter(has_text="Keycloak")
        await keycloak_button.first.click()

        # Keycloak's login form has standard input names.
        await page.wait_for_selector("input[name='username']", timeout=20_000)
        await page.fill("input[name='username']", username)
        await page.fill("input[name='password']", password)

        submit = page.locator(
            "input[type='submit'], button[type='submit'], #kc-login"
        ).first
        await submit.click()

        await _wait_for_authed(page, base_url)
        await page.wait_for_load_state("networkidle")

        state: dict = await context.storage_state()
        await context.close()
        await browser.close()
        return state


async def get_storage_state(
    base_url: str,
    storage_path: Path | None = None,
    force_relogin: bool = False,
    headless: bool = True,
    username: str | None = None,
    password: str | None = None,
) -> Path:
    """Ensure a valid Playwright storage_state file exists; return its path.

    On a cache hit, the saved file is reused (assumed valid until proven
    otherwise). On a cache miss or `force_relogin=True`, the Keycloak
    login flow runs and the result is written to disk.
    """
    path = storage_path or default_storage_path()

    if not force_relogin and _is_state_valid(path):
        logger.info("Reusing Playwright storage state: %s", path)
        return path

    reason = "force_relogin=True" if force_relogin else "no valid cached state"
    logger.info("%s; running Keycloak login flow", reason)

    user, pw = (username or resolve_ui_credentials()[0]), (
        password or resolve_ui_credentials()[1]
    )
    state = await _drive_login(base_url, user, pw, headless=headless)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, default=str))
    logger.info("Saved Playwright storage state: %s", path)
    return path


# ---------------------------------------------------------------------------
# Smoke-test CLI: `python -m tests.stress._ui_auth --base-url http://localhost`
# ---------------------------------------------------------------------------


async def _smoke(args: argparse.Namespace) -> int:
    path = await get_storage_state(
        base_url=args.base_url,
        force_relogin=args.force_relogin,
        headless=not args.no_headless,
    )

    # Verify by opening a new context with the saved state and screenshotting `/`.
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.no_headless)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            storage_state=str(path),
        )
        page = await context.new_page()
        await page.goto(args.base_url.rstrip("/") + "/", wait_until="networkidle")
        shot = project_root() / "tests" / "stress" / "data" / ".cache" / "ui-smoke.png"
        shot.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(shot), full_page=True)
        logger.info("Wrote smoke screenshot: %s", shot)
        await context.close()
        await browser.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument(
        "--force-relogin",
        action="store_true",
        help="Delete the cached storage state and re-run the login flow.",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run the browser visibly (default: headless).",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
    )
    return asyncio.run(_smoke(args))


if __name__ == "__main__":
    sys.exit(main())
