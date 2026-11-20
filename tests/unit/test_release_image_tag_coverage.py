"""Release coverage check for the image tag a chart advertises.

`.github/workflows/helm-chart-update.yml` bumps `global.image.tag` after a
release, over a HARDCODED list of values files. A chart that pins
`global.image.tag` but is missing from that list keeps advertising the previous
release's image forever, and nothing fails — the workflow reports success because
the files it does know about were updated.

That is not cosmetic. `charts/mongodb-configure` runs the registry image to
execute `scripts/init-mongodb-ce.py`, so a stale tag there means `helm upgrade`
runs an OLD database init script against the current schema — exactly the class of
silent drift that deleting the chart's embedded copy of that script was meant to
end.

So the list is asserted against the charts rather than trusted.
"""

import re
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

WORKFLOW = ".github/workflows/helm-chart-update.yml"
CHARTS_DIR = "charts"


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Repository root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def workflow_text(repo_root: Path) -> str:
    """Raw workflow source.

    Read as text, not YAML: the values-file list lives inside a shell `run:`
    block, so it is a string as far as YAML is concerned.
    """
    path = repo_root / WORKFLOW
    assert path.is_file(), f"{WORKFLOW} is missing"
    return path.read_text()


def _charts_pinning_global_tag(repo_root: Path) -> dict[str, str]:
    """Map each chart values.yaml to the `global.image.tag` it pins."""
    pinned: dict[str, str] = {}
    for values_file in sorted((repo_root / CHARTS_DIR).glob("*/values.yaml")):
        try:
            data = yaml.safe_load(values_file.read_text())
        except yaml.YAMLError:  # pragma: no cover - a chart with unparseable values
            continue
        if not isinstance(data, dict):
            continue
        tag = ((data.get("global") or {}).get("image") or {}).get("tag")
        if isinstance(tag, str) and tag.strip():
            pinned[str(values_file.relative_to(repo_root))] = tag.strip()
    return pinned


def _update_loop_targets(workflow_text: str) -> str:
    """The values files listed in the workflow's `for VALUES_FILE in ...` loop.

    Scoped deliberately: every path also appears in the PR body, so searching the
    whole file would pass even for a chart the loop never touches.
    """
    match = re.search(r"for VALUES_FILE in\s*\\\n(.*?);\s*do", workflow_text, re.S)
    assert match, f"{WORKFLOW} no longer has a `for VALUES_FILE in ...` loop"
    return match.group(1)


def test_every_chart_pinning_a_tag_is_updated_on_release(
    repo_root: Path, workflow_text: str
) -> None:
    """A chart that pins `global.image.tag` must be in the workflow's update loop."""
    pinned = _charts_pinning_global_tag(repo_root)
    assert pinned, "no chart pins global.image.tag - has the convention changed?"

    targets = _update_loop_targets(workflow_text)
    missing = [path for path in pinned if path not in targets]
    assert not missing, (
        "These charts pin global.image.tag but are absent from the update loop in "
        f"{WORKFLOW}, so they will keep advertising the previous release's "
        f"image after a version bump: {missing}"
    )


def test_release_pr_body_lists_every_file_it_updates(repo_root: Path, workflow_text: str) -> None:
    """The PR body must name each values file, so a reviewer sees the real scope."""
    pinned = _charts_pinning_global_tag(repo_root)
    body = workflow_text.split("Updated files:", 1)
    assert len(body) == 2, f"{WORKFLOW} PR body no longer lists updated files"

    undocumented = [path for path in pinned if f"`{path}`" not in body[1]]
    assert not undocumented, (
        f"{WORKFLOW} updates these files but does not list them in the PR body: {undocumented}"
    )


def test_all_pinned_tags_agree(repo_root: Path) -> None:
    """Every chart must pin the SAME version.

    They are all the one application image. A chart left behind by a release is
    most visible as a tag that disagrees with its siblings, so assert that
    directly rather than inferring it from the workflow.
    """
    pinned = _charts_pinning_global_tag(repo_root)
    distinct = sorted(set(pinned.values()))
    assert len(distinct) == 1, f"charts disagree on the app image tag: {pinned}"
