"""Wiring checks for the MongoDB initialization Job and its script.

`charts/mongodb-configure` used to embed a full copy of
`scripts/init-mongodb-ce.py` in a ConfigMap, because a Helm chart cannot
`.Files.Get` a path outside its own directory. That copy drifted from the source
in six functional ways: it was missing the audit identity-claim indexes, the
`agent_skills` indexes, the TTL `IndexOptionsConflict` recovery and MongoDB CE
no-auth support; it never loaded the `federation-service.json` scope; it shipped
scope seeds whose `group_mappings` were transposed (the exact regression fixed
for issue #1574, which leaves every administrator with zero server access); and
it still created a SINGLE-field unique index on `request_id`, so on a
Helm-deployed MongoDB CE the second audit record for a request was rejected and
dropped.

The Job now runs the registry image's own `/app/scripts/init-mongodb-ce.py`, so
there is no copy to drift. These assertions keep it that way, and check the
things the old byte-parity test could not: that the path the Job executes really
exists, that everything the script reads is shipped alongside it, and that the
two storage backends name the same logical index identically.
"""

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest
import yaml

pytestmark = [pytest.mark.unit]

MONGODB_CE_SCRIPT = "scripts/init-mongodb-ce.py"
DOCUMENTDB_SCRIPT = "scripts/init-documentdb-indexes.py"
CHART_DIR = "charts/mongodb-configure"
CHART_JOB = f"{CHART_DIR}/templates/job.yaml"
REGISTRY_DOCKERFILE = "docker/Dockerfile.registry"


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Repository root directory."""
    return Path(__file__).parent.parent.parent


def _load_script(repo_root: Path, relative: str) -> ModuleType:
    """Import one of the standalone init scripts as a module."""
    path = repo_root / relative
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    assert spec and spec.loader, f"cannot load {relative}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render_stub(template_text: str) -> str:
    """Make a Helm template parseable as YAML, without running Helm.

    Helm comment blocks (`{{- /* ... */ -}}`), which may span lines, are removed
    first; then standalone action lines are dropped and inline actions become a
    scalar placeholder. Only structure is asserted, so the substituted values do
    not matter.
    """
    without_comments = re.sub(r"\{\{-?\s*/\*.*?\*/\s*-?\}\}", "", template_text, flags=re.S)
    kept = []
    for line in without_comments.splitlines():
        if not line.strip() or line.lstrip().startswith("{{"):
            continue
        kept.append(re.sub(r"\{\{.*?\}\}", "stub", line))
    return "\n".join(kept)


def test_no_chart_template_embeds_the_init_script(repo_root: Path) -> None:
    """No chart may carry a second copy of an init script.

    This is the invariant the byte-parity test used to enforce the hard way.
    Reintroducing a copy reintroduces the drift, so fail on the shapes that a
    copy takes: a `script.py`-style ConfigMap key, or Python from the script
    pasted into a template.
    """
    offenders: list[str] = []
    for template in (repo_root / "charts").rglob("templates/*.yaml"):
        text = template.read_text()
        rel = template.relative_to(repo_root)
        if re.search(r"^\s{2,}(script|wait)\.py:\s*\|", text, re.M):
            offenders.append(f"{rel}: embeds a script under a *.py ConfigMap key")
        # Signatures unique to the init scripts, not to Helm YAML.
        for marker in ("_initialize_replica_set", "replSetInitiate", "AsyncIOMotorClient"):
            if marker in text:
                offenders.append(f"{rel}: contains init-script code ({marker})")
    assert not offenders, (
        "A chart is duplicating an init script again. Run the script from the "
        "registry image instead, which already ships it:\n  " + "\n  ".join(offenders)
    )


def test_job_executes_a_script_path_that_exists(repo_root: Path) -> None:
    """The Job's command must point at a real script in this repository."""
    job = yaml.safe_load(_render_stub((repo_root / CHART_JOB).read_text()))
    container = job["spec"]["template"]["spec"]["containers"][0]
    command = container["command"]

    script_args = [a for a in command if a.endswith(".py")]
    assert script_args, f"{CHART_JOB} command runs no script: {command}"
    for arg in script_args:
        # The image lays the repository out under /app.
        relative = arg.removeprefix("/app/")
        assert (repo_root / relative).is_file(), (
            f"{CHART_JOB} runs {arg}, which is not {relative} in this repository"
        )


def test_registry_image_ships_the_script_and_its_inputs(repo_root: Path) -> None:
    """The runtime image must contain the whole scripts/ directory.

    The Job runs the script straight out of the image, and the script resolves
    its scope seeds with `Path(__file__).parent`, so a partial copy would make a
    scope silently absent rather than fail.
    """
    dockerfile = (repo_root / REGISTRY_DOCKERFILE).read_text()
    assert re.search(r"^COPY .*\s/app/scripts\s*$", dockerfile, re.M), (
        f"{REGISTRY_DOCKERFILE} no longer copies scripts/ into /app/scripts, so the "
        f"{CHART_JOB} command cannot resolve"
    )


def test_every_scope_file_the_script_loads_is_present(repo_root: Path) -> None:
    """Each seed file the script loads must sit beside it.

    A missing file is only a `logger.warning` followed by `continue`, so the
    scope is never created and nothing fails. That is how the chart's copy came
    to mount four scope files while the script loaded five.
    """
    module = _load_script(repo_root, MONGODB_CE_SCRIPT)
    source = (repo_root / MONGODB_CE_SCRIPT).read_text()

    # The list lives inside _load_default_scopes; read it from the source.
    block = re.search(r"scope_files\s*=\s*\[(.*?)\]", source, re.S)
    assert block, f"{MONGODB_CE_SCRIPT} no longer declares a scope_files list"
    filenames = re.findall(r'"([^"]+\.json)"', block.group(1))
    assert filenames, "no scope filenames parsed"

    script_dir = (repo_root / MONGODB_CE_SCRIPT).parent
    missing = [name for name in filenames if not (script_dir / name).is_file()]
    assert not missing, f"{MONGODB_CE_SCRIPT} loads scope files that do not exist: {missing}"
    assert module is not None


def test_both_backends_name_the_claim_indexes_identically(repo_root: Path) -> None:
    """The two init scripts must agree on every audit claim index name.

    They previously let the engine auto-generate names, which produced different
    names per backend for the same logical index -- which is why an earlier
    migration had to guess at name variants in order to drop one. Names are now
    explicit, and must stay in step so a migration can address an index by name
    on either backend.
    """
    ce = _load_script(repo_root, MONGODB_CE_SCRIPT)
    doc = _load_script(repo_root, DOCUMENTDB_SCRIPT)

    assert ce.AUDIT_CLAIM_FIELDS == doc.AUDIT_CLAIM_FIELDS
    assert ce.AUDIT_CLAIM_NESTED_LOG_TYPES == doc.AUDIT_CLAIM_NESTED_LOG_TYPES
    assert ce.AUDIT_FLAT_LOG_TYPE == doc.AUDIT_FLAT_LOG_TYPE

    def names(module: ModuleType) -> set[str]:
        out = {
            module._audit_claim_index_name(log_type, field)
            for log_type in module.AUDIT_CLAIM_NESTED_LOG_TYPES
            for field in module.AUDIT_CLAIM_FIELDS
        }
        out |= {
            module._audit_claim_index_name(module.AUDIT_FLAT_LOG_TYPE, field)
            for field in ("username", *module.AUDIT_CLAIM_FIELDS)
        }
        return out

    ce_names, doc_names = names(ce), names(doc)
    assert ce_names == doc_names, (
        f"claim index names diverged between backends: "
        f"only in CE {sorted(ce_names - doc_names)}, "
        f"only in DocumentDB {sorted(doc_names - ce_names)}"
    )
    # Amazon DocumentDB has historically capped index names at 63 characters.
    too_long = sorted(n for n in ce_names if len(n) > 63)
    assert not too_long, f"index names too long for DocumentDB: {too_long}"


def test_audit_retention_cannot_be_shortened_by_default(repo_root: Path) -> None:
    """The TTL must not shrink without an explicit opt-in.

    Reducing the audit TTL makes MongoDB delete every record older than the new
    window, normally within a minute. The Job runs on every `helm upgrade`, so a
    default-valued TTL silently resetting an operator's longer retention is a
    data-loss bug, not a configuration nit.
    """
    for relative in (MONGODB_CE_SCRIPT, DOCUMENTDB_SCRIPT):
        source = (repo_root / relative).read_text()
        assert "AUDIT_LOG_MONGODB_TTL_ALLOW_SHRINK" in source, (
            f"{relative} no longer guards against shortening audit retention"
        )


def test_registry_api_stream_gets_no_claim_index(repo_root: Path) -> None:
    """`registry_api_access` must not appear among the claim index targets.

    Its records nest the claim fields the same way `mcp_server_access` does, but
    the auth server hands the registry a thin signed assertion rather than raw IdP
    claims, so those fields are permanent nulls on that stream and the audit API
    does not search them there (`_identity_search_clause` in
    registry/audit/routes.py returns the display username alone for it). Since it
    is also the largest stream, indexing four always-null fields on it is the most
    expensive way to serve no query. Both backends must agree on the exclusion.
    """
    for relative in (MONGODB_CE_SCRIPT, DOCUMENTDB_SCRIPT):
        module = _load_script(repo_root, relative)

        assert "registry_api_access" not in module.AUDIT_CLAIM_NESTED_LOG_TYPES, (
            f"{relative} indexes claim fields on registry_api_access, where they are "
            "always null and never queried"
        )
        names = {
            module._audit_claim_index_name(log_type, field)
            for log_type in module.AUDIT_CLAIM_NESTED_LOG_TYPES
            for field in module.AUDIT_CLAIM_FIELDS
        }
        assert not any(name.startswith("audit_claim_api_") for name in names), names

        # And a cluster that already built them must have them dropped.
        for field in module.AUDIT_CLAIM_FIELDS:
            assert f"audit_claim_api_{field}_idx" in module.LEGACY_AUDIT_CLAIM_INDEXES, (
                f"{relative} does not retire the superseded audit_claim_api_{field}_idx"
            )
