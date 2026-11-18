"""Opt-in PostgreSQL 17 bootstrap/restore behavior against the official image."""

import hashlib
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_KEYCLOAK_POSTGRES_INTEGRATION") != "1",
    reason="requires Docker and postgres:17.12; opt in explicitly",
)
FILES = Path(__file__).resolve().parents[2] / "charts/keycloak/files"
IMAGE = "postgres:17.12"


def docker(*args: str, check: bool = True, input: str | None = None):
    return subprocess.run(
        ["docker", *args], input=input, text=True, capture_output=True, check=check, timeout=180
    )


@pytest.fixture
def postgres(tmp_path: Path):
    containers: list[str] = []
    volume = "kc-bootstrap-test-" + uuid.uuid4().hex
    docker("volume", "create", volume)

    def start(
        mode: str,
        snapshot: Path | None = None,
        digest: str = "",
        *,
        username: str = "keycloak",
        database: str = "keycloak",
        existing_verified: bool = False,
    ) -> str:
        name = "kc-bootstrap-test-" + uuid.uuid4().hex
        containers.append(name)
        args = [
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "none",
            "--entrypoint",
            "/bin/bash",
            "-v",
            f"{volume}:/var/lib/postgresql/data",
            "-v",
            f"{FILES}:/opt/keycloak-bootstrap:ro",
            "-e",
            "PGDATA=/var/lib/postgresql/data/pgdata",
            "-e",
            "POSTGRES_USER=postgres",
            "-e",
            "POSTGRES_PASSWORD=test-only-admin-password",
            "-e",
            f"KEYCLOAK_DB_USER={username}",
            "-e",
            f"KEYCLOAK_DB_NAME={database}",
            "-e",
            "KEYCLOAK_DB_PASSWORD=test-only-role-password",
            "-e",
            f"BOOTSTRAP_MODE={mode}",
            "-e",
            f"BOOTSTRAP_SHA256={digest}",
            "-e",
            "BOOTSTRAP_SOURCE_QUIESCED=true",
            "-e",
            f"BOOTSTRAP_EXISTING_VERIFIED={str(existing_verified).lower()}",
        ]
        if snapshot:
            args.extend(["-v", f"{snapshot}:/migration:ro"])
        # Initialize ownership like the pod's fsGroup, then execute as postgres.
        args.extend(
            [
                IMAGE,
                "-c",
                "chown postgres:postgres /var/lib/postgresql/data; "
                "exec gosu postgres /bin/bash /opt/keycloak-bootstrap/bootstrap.sh",
            ]
        )
        docker(*args)
        return name

    yield start
    for name in containers:
        docker("rm", "-f", name, check=False)
    docker("volume", "rm", volume, check=False)


def ready(name: str):
    for _ in range(120):
        if docker("exec", name, "pg_isready", "-h", "127.0.0.1", check=False).returncode == 0:
            return
        if docker("inspect", "-f", "{{.State.Running}}", name).stdout.strip() == "false":
            pytest.fail(docker("logs", name).stdout + docker("logs", name).stderr)
        time.sleep(0.25)
    pytest.fail(docker("logs", name).stdout + docker("logs", name).stderr)


def sql(name: str, query: str, database: str = "keycloak", **variables: str):
    return docker(
        "exec",
        "-i",
        name,
        "psql",
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        "postgres",
        "-d",
        database,
        "-At",
        *(argument for key, value in variables.items() for argument in ("-v", f"{key}={value}")),
        input=query,
    ).stdout.strip()


def assert_database_identity(name: str, username: str, database: str):
    assert (
        sql(
            name,
            "SELECT r.rolname::text || ':' || d.datname::text FROM pg_roles r "
            "JOIN pg_database d ON d.datdba = r.oid "
            "WHERE d.datname::text = :'selected_db' AND r.rolcanlogin "
            "AND NOT r.rolsuper AND NOT r.rolcreatedb AND NOT r.rolcreaterole "
            "AND NOT r.rolreplication AND NOT r.rolbypassrls",
            database,
            selected_db=database,
        )
        == f"{username}:{database}"
    )


def refused_identity(name: str):
    assert docker("wait", name).stdout.strip() != "0"
    assert (
        "existing nonprivileged LOGIN role that owns KEYCLOAK_DB_NAME"
        in docker("logs", name).stderr
    )


def archive(tmp_path: Path, valid: bool) -> tuple[Path, str]:
    name = "kc-source-test-" + uuid.uuid4().hex
    source_volume = name + "-data"
    docker("volume", "create", source_volume)
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "none",
            "-e",
            "POSTGRES_PASSWORD=test-only",
            "-e",
            "POSTGRES_DB=keycloak",
            "-v",
            f"{source_volume}:/var/lib/postgresql/data",
            IMAGE,
        )
        ready(name)
        sql(
            name,
            "CREATE TABLE realm(id text); INSERT INTO realm VALUES ('retained-realm'); "
            "CREATE TABLE databasechangelog(id text); "
            "INSERT INTO databasechangelog VALUES ('schema-created'); "
            "CREATE TABLE user_entity(id text); "
            "INSERT INTO user_entity VALUES ('retained-user');",
        )
        if valid:
            sql(
                name, "CREATE TABLE client(id text); INSERT INTO client VALUES ('retained-client');"
            )
        docker(
            "exec",
            name,
            "pg_dump",
            "-U",
            "postgres",
            "-d",
            "keycloak",
            "--format=custom",
            "--no-owner",
            "--no-acl",
            "--file=/tmp/restore.dump",
        )
        docker("cp", f"{name}:/tmp/restore.dump", str(tmp_path / "restore.dump"))
        tmp_path.chmod(0o755)
        (tmp_path / "restore.dump").chmod(0o444)
    finally:
        docker("rm", "-f", name, check=False)
        docker("volume", "rm", source_volume, check=False)
    return tmp_path, hashlib.sha256((tmp_path / "restore.dump").read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("username", "database"),
    [("keycloak", "keycloak"), ("Realm_Owner-7", "Identity_Store-7"), ("R" * 63, "D" * 63)],
)
def test_fresh_install_preserves_writes_on_regular_restart(postgres, username, database):
    name = postgres("fresh-install", username=username, database=database)
    ready(name)
    assert_database_identity(name, username, database)
    # Keycloak has not started yet, so this restart must not require its schema.
    assert sql(name, "SELECT to_regclass('public.realm') IS NULL", database) == "t"
    sql(
        name,
        "CREATE TABLE retained(id text); INSERT INTO retained VALUES ('written-after-install')",
        database,
    )
    docker("stop", name)
    second = postgres("fresh-install", username=username, database=database)
    ready(second)
    assert_database_identity(second, username, database)
    assert sql(second, "SELECT id FROM retained", database) == "written-after-install"


@pytest.mark.parametrize(
    ("username", "database"),
    [("keycloak", "keycloak"), ("Realm_Owner-7", "Identity_Store-7")],
)
def test_restore_preserves_records_and_restart_does_not_replay(
    postgres, tmp_path, username, database
):
    # The source archive always comes from a database named keycloak.
    snapshot, digest = archive(tmp_path, valid=True)
    name = postgres("restore", snapshot, digest, username=username, database=database)
    ready(name)
    assert sql(name, "SELECT id FROM user_entity", database) == "retained-user"
    assert sql(name, "SELECT id FROM client", database) == "retained-client"
    assert_database_identity(name, username, database)
    sql(name, "INSERT INTO user_entity VALUES ('post-cutover-user')", database)
    docker("stop", name)
    second = postgres("restore", digest=digest, username=username, database=database)
    ready(second)
    assert_database_identity(second, username, database)
    assert (
        sql(second, "SELECT id FROM user_entity ORDER BY id", database)
        == "post-cutover-user\nretained-user"
    )
    # Model an externally verified cluster that predates the bootstrap receipt.
    docker("exec", second, "rm", "/var/lib/postgresql/data/pgdata/.keycloak-bootstrap-complete")
    docker("stop", second)
    existing = postgres("existing", username=username, database=database, existing_verified=True)
    ready(existing)
    assert_database_identity(existing, username, database)
    assert (
        sql(existing, "SELECT id FROM user_entity ORDER BY id", database)
        == "post-cutover-user\nretained-user"
    )
    assert hashlib.sha256((snapshot / "restore.dump").read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("mode", ["fresh-install", "restore", "existing"])
@pytest.mark.parametrize(
    "changed", ["other-role", "missing-role", "missing-database", "wrong-owner-database"]
)
def test_wrong_identity_refuses_legacy_completed_volume_without_renaming(
    postgres, tmp_path, mode, changed
):
    username, database = "Realm_Owner-7", "Identity_Store-7"
    original_mode = "fresh-install" if mode == "fresh-install" else "restore"
    snapshot, digest = (None, "") if original_mode == "fresh-install" else archive(tmp_path, True)
    name = postgres(original_mode, snapshot, digest, username=username, database=database)
    ready(name)
    sql(
        name,
        'CREATE ROLE "Other_Owner-8" LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE '
        "NOREPLICATION NOBYPASSRLS; "
        "CREATE TABLE retained(id text); INSERT INTO retained VALUES ('keep-original-data')",
        database,
    )
    if changed == "wrong-owner-database":
        sql(name, 'CREATE DATABASE "Other_Store-8" OWNER "Other_Owner-8"', database)
    # Legacy completed volumes have a receipt but no identity marker.
    docker("exec", name, "rm", "/var/lib/postgresql/data/pgdata/.keycloak-bootstrap-identity")
    docker("stop", name)
    selected_user = "Other_Owner-8" if changed == "other-role" else username
    if changed == "missing-role":
        selected_user = username.lower()
    selected_db = database.lower() if changed == "missing-database" else database
    if changed == "wrong-owner-database":
        selected_db = "Other_Store-8"
    refused = postgres(
        mode,
        digest=digest,
        username=selected_user,
        database=selected_db,
        existing_verified=True,
    )
    refused_identity(refused)
    recovered = postgres(original_mode, digest=digest, username=username, database=database)
    ready(recovered)
    assert_database_identity(recovered, username, database)
    assert sql(recovered, "SELECT id FROM retained", database) == "keep-original-data"
    assert (
        sql(
            recovered,
            "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname::text = :'missing_user') "
            "OR EXISTS (SELECT 1 FROM pg_database WHERE datname::text = :'missing_db')",
            database,
            missing_user=username.lower(),
            missing_db=database.lower(),
        )
        == "f"
    )
    if changed == "wrong-owner-database":
        assert (
            sql(
                recovered,
                "SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname='Other_Store-8'",
                database,
            )
            == "Other_Owner-8"
        )


@pytest.mark.parametrize("mode", ["fresh-install", "existing"])
@pytest.mark.parametrize(
    "attributes",
    [
        "NOLOGIN",
        "LOGIN SUPERUSER",
        "LOGIN CREATEDB",
        "LOGIN CREATEROLE",
        "LOGIN REPLICATION",
        "LOGIN BYPASSRLS",
    ],
)
def test_bootstrap_rejects_nonlogin_or_privileged_database_owner(postgres, mode, attributes):
    name = postgres("fresh-install")
    ready(name)
    # Each role owns its selected database, so ownership alone cannot pass this check.
    sql(name, f'CREATE ROLE "Unsafe_Owner-9" {attributes}')
    sql(name, 'CREATE DATABASE "Unsafe_Store-9" OWNER "Unsafe_Owner-9"')
    sql(name, "CREATE TABLE retained(id text); INSERT INTO retained VALUES ('keep')")
    docker("exec", name, "rm", "/var/lib/postgresql/data/pgdata/.keycloak-bootstrap-identity")
    docker("stop", name)
    refused = postgres(
        mode, username="Unsafe_Owner-9", database="Unsafe_Store-9", existing_verified=True
    )
    refused_identity(refused)
    recovered = postgres("fresh-install")
    ready(recovered)
    assert sql(recovered, "SELECT id FROM retained") == "keep"
    assert (
        sql(
            recovered,
            "SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname='Unsafe_Store-9'",
        )
        == "Unsafe_Owner-9"
    )


def test_completed_marker_rejects_another_database_owned_by_the_same_role(postgres):
    username, database = "Realm_Owner-7", "Identity_Store-7"
    name = postgres("fresh-install", username=username, database=database)
    ready(name)
    sql(name, 'CREATE DATABASE "Other_Store-8" OWNER "Realm_Owner-7"', database)
    sql(name, "CREATE TABLE retained(id text); INSERT INTO retained VALUES ('keep')", database)
    docker("stop", name)
    refused = postgres("fresh-install", username=username, database="Other_Store-8")
    assert docker("wait", refused).stdout.strip() != "0"
    assert "completed database has a different identity" in docker("logs", refused).stderr
    recovered = postgres("fresh-install", username=username, database=database)
    ready(recovered)
    assert sql(recovered, "SELECT id FROM retained", database) == "keep"
    assert_database_identity(recovered, username, database)
    assert_database_identity(recovered, username, "Other_Store-8")


def test_incomplete_archive_rolls_back_and_retry_stays_closed(postgres, tmp_path):
    snapshot, digest = archive(tmp_path, valid=False)
    for _ in range(2):
        name = postgres("restore", snapshot, digest)
        result = docker("wait", name)
        assert result.stdout.strip() != "0"
        logs = docker("logs", name)
        assert "Snapshot does not contain initialized Keycloak schema/data" in logs.stderr
        for changed_user, changed_db in [
            ("Changed_Owner-8", "keycloak"),
            ("keycloak", "Changed_Store-8"),
        ]:
            refused = postgres(
                "restore", snapshot, digest, username=changed_user, database=changed_db
            )
            assert docker("wait", refused).stdout.strip() != "0"
            assert "staging database has a different identity" in docker("logs", refused).stderr
        volume = docker(
            "inspect",
            "-f",
            '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}'
            "{{.Name}}{{end}}{{end}}",
            name,
        ).stdout.strip()
        inspector = "kc-inspect-test-" + uuid.uuid4().hex
        try:
            docker(
                "run",
                "-d",
                "--name",
                inspector,
                "--network",
                "none",
                "--user",
                "postgres",
                "--entrypoint",
                "postgres",
                "-v",
                f"{volume}:/var/lib/postgresql/data",
                IMAGE,
                "-D",
                "/var/lib/postgresql/data/pgdata.bootstrap",
                "-c",
                "listen_addresses=",
            )
            for _ in range(120):
                if docker("exec", inspector, "pg_isready", check=False).returncode == 0:
                    break
                time.sleep(0.25)
            assert (
                sql(
                    inspector,
                    "SELECT to_regclass('public.realm') IS NULL "
                    "AND to_regclass('_mcp_bootstrap.receipt') IS NULL",
                )
                == "t"
            )
            assert (
                sql(
                    inspector,
                    "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='Changed_Owner-8') "
                    "OR EXISTS (SELECT 1 FROM pg_database WHERE datname='Changed_Store-8')",
                )
                == "f"
            )
        finally:
            docker("rm", "-f", inspector, check=False)
    assert hashlib.sha256((snapshot / "restore.dump").read_bytes()).hexdigest() == digest
