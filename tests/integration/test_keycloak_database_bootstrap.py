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

    def start(mode: str, snapshot: Path | None = None, digest: str = "") -> str:
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
            "KEYCLOAK_DB_USER=keycloak",
            "-e",
            "KEYCLOAK_DB_NAME=keycloak",
            "-e",
            "KEYCLOAK_DB_PASSWORD=test-only-role-password",
            "-e",
            f"BOOTSTRAP_MODE={mode}",
            "-e",
            f"BOOTSTRAP_SHA256={digest}",
            "-e",
            "BOOTSTRAP_SOURCE_QUIESCED=true",
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


def sql(name: str, query: str):
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
        "keycloak",
        "-At",
        input=query,
    ).stdout.strip()


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


def test_fresh_install_preserves_writes_on_regular_restart(postgres):
    name = postgres("fresh-install")
    ready(name)
    sql(
        name,
        "CREATE TABLE retained(id text); INSERT INTO retained VALUES ('written-after-install')",
    )
    docker("stop", name)
    second = postgres("fresh-install")
    ready(second)
    assert sql(second, "SELECT id FROM retained") == "written-after-install"


def test_restore_preserves_records_and_restart_does_not_replay(postgres, tmp_path):
    snapshot, digest = archive(tmp_path, valid=True)
    name = postgres("restore", snapshot, digest)
    ready(name)
    assert sql(name, "SELECT id FROM user_entity") == "retained-user"
    assert sql(name, "SELECT id FROM client") == "retained-client"
    assert (
        sql(
            name,
            "SELECT rolsuper::text || ':' || rolcreatedb::text || ':' || rolcreaterole::text "
            "FROM pg_roles WHERE rolname='keycloak'",
        )
        == "false:false:false"
    )
    sql(name, "INSERT INTO user_entity VALUES ('post-cutover-user')")
    docker("stop", name)
    second = postgres("restore", snapshot, digest)
    ready(second)
    assert (
        sql(second, "SELECT id FROM user_entity ORDER BY id") == "post-cutover-user\nretained-user"
    )
    assert hashlib.sha256((snapshot / "restore.dump").read_bytes()).hexdigest() == digest


def test_incomplete_archive_rolls_back_and_retry_stays_closed(postgres, tmp_path):
    snapshot, digest = archive(tmp_path, valid=False)
    for _ in range(2):
        name = postgres("restore", snapshot, digest)
        result = docker("wait", name)
        assert result.stdout.strip() != "0"
        logs = docker("logs", name)
        assert "Snapshot does not contain initialized Keycloak schema/data" in logs.stderr
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
        finally:
            docker("rm", "-f", inspector, check=False)
    assert hashlib.sha256((snapshot / "restore.dump").read_bytes()).hexdigest() == digest
