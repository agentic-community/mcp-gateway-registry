"""
Lambda: upsert UI-scope group docs into DocumentDB.

Bridges a parity gap with the upstream registry image (which only seeds
`registry-admins`). Invoked directly via SDK by an `AwsCustomResource` on
every deploy of Registry-Service. Idempotent (upsert).

Env:
  DOCUMENTDB_HOST, DOCUMENTDB_PORT, DOCUMENTDB_DATABASE, DOCUMENTDB_NAMESPACE
  DOCUMENTDB_SECRET_ARN  (Secrets Manager ARN with username/password keys)
  SCOPES_YML             (full YAML content, inlined at synth time)
"""
import json
import logging
import os

import boto3
import yaml
from pymongo import MongoClient

log = logging.getLogger()
log.setLevel(logging.INFO)


def _connect():
    secret = boto3.client("secretsmanager").get_secret_value(SecretId=os.environ["DOCUMENTDB_SECRET_ARN"])
    creds = json.loads(secret["SecretString"])
    uri = (
        f"mongodb://{creds['username']}:{creds['password']}@"
        f"{os.environ['DOCUMENTDB_HOST']}:{os.environ.get('DOCUMENTDB_PORT', '27017')}/"
        f"?authMechanism=SCRAM-SHA-1&authSource=admin&tls=true"
        f"&tlsCAFile=/var/task/global-bundle.pem&directConnection=true&retryWrites=false"
    )
    # No ServerApi — DocumentDB doesn't support pymongo 4.x stable-API framing.
    return MongoClient(uri)[os.environ["DOCUMENTDB_DATABASE"]]


def handler(event, _context):
    log.info("event: %s", json.dumps(event))

    # Debug mode: dump what's currently in the collection without changing anything.
    if event.get("debug"):
        namespace = os.environ.get("DOCUMENTDB_NAMESPACE", "default")
        coll = _connect()[f"mcp_scopes_{namespace}"]
        return {"docs": [{"_id": d["_id"], "group_mappings": d.get("group_mappings", []),
                          "ui_permissions_keys": list(d.get("ui_permissions", {}).keys())}
                         for d in coll.find({})]}

    # Mirrors Terraform's `scopes-init` busybox task: ship scopes.yml to EFS
    # so the auth-server reads the canonical file. Failure aborts the deploy.
    src = "/var/task/scopes.yml"
    with open(src, "rb") as fin, open("/mnt/auth_config/scopes.yml", "wb") as fout:
        fout.write(fin.read())
    log.info("scopes.yml copied to EFS")

    with open(src) as f:
        data = yaml.safe_load(f)
    ui = data.get("UI-Scopes", {}) or {}
    group_mappings = data.get("group_mappings", {}) or {}

    # Collection naming mirrors registry/repositories/documentdb/client.py
    # get_collection_name(): always "mcp_scopes_<namespace>" (no special case).
    namespace = os.environ.get("DOCUMENTDB_NAMESPACE", "default")
    coll = _connect()[f"mcp_scopes_{namespace}"]

    # For each UI-Scope name, find which IdP groups map to it (reverse of YAML
    # group_mappings: group -> [scopes]). Upsert one doc per UI-Scope so the
    # auth-server can resolve groups -> scopes via find({group_mappings: G}).
    upserted = 0
    for name, perms in ui.items():
        groups = [g for g, scopes in group_mappings.items() if name in scopes]
        # _id is immutable on update — DocumentDB rejects $set with _id present.
        result = coll.update_one(
            {"_id": name},
            {"$set": {"group_mappings": groups, "server_access": [], "ui_permissions": perms}},
            upsert=True,
        )
        if result.upserted_id or result.modified_count > 0:
            upserted += 1

    log.info("upserted=%d", upserted)
    return {"upserted": upserted}
