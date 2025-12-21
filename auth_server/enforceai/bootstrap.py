from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
import uuid
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from auth_server.enforceai.crypto.keyring import (
    GatewayKeyring,
)
from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from auth_server.enforceai.tokens.mint import (
    mint_gateway_token,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_ACTIVE_KID: str = "kid-prod-1"
DEFAULT_GATEWAY_ISSUER: str = "enforceai-gateway"
DEFAULT_BOOTSTRAP_SCOPES: tuple[str, ...] = ("registry-admins",)


def _ensure_dir(
    *,
    path: Path,
    mode: int,
) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
    try:
        path.chmod(mode)
    except OSError:
        logger.debug("Unable to chmod directory %s", path)


def _write_file_if_missing(
    *,
    path: Path,
    contents: bytes,
    mode: int,
    force: bool,
) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_bytes(contents)
    try:
        path.chmod(mode)
    except OSError:
        logger.debug("Unable to chmod file %s", path)


def _generate_gateway_keys_if_missing(
    *,
    private_key_path: Path,
    public_keys_dir: Path,
    active_kid: str,
    force: bool,
) -> None:
    public_key_path = public_keys_dir / f"{active_kid}.pem"

    if (
        private_key_path.exists()
        and public_key_path.exists()
        and private_key_path.is_file()
        and public_key_path.is_file()
        and not force
    ):
        return

    _ensure_dir(
        path=public_keys_dir,
        mode=0o700,
    )

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    _write_file_if_missing(
        path=private_key_path,
        contents=private_pem,
        mode=0o600,
        force=force,
    )
    _write_file_if_missing(
        path=public_key_path,
        contents=public_pem,
        mode=0o644,
        force=force,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap EnforceAI state for ECS/Fargate deployments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage (run inside auth-server container):
  source .venv/bin/activate
  python -m auth_server.enforceai.bootstrap \
    --state-dir /app/enforceai_state \
    --bootstrap-user-id "local|admin"
""",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default=os.getenv("ENFORCEAI_STATE_DIR"),
        help="State directory for EnforceAI DB + secrets (or ENFORCEAI_STATE_DIR).",
    )
    parser.add_argument(
        "--active-kid",
        type=str,
        default=DEFAULT_ACTIVE_KID,
        help=f"Active gateway key id (default: {DEFAULT_ACTIVE_KID}).",
    )
    parser.add_argument(
        "--gateway-issuer",
        type=str,
        default=DEFAULT_GATEWAY_ISSUER,
        help=f"Gateway token issuer to mint/accept (default: {DEFAULT_GATEWAY_ISSUER}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated secrets/keys (dangerous).",
    )
    parser.add_argument(
        "--bootstrap-user-id",
        type=str,
        default=None,
        help="If set, create an initial agent for this user_id and mint a bootstrap gateway token.",
    )
    parser.add_argument(
        "--bootstrap-agent-id",
        type=str,
        default=None,
        help="Optional agent_id to create (defaults to a new uuid4).",
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Bootstrap agent scope (repeatable). Defaults to registry-admins.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print a JSON summary to stdout (includes token if minted).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.state_dir:
        raise ValueError("state-dir must be provided via --state-dir or ENFORCEAI_STATE_DIR")

    state_dir = Path(args.state_dir)
    secrets_dir = state_dir / "secrets"

    db_path = state_dir / "enforceai.db"
    api_key_pepper_path = secrets_dir / "api_key_pepper"
    upstream_kek_path = secrets_dir / "upstream_kek"

    gateway_private_key_path = secrets_dir / "gateway_private.pem"
    gateway_public_keys_dir = secrets_dir / "gateway_public_keys"

    active_kid = str(args.active_kid).strip()
    gateway_issuer = str(args.gateway_issuer).strip()
    if not active_kid:
        raise ValueError("--active-kid must be a non-empty string")
    if not gateway_issuer:
        raise ValueError("--gateway-issuer must be a non-empty string")

    _ensure_dir(
        path=state_dir,
        mode=0o700,
    )
    _ensure_dir(
        path=secrets_dir,
        mode=0o700,
    )

    _write_file_if_missing(
        path=api_key_pepper_path,
        contents=f"{secrets.token_hex(32)}\n".encode("utf-8"),
        mode=0o600,
        force=args.force,
    )
    _write_file_if_missing(
        path=upstream_kek_path,
        contents=f"{secrets.token_hex(32)}\n".encode("utf-8"),
        mode=0o600,
        force=args.force,
    )
    _generate_gateway_keys_if_missing(
        private_key_path=gateway_private_key_path,
        public_keys_dir=gateway_public_keys_dir,
        active_kid=active_kid,
        force=args.force,
    )

    data_layer = EnforceAIDataLayer(
        db_path=db_path,
    )
    data_layer.initialize()
    stores = data_layer.build_stores()

    summary: dict[str, object] = {
        "state_dir": str(state_dir),
        "db_path": str(db_path),
        "gateway_private_key_path": str(gateway_private_key_path),
        "gateway_public_keys_dir": str(gateway_public_keys_dir),
        "gateway_active_kid": active_kid,
        "gateway_issuer": gateway_issuer,
        "api_key_pepper_path": str(api_key_pepper_path),
        "upstream_kek_path": str(upstream_kek_path),
        "bootstrap": None,
    }

    if args.bootstrap_user_id is not None:
        bootstrap_user_id = str(args.bootstrap_user_id).strip()
        if not bootstrap_user_id:
            raise ValueError("--bootstrap-user-id must be a non-empty string when provided")

        bootstrap_agent_id = args.bootstrap_agent_id
        if bootstrap_agent_id is None:
            bootstrap_agent_id = str(uuid.uuid4())
        bootstrap_agent_id = bootstrap_agent_id.strip()
        if not bootstrap_agent_id:
            raise ValueError("--bootstrap-agent-id must be a non-empty string when provided")

        scopes = [scope.strip() for scope in args.scope if scope.strip()]
        if not scopes:
            scopes = list(DEFAULT_BOOTSTRAP_SCOPES)

        existing = stores.agent_store.get_agent_by_id(
            agent_id=bootstrap_agent_id,
        )
        if existing is None:
            stores.agent_store.create_agent(
                user_id=bootstrap_user_id,
                agent_id=bootstrap_agent_id,
                scopes=scopes,
                allowed_tools=None,
                alias="bootstrap",
                metadata={"bootstrap": True},
            )

        keyring = GatewayKeyring.load(
            private_key_path=gateway_private_key_path,
            public_keys_dir=gateway_public_keys_dir,
            active_kid=active_kid,
        )
        token = mint_gateway_token(
            keyring=keyring,
            issuer=gateway_issuer,
            user_id=bootstrap_user_id,
            agent_id=bootstrap_agent_id,
            scopes=scopes,
            ttl_seconds=60 * 60,
        )

        bootstrap_token_path = state_dir / "bootstrap_gateway_token.txt"
        _write_file_if_missing(
            path=bootstrap_token_path,
            contents=f"{token}\n".encode("utf-8"),
            mode=0o600,
            force=args.force,
        )

        summary["bootstrap"] = {
            "user_id": bootstrap_user_id,
            "agent_id": bootstrap_agent_id,
            "scopes": scopes,
            "token_path": str(bootstrap_token_path),
            "token": token,
        }

    if args.print_json:
        print(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    else:
        logger.info(
            "EnforceAI bootstrap complete:\n%s",
            json.dumps(summary, indent=2, sort_keys=True, default=str),
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("EnforceAI bootstrap failed")
        sys.exit(2)

