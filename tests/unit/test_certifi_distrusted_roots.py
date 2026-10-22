"""
Unit tests that keep the eMudhra-chain roots out of our Python trust store.

On 2026-10-20 the root programs distrusted "emSign ECC Root CA - G3" and
"AAA Certificate Services" after eMudhra intermediates were found with
recoverable private keys (https://emudhra.we-have-a-problem.com/). httpx reads
its roots from certifi, not the OS store, so every service's outbound TLS
trusts whatever certifi bundle its lockfile pins. certifi 2026.10.19 is the
first release without these roots. These tests fail if a later lock change
brings an older certifi, or a bundle carrying either root, back in.
"""

import hashlib
import re
import ssl
from pathlib import Path

import certifi
import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

MIN_CERTIFI_VERSION: tuple[int, int, int] = (2026, 10, 19)

# SHA-256 fingerprints of the DER-encoded roots distrusted on 2026-10-20.
DISTRUSTED_ROOT_FINGERPRINTS: dict[str, str] = {
    "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b": "emSign ECC Root CA - G3",
    "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4": "AAA Certificate Services",
}

# Every uv project whose lockfile ends up in a container image.
LOCKFILES: list[str] = [
    "uv.lock",
    "auth_server/uv.lock",
    "metrics-service/uv.lock",
    "servers/mcpgw/uv.lock",
    "servers/currenttime/uv.lock",
    "servers/realserverfaketools/uv.lock",
    "agents/a2a/uv.lock",
]

PEM_PATTERN: re.Pattern = re.compile(
    r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
    re.S,
)


def _bundle_fingerprints(
    bundle_path: str,
) -> set[str]:
    """Return the SHA-256 fingerprint of every certificate in a PEM bundle."""
    text = Path(bundle_path).read_text()
    fingerprints = set()
    for pem in PEM_PATTERN.findall(text):
        der = ssl.PEM_cert_to_DER_cert(pem)
        fingerprints.add(hashlib.sha256(der).hexdigest())
    return fingerprints


def _locked_certifi_version(
    lockfile: Path,
) -> tuple[int, int, int] | None:
    """Return the certifi version pinned in a uv.lock, or None if it has none."""
    match = re.search(
        r'^name = "certifi"\nversion = "(\d+)\.(\d+)\.(\d+)"',
        lockfile.read_text(),
        re.M,
    )
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


class TestCertifiDistrustedRoots:
    """The certifi bundle the services use excludes the distrusted roots."""

    def test_installed_bundle_excludes_distrusted_roots(self):
        """The installed certifi bundle carries neither distrusted root."""
        fingerprints = _bundle_fingerprints(certifi.where())

        present = [
            name
            for fingerprint, name in DISTRUSTED_ROOT_FINGERPRINTS.items()
            if fingerprint in fingerprints
        ]

        assert present == []

    @pytest.mark.parametrize("lockfile", LOCKFILES)
    def test_lockfile_pins_fixed_certifi(
        self,
        lockfile: str,
    ):
        """Each lockfile that pins certifi pins 2026.10.19 or later."""
        version = _locked_certifi_version(REPO_ROOT / lockfile)

        if version is None:
            pytest.skip(f"{lockfile} does not depend on certifi")
        assert version >= MIN_CERTIFI_VERSION
