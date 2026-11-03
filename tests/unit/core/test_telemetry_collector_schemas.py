"""Unit tests for the telemetry collector's cloud-detection consistency rules.

The collector Lambda drops any event that fails validation, so a registry
that sends a (cloud, cloud_detection_method) pair the collector rejects
disappears from telemetry. These tests pin the pairs from issue #1911.

schemas.py is loaded by file path: its directory also holds the Lambda's
vendored dependencies, which must not shadow the repo's own packages.
"""

import importlib.util
import re
from pathlib import Path

import pytest

SCHEMAS_PATH: Path = (
    Path(__file__).resolve().parents[3]
    / "terraform"
    / "telemetry-collector"
    / "lambda"
    / "collector"
    / "schemas.py"
)


def _load_schemas():
    """Import the collector's schemas.py without adding its folder to sys.path."""
    spec = importlib.util.spec_from_file_location("collector_schemas", SCHEMAS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schemas = _load_schemas()


class TestCloudDetectionConsistency:
    """Pairs the registry sends must validate; inconsistent pairs must not."""

    @pytest.mark.parametrize(
        ("cloud", "method"),
        [
            ("aws", "operator_declared"),
            ("azure", "operator_declared"),
            ("gcp", "operator_declared"),
            ("on_premises", "operator_declared"),
            ("other", "operator_declared"),
            ("unknown", "operator_declined"),
            ("local", "host_os"),
            ("local", "wsl"),
            ("local", "docker_desktop"),
            ("aws", "k8s_heuristic"),
            ("aws", "env"),
            ("unknown", "unknown"),
        ],
    )
    def test_accepts_consistent_pair(self, cloud, method):
        schemas._check_cloud_detection_consistency(cloud, method)

    @pytest.mark.parametrize(
        ("cloud", "method"),
        [
            ("unknown", "operator_declared"),
            ("local", "env"),
            ("local", "unknown"),
            ("local", "operator_declared"),
            ("aws", "wsl"),
            ("unknown", "docker_desktop"),
            ("aws", "operator_declined"),
        ],
    )
    def test_rejects_inconsistent_pair(self, cloud, method):
        with pytest.raises(ValueError):
            schemas._check_cloud_detection_consistency(cloud, method)

    @pytest.mark.parametrize("method", ["host_os", "wsl", "docker_desktop"])
    def test_local_methods_in_method_allowlist(self, method):
        assert re.match(schemas._CLOUD_DETECTION_METHOD_PATTERN, method)
