"""Unit tests for the cloud-detection cascade in registry/core/telemetry.py.

Covers issue #986 changes:
- Sequential cascade: env -> DMI -> ECS metadata -> k8s heuristic -> IMDS -> unknown
- Per-process lru_cache
- IMDS opt-out via telemetry_imds_probe_disabled
- IMDS opt-out via MCP_TELEMETRY_DISABLED
- IMDS probe latency bound (< 1000ms worst case)
- httpx.Client trust_env=False (no corporate-proxy interception)
- Prometheus counter increment

Covers issue #1911 changes:
- EC2 instance-ID node names (EKS Auto Mode, Karpenter)
- Developer-machine label (cloud=local)
- AWS/Azure/GCP operator hints honored by _resolve_cloud
- WARNING when detection ends at unknown
"""

import io
import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from registry.core.telemetry import (
    _DETECTION_METHOD_DMI,
    _DETECTION_METHOD_DOCKER_DESKTOP,
    _DETECTION_METHOD_ECS_META,
    _DETECTION_METHOD_ENV,
    _DETECTION_METHOD_EXPLICIT,
    _DETECTION_METHOD_HOST_OS,
    _DETECTION_METHOD_IMDS,
    _DETECTION_METHOD_K8S_HEURISTIC,
    _DETECTION_METHOD_OPERATOR_DECLARED,
    _DETECTION_METHOD_OPERATOR_DECLINED,
    _DETECTION_METHOD_UNKNOWN,
    _DETECTION_METHOD_WSL,
    _IMDS_PROBE_TIMEOUT_SECONDS,
    _KERNEL_RELEASE_PATH,
    _detect_cloud_from_env,
    _detect_cloud_from_k8s_heuristic,
    _detect_cloud_provider_with_method,
    _detect_local_environment,
    _probe_imds,
    _resolve_cloud,
    _should_probe_imds,
)

_TELEMETRY_LOGGER = "registry.core.telemetry"


@pytest.fixture(autouse=True)
def _clear_detection_cache():
    """Ensure every test starts with a fresh lru_cache state."""
    _detect_cloud_provider_with_method.cache_clear()
    yield
    _detect_cloud_provider_with_method.cache_clear()


@pytest.fixture
def _clean_env(monkeypatch):
    """Clear all cloud-detection-relevant env vars."""
    for var in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "WEBSITE_INSTANCE_ID",
        "AZURE_CLIENT_ID",
        "ECS_CONTAINER_METADATA_URI",
        "ECS_CONTAINER_METADATA_URI_V4",
        "KUBERNETES_SERVICE_HOST",
        "NODE_NAME",
        "MCP_TELEMETRY_DISABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    # Pin the host OS so the local-machine tier behaves the same on a
    # developer Mac as on the Linux CI runner.
    monkeypatch.setattr("platform.system", lambda: "Linux")


def _patch_dmi_empty():
    """Return a patch context that makes DMI reads fail."""
    return patch(
        "builtins.open",
        side_effect=FileNotFoundError("no dmi"),
    )


def _patch_kernel_release(
    release: str,
):
    """Return a patch context where only the kernel release file is readable."""

    def _fake_open(path, *args, **kwargs):
        if path == _KERNEL_RELEASE_PATH:
            return io.StringIO(release)
        raise FileNotFoundError(path)

    return patch("builtins.open", side_effect=_fake_open)


def _patch_imds_off():
    """Return a patch context that skips the IMDS tier."""
    return patch(
        "registry.core.telemetry._should_probe_imds",
        return_value=False,
    )


class TestDetectionCascade:
    """Verify each tier of the detection cascade fires in order."""

    def test_env_aws_region_short_circuits(self, _clean_env, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        cloud, method = _detect_cloud_provider_with_method()
        assert cloud == "aws"
        assert method == _DETECTION_METHOD_ENV

    def test_env_gcp_project_short_circuits(self, _clean_env, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
        cloud, method = _detect_cloud_provider_with_method()
        assert cloud == "gcp"
        assert method == _DETECTION_METHOD_ENV

    def test_env_azure_website_instance(self, _clean_env, monkeypatch):
        monkeypatch.setenv("WEBSITE_INSTANCE_ID", "instance-42")
        cloud, method = _detect_cloud_provider_with_method()
        assert cloud == "azure"
        assert method == _DETECTION_METHOD_ENV

    def test_dmi_aws_board_asset_tag(self, _clean_env):
        """DMI read succeeds for AWS instance-id prefix."""
        fake_file = MagicMock()
        fake_file.__enter__ = MagicMock(return_value=fake_file)
        fake_file.__exit__ = MagicMock(return_value=False)
        fake_file.read = MagicMock(return_value="i-0123456789abcdef0\n")

        def open_side_effect(path, *args, **kwargs):
            if str(path).endswith("board_asset_tag"):
                return fake_file
            raise FileNotFoundError(path)

        with patch("builtins.open", side_effect=open_side_effect):
            cloud, method = _detect_cloud_provider_with_method()
        assert cloud == "aws"
        assert method == _DETECTION_METHOD_DMI

    def test_ecs_metadata_uri_classifies_as_aws(self, _clean_env, monkeypatch):
        monkeypatch.setenv(
            "ECS_CONTAINER_METADATA_URI_V4",
            "http://169.254.170.2/v4/some-id",
        )
        with _patch_dmi_empty():
            cloud, method = _detect_cloud_provider_with_method()
        assert cloud == "aws"
        assert method == _DETECTION_METHOD_ECS_META

    def test_k8s_heuristic_aws_compute_internal(self, _clean_env, monkeypatch):
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        monkeypatch.setenv("NODE_NAME", "ip-10-0-1-23.us-east-1.compute.internal")
        with _patch_dmi_empty():
            cloud, method = _detect_cloud_provider_with_method()
        assert cloud == "aws"
        assert method == _DETECTION_METHOD_K8S_HEURISTIC

    def test_k8s_heuristic_gke_prefix(self, _clean_env, monkeypatch):
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        monkeypatch.setenv("NODE_NAME", "gke-mycluster-default-pool-abc123-def0")
        with _patch_dmi_empty():
            cloud, method = _detect_cloud_provider_with_method()
        assert cloud == "gcp"
        assert method == _DETECTION_METHOD_K8S_HEURISTIC

    def test_k8s_heuristic_aks_prefix(self, _clean_env, monkeypatch):
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        monkeypatch.setenv("NODE_NAME", "aks-agentpool-12345678-vmss000000")
        with _patch_dmi_empty():
            cloud, method = _detect_cloud_provider_with_method()
        assert cloud == "azure"
        assert method == _DETECTION_METHOD_K8S_HEURISTIC

    def test_k8s_heuristic_requires_kubernetes_env_var(self, _clean_env, monkeypatch):
        """Heuristic must not fire when KUBERNETES_SERVICE_HOST is unset."""
        monkeypatch.setenv("NODE_NAME", "gke-foo")
        assert _detect_cloud_from_k8s_heuristic() is None

    def test_k8s_heuristic_unknown_node_name_returns_none(self, _clean_env, monkeypatch):
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        monkeypatch.setenv("NODE_NAME", "weirdly-named-node")
        assert _detect_cloud_from_k8s_heuristic() is None

    def test_all_fail_returns_unknown(self, _clean_env):
        """No env, no DMI, no ECS, no k8s, no IMDS -> unknown."""
        with (
            _patch_dmi_empty(),
            patch(
                "registry.core.telemetry._should_probe_imds",
                return_value=False,
            ),
        ):
            cloud, method = _detect_cloud_provider_with_method()
        assert cloud == "unknown"
        assert method == _DETECTION_METHOD_UNKNOWN


class TestImdsProbe:
    """Cover the IMDS-probe tier in isolation (no real network calls)."""

    def test_imds_aws_hit(self, _clean_env):
        """Simulated AWS IMDSv2 success returns cloud=aws."""
        mock_resp = MagicMock(status_code=200)
        mock_client = MagicMock()
        mock_client.put.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            _patch_dmi_empty(),
            patch(
                "registry.core.telemetry.httpx.Client",
                return_value=mock_client,
            ),
            patch(
                "registry.core.telemetry._should_probe_imds",
                return_value=True,
            ),
        ):
            cloud, method = _detect_cloud_provider_with_method()

        assert cloud == "aws"
        assert method == _DETECTION_METHOD_IMDS
        # AWS probe is a PUT with a token-TTL header
        mock_client.put.assert_called_once()
        put_args, put_kwargs = mock_client.put.call_args
        assert put_args[0].startswith("http://169.254.169.254/latest/api/token")
        assert "X-aws-ec2-metadata-token-ttl-seconds" in put_kwargs["headers"]

    def test_imds_all_timeout_returns_unknown(self, _clean_env):
        """All three probes raising timeouts -> cloud=unknown."""
        mock_client = MagicMock()
        mock_client.put.side_effect = httpx.ReadTimeout("timeout")
        mock_client.get.side_effect = httpx.ReadTimeout("timeout")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            _patch_dmi_empty(),
            patch(
                "registry.core.telemetry.httpx.Client",
                return_value=mock_client,
            ),
            patch(
                "registry.core.telemetry._should_probe_imds",
                return_value=True,
            ),
        ):
            cloud, method = _detect_cloud_provider_with_method()
        assert cloud == "unknown"
        assert method == _DETECTION_METHOD_UNKNOWN

    def test_imds_trust_env_false(self, _clean_env):
        """httpx.Client must be built with trust_env=False to bypass proxies."""
        mock_client = MagicMock()
        mock_client.put.side_effect = httpx.ConnectError("no route")
        mock_client.get.side_effect = httpx.ConnectError("no route")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            "registry.core.telemetry.httpx.Client",
            return_value=mock_client,
        ) as mock_client_cls:
            _probe_imds()

        _, kwargs = mock_client_cls.call_args
        assert kwargs.get("trust_env") is False
        assert kwargs.get("timeout") == _IMDS_PROBE_TIMEOUT_SECONDS


class TestOptOut:
    """Verify probe opt-outs are honored."""

    def test_should_probe_imds_true_when_enabled(self, _clean_env):
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = True
            mock_settings.telemetry_imds_probe_disabled = False
            assert _should_probe_imds() is True

    def test_should_probe_imds_false_when_imds_disabled(self, _clean_env):
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = True
            mock_settings.telemetry_imds_probe_disabled = True
            assert _should_probe_imds() is False

    def test_should_probe_imds_false_when_telemetry_disabled(self, _clean_env):
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = False
            mock_settings.telemetry_imds_probe_disabled = False
            assert _should_probe_imds() is False

    def test_should_probe_imds_false_when_env_var_set(self, _clean_env, monkeypatch):
        monkeypatch.setenv("MCP_TELEMETRY_DISABLED", "1")
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = True
            mock_settings.telemetry_imds_probe_disabled = False
            assert _should_probe_imds() is False


class TestCaching:
    """Detection must only run once per process."""

    def test_result_cached_across_calls(self, _clean_env, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        first = _detect_cloud_provider_with_method()
        # If caching works, changing the env after the first call must not
        # change what the next call returns.
        monkeypatch.delenv("AWS_REGION")
        second = _detect_cloud_provider_with_method()
        assert first == second == ("aws", _DETECTION_METHOD_ENV)


class TestCounterIncrements:
    """The Prometheus counter must be incremented once per detection."""

    def test_counter_incremented_with_correct_labels(self, _clean_env, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        with patch("registry.core.metrics.CLOUD_DETECTION_TOTAL") as mock_counter:
            _detect_cloud_provider_with_method()
        mock_counter.labels.assert_called_once_with(
            cloud="aws",
            method=_DETECTION_METHOD_ENV,
        )
        mock_counter.labels.return_value.inc.assert_called_once()


class TestEnvHelper:
    """Targeted tests for _detect_cloud_from_env precedence."""

    def test_aws_default_region_works(self, _clean_env, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        assert _detect_cloud_from_env() == "aws"

    def test_gcloud_project_works(self, _clean_env, monkeypatch):
        monkeypatch.setenv("GCLOUD_PROJECT", "my-proj")
        assert _detect_cloud_from_env() == "gcp"

    def test_azure_client_id_works(self, _clean_env, monkeypatch):
        monkeypatch.setenv("AZURE_CLIENT_ID", "00000000-0000-0000-0000-000000000000")
        assert _detect_cloud_from_env() == "azure"

    def test_no_signal_returns_none(self, _clean_env):
        assert _detect_cloud_from_env() is None


class TestInstanceIdNodeName:
    """EKS Auto Mode and Karpenter name nodes after the EC2 instance ID (#1911)."""

    @pytest.mark.parametrize(
        "node_name",
        [
            "i-0123456789abcdef0",
            "i-0123abcd",
            "I-0123456789ABCDEF0",
        ],
    )
    def test_instance_id_node_name_is_aws(self, _clean_env, monkeypatch, node_name):
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        monkeypatch.setenv("NODE_NAME", node_name)
        with _patch_dmi_empty():
            cloud, method = _detect_cloud_provider_with_method()
        assert (cloud, method) == ("aws", _DETECTION_METHOD_K8S_HEURISTIC)

    @pytest.mark.parametrize(
        "node_name",
        [
            "ip-10-0-1-5",
            "i-xyz",
            "i-0123456789abcdef0-extra",
            "i-0123456789abcdef",
        ],
    )
    def test_non_instance_id_node_name_is_not_aws(self, _clean_env, monkeypatch, node_name):
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        monkeypatch.setenv("NODE_NAME", node_name)
        assert _detect_cloud_from_k8s_heuristic() is None


class TestLocalEnvironment:
    """Developer machines report cloud=local instead of unknown (#1911)."""

    @pytest.mark.parametrize("system", ["Darwin", "Windows"])
    def test_desktop_host_os_is_local(self, _clean_env, monkeypatch, system):
        monkeypatch.setattr("platform.system", lambda: system)
        with _patch_dmi_empty(), _patch_imds_off():
            cloud, method = _detect_cloud_provider_with_method()
        assert (cloud, method) == ("local", _DETECTION_METHOD_HOST_OS)

    def test_wsl_kernel_is_local(self, _clean_env):
        with _patch_kernel_release("5.15.153.1-microsoft-standard-WSL2\n"), _patch_imds_off():
            cloud, method = _detect_cloud_provider_with_method()
        assert (cloud, method) == ("local", _DETECTION_METHOD_WSL)

    def test_docker_desktop_kernel_is_local(self, _clean_env):
        with _patch_kernel_release("6.10.14-linuxkit\n"), _patch_imds_off():
            cloud, method = _detect_cloud_provider_with_method()
        assert (cloud, method) == ("local", _DETECTION_METHOD_DOCKER_DESKTOP)

    def test_plain_linux_kernel_is_not_local(self, _clean_env):
        with _patch_kernel_release("6.8.0-1015-aws\n"):
            assert _detect_local_environment() is None

    def test_missing_kernel_release_is_not_local(self, _clean_env):
        with _patch_dmi_empty():
            assert _detect_local_environment() is None

    def test_env_signal_beats_local(self, _clean_env, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        assert _detect_cloud_provider_with_method() == ("aws", _DETECTION_METHOD_ENV)

    def test_imds_signal_beats_local(self, _clean_env, monkeypatch):
        """An EC2 Mac instance answers IMDS, so it stays aws."""
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        with (
            _patch_dmi_empty(),
            patch("registry.core.telemetry._should_probe_imds", return_value=True),
            patch("registry.core.telemetry._probe_imds", return_value="aws"),
        ):
            cloud, method = _detect_cloud_provider_with_method()
        assert (cloud, method) == ("aws", _DETECTION_METHOD_IMDS)


class TestUnknownWarning:
    """Detection that ends at unknown tells the operator how to fix it (#1911)."""

    def _warnings(self, caplog) -> list[logging.LogRecord]:
        return [
            r
            for r in caplog.records
            if r.name == _TELEMETRY_LOGGER
            and r.levelno == logging.WARNING
            and "MCP_CLOUD_PROVIDER" in r.getMessage()
        ]

    def test_warning_logged_once_per_process(self, _clean_env, caplog):
        caplog.set_level(logging.WARNING, logger=_TELEMETRY_LOGGER)
        with (
            _patch_dmi_empty(),
            _patch_imds_off(),
            patch("registry.core.telemetry.settings") as mock_settings,
        ):
            mock_settings.telemetry_enabled = True
            _detect_cloud_provider_with_method()
            _detect_cloud_provider_with_method()
        assert len(self._warnings(caplog)) == 1

    def test_no_warning_when_telemetry_disabled(self, _clean_env, caplog):
        caplog.set_level(logging.WARNING, logger=_TELEMETRY_LOGGER)
        with (
            _patch_dmi_empty(),
            _patch_imds_off(),
            patch("registry.core.telemetry.settings") as mock_settings,
        ):
            mock_settings.telemetry_enabled = False
            _detect_cloud_provider_with_method()
        assert self._warnings(caplog) == []

    def test_no_warning_when_cloud_detected(self, _clean_env, monkeypatch, caplog):
        caplog.set_level(logging.WARNING, logger=_TELEMETRY_LOGGER)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        _detect_cloud_provider_with_method()
        assert self._warnings(caplog) == []


class TestResolveCloudOperatorHint:
    """The admin-banner answer overrides auto-detection (#1120, fixed in #1911)."""

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            ("aws", ("aws", _DETECTION_METHOD_OPERATOR_DECLARED)),
            ("azure", ("azure", _DETECTION_METHOD_OPERATOR_DECLARED)),
            ("gcp", ("gcp", _DETECTION_METHOD_OPERATOR_DECLARED)),
            ("on_premises", ("on_premises", _DETECTION_METHOD_OPERATOR_DECLARED)),
            ("other", ("other", _DETECTION_METHOD_OPERATOR_DECLARED)),
            ("declined", ("unknown", _DETECTION_METHOD_OPERATOR_DECLINED)),
        ],
    )
    async def test_each_hint_resolves(self, _clean_env, hint, expected):
        with (
            patch("registry.core.telemetry.settings") as mock_settings,
            patch(
                "registry.core.telemetry._get_operator_cloud_hint",
                return_value=hint,
            ),
            patch("registry.core.telemetry._detect_cloud_provider_with_method") as mock_cascade,
        ):
            mock_settings.mcp_cloud_provider = None
            result = await _resolve_cloud()
        assert result == expected
        mock_cascade.assert_not_called()

    async def test_no_hint_falls_through_to_cascade(self, _clean_env):
        with (
            patch("registry.core.telemetry.settings") as mock_settings,
            patch(
                "registry.core.telemetry._get_operator_cloud_hint",
                return_value=None,
            ),
            patch(
                "registry.core.telemetry._detect_cloud_provider_with_method",
                return_value=("unknown", _DETECTION_METHOD_UNKNOWN),
            ),
        ):
            mock_settings.mcp_cloud_provider = None
            result = await _resolve_cloud()
        assert result == ("unknown", _DETECTION_METHOD_UNKNOWN)

    async def test_explicit_setting_beats_hint(self, _clean_env):
        with (
            patch("registry.core.telemetry.settings") as mock_settings,
            patch(
                "registry.core.telemetry._get_operator_cloud_hint",
                return_value="azure",
            ) as mock_hint,
        ):
            mock_settings.mcp_cloud_provider = "aws"
            result = await _resolve_cloud()
        assert result == ("aws", _DETECTION_METHOD_EXPLICIT)
        mock_hint.assert_not_called()
