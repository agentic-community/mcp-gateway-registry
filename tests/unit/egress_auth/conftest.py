"""Shared fixtures for the egress-auth unit tests."""

import pytest

from registry.egress_auth import providers
from tests.unit.egress_auth.dcr_fixtures import (
    DCR_TEST_PROVIDER,
    make_dcr_test_config,
)


@pytest.fixture
def dcr_provider(monkeypatch):
    """Register the test-local ``dcr-test`` provider in the built-in table."""
    cfg = make_dcr_test_config()
    monkeypatch.setitem(providers.PROVIDER_REGISTRY, DCR_TEST_PROVIDER, cfg)
    return cfg
