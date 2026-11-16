"""Clients cannot submit a metadata verdict on PUT or PATCH (issue #1811).

The repositories recompute ``metadata_scan`` on every write regardless (see
tests/unit/repositories/test_metadata_scan_stamping.py); these pin that the
strict request models reject it outright rather than silently accepting it.
"""

import pytest
from pydantic import ValidationError

from registry.schemas.agent_models import AgentCardPatch
from registry.schemas.server_update_models import ServerCardPatch, ServerUpdateRequest

pytestmark = pytest.mark.unit

FORGED = {"status": "clean", "metadata_digest": "sha256:forged"}


def test_server_put_rejects_metadata_scan():
    with pytest.raises(ValidationError, match="metadata_scan"):
        ServerUpdateRequest(server_name="s", description="d", metadata_scan=FORGED)


def test_server_patch_rejects_metadata_scan():
    with pytest.raises(ValidationError, match="metadata_scan"):
        ServerCardPatch(metadata_scan=FORGED)


def test_agent_patch_rejects_metadata_scan():
    with pytest.raises(ValidationError, match="metadata_scan"):
        AgentCardPatch(metadata_scan=FORGED)
