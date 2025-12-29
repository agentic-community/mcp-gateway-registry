"""
Integration test conftest for EnforceAI tests.

This conftest ensures that ENFORCEAI_DB_PATH is set before any imports
so that the EnforceAI management routes are mounted on the FastAPI app.
"""

from __future__ import annotations

import os
import tempfile

# CRITICAL: This must be set BEFORE any imports of auth_server.server
# because the routes are mounted at module load time.
# We use a temporary path here; individual tests will override with their own.
_tmp_db = tempfile.NamedTemporaryFile(
    suffix=".db",
    delete=False,
)
_TEMP_DB_PATH = _tmp_db.name
_tmp_db.close()
os.environ.setdefault("ENFORCEAI_DB_PATH", _TEMP_DB_PATH)
