#!/usr/bin/env python3
"""Add a test API key to the database"""

import asyncio
import aiosqlite
import sys
from pathlib import Path

# Compute the hash with the same peppered HMAC the service uses, so the
# inserted row matches what verify_api_key looks up. Requires METRICS_KEY_PEPPER
# to be set in the environment (same value the running service uses).
sys.path.insert(0, str(Path(__file__).parent))
from app.utils.helpers import hash_api_key


async def add_test_key():
    db_path = "/var/lib/sqlite/metrics.db"
    api_key = "test_key_123"
    key_hash = hash_api_key(api_key)

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO api_keys (key_hash, service_name, created_at, is_active)
            VALUES (?, ?, datetime('now'), 1)
        """,
            (key_hash, "test-service"),
        )
        await db.commit()
        print(f"Added test API key for service: test-service")
        print(f"API Key: {api_key}")
        print(f"Key Hash: {key_hash}")


if __name__ == "__main__":
    asyncio.run(add_test_key())
