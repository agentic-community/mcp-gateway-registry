#!/usr/bin/env python3
"""
Script to create API keys for the metrics service.
Run this script to generate API keys for different services.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.storage.database import init_database, MetricsStorage
from app.utils.helpers import generate_api_key, hash_api_key


async def create_api_key_for_service(service_name: str):
    """Create an API key for a service."""
    # Initialize database
    await init_database()

    # Generate API key
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)

    # Store in database
    storage = MetricsStorage()
    success = await storage.create_api_key(key_hash, service_name)

    if success:
        masked_key = f"{api_key[:12]}...{api_key[-4:]}" if len(api_key) > 16 else "***"
        print(f"API Key created for service '{service_name}':")
        print(f"API Key (masked): {masked_key}")
        print(f"Hash: {key_hash}")

        # Write full key to a file so it is not logged in plain text.
        # Create the file with owner-only permissions (0600) atomically so the
        # secret is never briefly world/group-readable between create and chmod.
        key_file = Path(f".api_key_{service_name}.txt")
        fd = os.open(str(key_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"METRICS_API_KEY={api_key}\n")
        os.chmod(key_file, 0o600)  # enforce 0600 even if the file pre-existed
        print(f"\nFull API key written to: {key_file}")
        print("Store the key securely and delete the file after use.")
        return api_key
    else:
        print(f"Failed to create API key for service '{service_name}'")
        return None


async def main():
    """Main function to create API keys for common services."""
    services = ["auth-server", "registry-service", "mcpgw-server", "test-client"]

    print("Creating API keys for MCP services...\n")

    for service in services:
        api_key = await create_api_key_for_service(service)
        if api_key:
            print("-" * 80)
        print()


if __name__ == "__main__":
    asyncio.run(main())
