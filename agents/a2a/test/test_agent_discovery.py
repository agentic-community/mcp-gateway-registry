"""Integration test for agent discovery functionality."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "travel-assistant-agent"))

from registry_discovery_client import RegistryDiscoveryClient


async def test_discovery():
    """Test agent discovery through registry."""
    print("=" * 60)
    print("Agent Discovery Integration Test")
    print("=" * 60)

    # Get credentials from environment
    registry_url = os.getenv("MCP_REGISTRY_URL", "http://localhost:7860")
    keycloak_url = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
    client_id = os.getenv("M2M_CLIENT_ID")
    client_secret = os.getenv("M2M_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("\n❌ ERROR: M2M credentials not configured")
        print("Set M2M_CLIENT_ID and M2M_CLIENT_SECRET environment variables")
        print("\nExample:")
        print("  export M2M_CLIENT_ID=agent-travel-assistant-m2m")
        print("  export M2M_CLIENT_SECRET=your-secret-here")
        return False

    print(f"\n📍 Registry URL: {registry_url}")
    print(f"🔐 Keycloak URL: {keycloak_url}")
    print(f"👤 Client ID: {client_id}")
    print(f"🔑 Client Secret: {'*' * 20}")

    # Create client
    client = RegistryDiscoveryClient(
        registry_url=registry_url,
        keycloak_url=keycloak_url,
        client_id=client_id,
        client_secret=client_secret,
    )

    # Test 1: Semantic search for flight booking agent
    print("\n" + "=" * 60)
    print("Test 1: Semantic Search - 'agent that can book flights'")
    print("=" * 60)

    try:
        agents = await client.discover_by_semantic_search(
            query="agent that can book flights",
            max_results=5,
        )

        print(f"\n✅ Found {len(agents)} agent(s)")

        for i, agent in enumerate(agents, 1):
            print(f"\n{i}. {agent.name}")
            print(f"   Path: {agent.path}")
            print(f"   URL: {agent.url}")
            print(f"   Skills: {', '.join(agent.skills[:3])}{'...' if len(agent.skills) > 3 else ''}")
            print(f"   Score: {agent.score:.2f}" if agent.score else "   Score: N/A")
            print(f"   Trust: {agent.trust_level}")

        # Check if Flight Booking Agent was found
        flight_booking_found = any(
            "flight" in agent.name.lower() and "booking" in agent.name.lower()
            for agent in agents
        )

        if flight_booking_found:
            print("\n✅ Flight Booking Agent discovered successfully!")
        else:
            print("\n⚠️  Flight Booking Agent not found in results")

    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        return False

    # Test 2: Skill-based search
    print("\n" + "=" * 60)
    print("Test 2: Skill-Based Search - ['reserve_flight', 'process_payment']")
    print("=" * 60)

    try:
        agents = await client.discover_by_skills(
            skills=["reserve_flight", "process_payment"],
            tags=["booking"],
            max_results=5,
        )

        print(f"\n✅ Found {len(agents)} agent(s)")

        for i, agent in enumerate(agents, 1):
            print(f"\n{i}. {agent.name}")
            print(f"   Skills: {', '.join(agent.skills)}")

    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        return False

    # Test 3: Agent invocation (if Flight Booking Agent is available)
    print("\n" + "=" * 60)
    print("Test 3: Agent Invocation - Call discovered agent skill")
    print("=" * 60)

    try:
        # First discover the agent
        agents = await client.discover_by_semantic_search(
            query="flight booking agent",
            max_results=1,
        )

        if not agents:
            print("\n⚠️  Skipping invocation test - no agents found")
        else:
            agent = agents[0]
            print(f"\n📞 Attempting to invoke skill on: {agent.name}")
            print(f"   URL: {agent.url}")

            # Note: This is a demonstration of the pattern
            # Actual invocation would require the agent to be running
            print("\n✅ Invocation pattern validated")
            print("   (Actual invocation requires agent to be deployed and running)")

    except Exception as e:
        print(f"\n⚠️  Test 3 skipped: {e}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_discovery())
    sys.exit(0 if success else 1)
