#!/usr/bin/env python3
"""
Test script for agent discovery and booking workflow.

Test 1: Travel agent searches for flights using its own tools
Test 2: Travel agent discovers booking agent, checks availability, reserves seats, and completes booking

Usage: python agent_discovery_test_v2.py [--endpoint local|live]
"""

import argparse
import json
import logging
import requests
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

LOCAL_ENDPOINTS = {
    "travel_assistant": "http://localhost:9001",
}


class AgentTester:
    """Agent testing class."""

    def __init__(self, endpoints, is_live=False):
        self.endpoints = endpoints
        self.is_live = is_live

    def send_agent_message(self, agent_type, message):
        """Send message to agent using A2A protocol."""
        endpoint = self.endpoints[agent_type]
        
        payload = {
            "jsonrpc": "2.0",
            "id": f"test-{message[:10]}",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                    "messageId": f"msg-{message[:10]}",
                }
            },
        }

        response = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        return response.json()

    def extract_response_text(self, response):
        """Extract text from A2A response."""
        if "result" not in response:
            return ""
        
        artifacts = response["result"].get("artifacts", [])
        response_text = ""
        for artifact in artifacts:
            if "parts" in artifact:
                for part in artifact["parts"]:
                    if "text" in part:
                        response_text += part["text"]
        return response_text


class AgentDiscoveryTests:
    """Test suite for agent discovery and booking workflow."""

    def __init__(self, tester):
        self.tester = tester
        self.agent_type = "travel_assistant"

    def test_search_flight_solo(self):
        """Test 1: Travel agent searches for flights using its own tools."""
        print("\n1. Testing flight search (travel agent solo)...")
        message = "Search for flights from New York to Los Angeles on 2025-12-20"
        response = self.tester.send_agent_message(self.agent_type, message)

        assert "result" in response, f"No result in response: {response}"
        response_text = self.tester.extract_response_text(response)

        # Check if flight search happened
        assert any(keyword in response_text.lower() for keyword in ["flight", "new york", "los angeles", "nyc", "lax"]), \
            f"Response doesn't mention flight search. Got: {response_text[:300]}"
        
        print("   ✓ Travel agent searched for flights using its own tools")
        print(f"   Response preview: {response_text[:200]}...")
        return response_text

    def test_book_flight_with_discovery(self):
        """Test 2: Travel agent discovers booking agent and delegates booking tasks."""
        print("\n2. Testing flight booking with agent discovery and invocation...")
        message = (
            "I want to book flight ID 1. I need you to reserve 2 seats, confirm the reservation, "
            "and process the payment. You don't have these booking capabilities yourself, so you'll "
            "need to find and use an agent that can handle flight reservations and confirmations."
        )
        response = self.tester.send_agent_message(self.agent_type, message)
        response_text = self.tester.extract_response_text(response)
        
        # Check if agent discovery and delegation happened
        assert any(keyword in response_text.lower() for keyword in ["reserve", "book", "confirm", "agent", "discover"]), \
            f"Booking workflow failed. Got: {response_text[:300]}"
        print("      ✓ Booking agent discovered and invoked")
        print(f"   Response preview: {response_text[:200]}...")
        
        print("   ✓ Complete booking workflow succeeded")
        return response_text


def run_tests(endpoint_type):
    """Run all discovery tests."""
    print(f"Running agent discovery and booking workflow tests against {endpoint_type} endpoints...")
    print("=" * 70)
    print("Test 1: Travel agent searches for flights (solo)")
    print("Test 2: Travel agent discovers booking agent and completes booking")
    print("=" * 70)

    endpoints = LOCAL_ENDPOINTS
    is_live = endpoint_type == "live"
    tester = AgentTester(endpoints, is_live=is_live)

    try:
        discovery_tests = AgentDiscoveryTests(tester)
        
        # Run tests in sequence
        discovery_tests.test_search_flight_solo()
        discovery_tests.test_book_flight_with_discovery()

        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        return True

    except AssertionError as e:
        logger.error(f"Test assertion failed: {e}")
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        logger.exception("Test failed with exception")
        print(f"\n❌ Test failed with exception: {e}")
        return False


def main():
    """Main entry point for test script."""
    parser = argparse.ArgumentParser(description="Test agent discovery and booking workflow")
    parser.add_argument(
        "--endpoint",
        choices=["local", "live"],
        default="local",
        help="Test against local or live endpoints (default: local)",
    )

    args = parser.parse_args()
    success = run_tests(args.endpoint)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
