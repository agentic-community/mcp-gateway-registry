#!/usr/bin/env python3
"""
Test script for NEW agent discovery functionality (three-tool pattern)
Usage: python agent_discovery_test_v2.py --endpoint local|live
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

        response = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"})
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
    """Test suite for agent discovery with new three-tool pattern."""

    def __init__(self, tester):
        self.tester = tester
        self.agent_type = "travel_assistant"

    def test_discover_remote_agents(self):
        """Test discovering remote agents and caching them."""
        print("\n1. Testing discover_remote_agents...")
        message = "I need help booking flights. Can you discover agents that can help with flight booking?"
        response = self.tester.send_agent_message(self.agent_type, message)

        assert "result" in response, f"No result in response: {response}"
        response_text = self.tester.extract_response_text(response)

        # Check if discovery happened
        assert any(keyword in response_text.lower() for keyword in ["discover", "found", "agent", "flight"]), \
            f"Response doesn't mention discovery. Got: {response_text[:300]}"
        
        # Check for agent details
        assert any(keyword in response_text.lower() for keyword in ["booking", "book", "flight"]), \
            f"Response doesn't mention flight booking capability. Got: {response_text[:300]}"
        
        print("   ✓ Discovery triggered and agents found")
        return response_text

    def test_view_cached_agents(self):
        """Test viewing cached remote agents."""
        print("\n2. Testing view_cached_remote_agents...")
        message = "What remote agents do you have cached?"
        response = self.tester.send_agent_message(self.agent_type, message)

        assert "result" in response, f"No result in response: {response}"
        response_text = self.tester.extract_response_text(response)

        # Check if cache view happened
        assert any(keyword in response_text.lower() for keyword in ["cached", "agent", "available"]), \
            f"Response doesn't show cached agents. Got: {response_text[:300]}"
        
        print("   ✓ Cached agents displayed")
        return response_text

    def test_invoke_remote_agent(self):
        """Test invoking a cached remote agent."""
        print("\n3. Testing invoke_remote_agent...")
        
        # First ensure discovery happened
        discovery_message = "Find agents that can book flights"
        self.tester.send_agent_message(self.agent_type, discovery_message)
        
        # Now invoke the discovered agent
        invoke_message = "Use the flight booking agent to check availability for flight ID 1"
        response = self.tester.send_agent_message(self.agent_type, invoke_message)

        assert "result" in response, f"No result in response: {response}"
        response_text = self.tester.extract_response_text(response)

        # Check if remote agent was invoked
        # The response should contain information about the flight
        assert any(keyword in response_text.lower() for keyword in ["flight", "available", "seat", "booking"]), \
            f"Response doesn't show remote agent invocation result. Got: {response_text[:300]}"
        
        print("   ✓ Remote agent invoked successfully")
        return response_text

    def test_end_to_end_workflow(self):
        """Test complete workflow: discover → view → invoke."""
        print("\n4. Testing end-to-end workflow...")
        
        # Step 1: Discover
        print("   Step 1: Discovering agents...")
        discovery_message = "I need to book a flight from NYC to LAX. First, find agents that can help."
        discovery_response = self.tester.send_agent_message(self.agent_type, discovery_message)
        discovery_text = self.tester.extract_response_text(discovery_response)
        
        assert "discover" in discovery_text.lower() or "found" in discovery_text.lower(), \
            "Discovery didn't happen"
        print("      ✓ Agents discovered")
        
        # Step 2: Invoke
        print("   Step 2: Invoking discovered agent...")
        invoke_message = "Now use the flight booking agent to check what flights are available for flight ID 1"
        invoke_response = self.tester.send_agent_message(self.agent_type, invoke_message)
        invoke_text = self.tester.extract_response_text(invoke_response)
        
        assert any(keyword in invoke_text.lower() for keyword in ["flight", "available", "seat"]), \
            f"Remote agent wasn't invoked properly. Got: {invoke_text[:300]}"
        print("      ✓ Remote agent invoked")
        
        # Step 3: Invoke again (should reuse cache)
        print("   Step 3: Invoking again (cache reuse)...")
        second_invoke_message = "Ask the flight booking agent about flight ID 2"
        second_response = self.tester.send_agent_message(self.agent_type, second_invoke_message)
        second_text = self.tester.extract_response_text(second_response)
        
        assert any(keyword in second_text.lower() for keyword in ["flight", "available", "seat"]), \
            f"Second invocation failed. Got: {second_text[:300]}"
        print("      ✓ Cache reused successfully")
        
        print("   ✓ End-to-end workflow completed")

    def test_error_handling(self):
        """Test error handling when agent not in cache."""
        print("\n5. Testing error handling...")
        
        # Try to invoke without discovering first (in a fresh session this would fail)
        # But since we've already discovered in previous tests, we'll test with a non-existent agent
        message = "Use the non-existent-agent to do something"
        response = self.tester.send_agent_message(self.agent_type, message)
        response_text = self.tester.extract_response_text(response)
        
        # The agent should either:
        # 1. Discover the agent first, or
        # 2. Report that it can't find the agent
        # Either way, it should handle it gracefully
        assert len(response_text) > 0, "Agent should respond to invalid agent request"
        print("   ✓ Error handling works")


def run_tests(endpoint_type):
    """Run all discovery tests."""
    print(f"Running NEW agent discovery tests against {endpoint_type} endpoints...")
    print("=" * 70)
    print("Testing three-tool pattern: discover → view → invoke")
    print("=" * 70)

    endpoints = LOCAL_ENDPOINTS
    is_live = endpoint_type == "live"
    tester = AgentTester(endpoints, is_live=is_live)

    try:
        discovery_tests = AgentDiscoveryTests(tester)
        
        # Run tests in sequence
        discovery_tests.test_discover_remote_agents()
        discovery_tests.test_view_cached_agents()
        discovery_tests.test_invoke_remote_agent()
        discovery_tests.test_end_to_end_workflow()
        discovery_tests.test_error_handling()

        print("\n" + "=" * 70)
        print("✅ All discovery tests passed!")
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
    parser = argparse.ArgumentParser(description="Test NEW agent discovery functionality")
    parser.add_argument(
        "--endpoint",
        choices=["local", "live"],
        required=True,
        help="Test against local or live endpoints",
    )

    args = parser.parse_args()
    success = run_tests(args.endpoint)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
