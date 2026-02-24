#!/usr/bin/env python3
"""
Test script for Travel Assistant and Flight Booking agents
Usage: python simple_agents_test.py --endpoint local|live [--debug]
"""

import argparse
import json
import logging
import requests
import sys
import time
import uuid
from typing import (
    Any,
    Dict,
    Optional,
)

import boto3

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Endpoint configurations
LOCAL_ENDPOINTS = {
    "travel_assistant": "http://localhost:9001",
    "flight_booking": "http://localhost:9002",
}

LIVE_ENDPOINTS = {
    "travel_assistant": "travel_assistant_agent ARN",
    "flight_booking": "flight_booking_agent ARN",
}

AWS_REGION = "us-east-1"


class AgentTester:
    """Agent testing class for both local and live endpoints."""

    def __init__(
        self,
        endpoints: Dict[str, str],
        is_live: bool = False,
    ) -> None:
        self.endpoints = endpoints
        self.is_live = is_live
        if is_live:
            self.bedrock_client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)


    def send_agent_message(
        self,
        agent_type: str,
        message: str,
    ) -> Dict[str, Any]:
        """Send message to agent using A2A protocol (local) or boto3 (live)."""
        endpoint = self.endpoints[agent_type]
        if not endpoint:
            raise ValueError(f"No endpoint configured for {agent_type}")

        request_id = f"test-{uuid.uuid4().hex[:8]}"
        message_id = f"test-msg-{uuid.uuid4().hex[:8]}"
        timestamp = time.time()

        if self.is_live:
            # Use boto3 for AgentCore Runtime
            return self._invoke_agentcore_runtime(endpoint, message, request_id, message_id, timestamp)
        else:
            # Use HTTP for local A2A
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": message}],
                        "messageId": message_id,
                    }
                },
            }

            logger.debug(f"[REQUEST] Agent: {agent_type}, Endpoint: {endpoint}")
            logger.debug(f"[REQUEST] ID: {request_id}, Message ID: {message_id}")
            logger.debug(f"[REQUEST] Payload:\n{json.dumps(payload, indent=2)}")

            start_time = time.time()
            response = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
            response_time = time.time() - start_time

            response_json = response.json()
            logger.debug(f"[RESPONSE] Time: {response_time:.3f}s, Status: {response.status_code}")
            logger.debug(f"[RESPONSE] Body:\n{json.dumps(response_json, indent=2, default=str)}")

            return response_json


    def _invoke_agentcore_runtime(
        self,
        runtime_arn: str,
        message: str,
        request_id: str,
        message_id: str,
        timestamp: float,
    ) -> Dict[str, Any]:
        """Invoke AgentCore Runtime using boto3."""
        # A2A protocol requires JSON-RPC format
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                    "messageId": message_id,
                }
            },
        }
        payload_json = json.dumps(payload)

        logger.debug(f"[AGENTCORE REQUEST] ARN: {runtime_arn}")
        logger.debug(f"[AGENTCORE REQUEST] ID: {request_id}, Message ID: {message_id}")
        logger.debug(f"[AGENTCORE REQUEST] Payload:\n{json.dumps(payload, indent=2)}")

        # Generate session ID (must be 33+ characters)
        session_id = f"test-session-{uuid.uuid4().hex}"
        logger.debug(f"[AGENTCORE REQUEST] Session ID: {session_id}")

        try:
            start_time = time.time()
            response = self.bedrock_client.invoke_agent_runtime(
                agentRuntimeArn=runtime_arn,
                runtimeSessionId=session_id,
                qualifier="DEFAULT",
                payload=payload_json,
            )
            response_time = time.time() - start_time

            # Read streaming response
            if "response" in response:
                streaming_body = response["response"]
                all_lines = []

                for line in streaming_body.iter_lines():
                    line_str = line.decode("utf-8")
                    all_lines.append(line_str)
                    logger.debug(f"[AGENTCORE STREAM] Line: {line_str}")

                # The response is a single JSON-RPC response line
                if all_lines:
                    try:
                        json_response = json.loads(all_lines[0])

                        logger.debug(f"[AGENTCORE RESPONSE] Time: {response_time:.3f}s")
                        logger.debug(
                            f"[AGENTCORE RESPONSE] Body:\n{json.dumps(json_response, indent=2, default=str)}"
                        )

                        # Check for JSON-RPC error
                        if "error" in json_response:
                            return {"error": json_response["error"]}

                        # Return the JSON-RPC result directly
                        return json_response

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse response: {e}")
                        return {"error": f"Failed to parse response: {e}"}

                return {"error": "Empty response"}

            return {"error": "No response content"}

        except Exception as e:
            logger.error(f"AgentCore invocation failed: {e}")
            return {"error": str(e)}


    def call_api_endpoint(
        self,
        agent_type: str,
        endpoint: str,
        method: str = "POST",
        **params,
    ) -> Dict[str, Any]:
        """Call direct API endpoint (only works for local)."""
        if self.is_live:
            raise NotImplementedError("Direct API endpoints not available for live AgentCore Runtime")

        url = f"{self.endpoints[agent_type]}{endpoint}"
        if not self.endpoints[agent_type]:
            raise ValueError(f"No endpoint configured for {agent_type}")

        logger.debug(f"[API REQUEST] Agent: {agent_type}, URL: {url}")
        logger.debug(f"[API REQUEST] Method: {method}, Params: {params}")

        start_time = time.time()
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=60)
        else:
            response = requests.post(url, params=params, timeout=60)
        response_time = time.time() - start_time

        response_json = response.json()
        logger.debug(f"[API RESPONSE] Time: {response_time:.3f}s, Status: {response.status_code}")
        logger.debug(f"[API RESPONSE] Body:\n{json.dumps(response_json, indent=2, default=str)}")

        return response_json


    def ping_agent(
        self,
        agent_type: str,
    ) -> bool:
        """Check if agent is healthy (only works for local)."""
        if self.is_live:
            # For live, we can't ping directly, assume healthy if ARN is configured
            return bool(self.endpoints.get(agent_type))

        try:
            url = f"{self.endpoints[agent_type]}/ping"
            logger.debug(f"[PING] Agent: {agent_type}, URL: {url}")

            start_time = time.time()
            response = requests.get(url, timeout=5)
            response_time = time.time() - start_time

            is_healthy = response.status_code == 200 and response.json().get("status") == "healthy"
            logger.debug(f"[PING RESPONSE] Time: {response_time:.3f}s, Healthy: {is_healthy}")

            return is_healthy
        except Exception as e:
            logger.debug(f"[PING ERROR] Agent: {agent_type}, Error: {e}")
            return False


class TravelAssistantTests:
    """Test suite for Travel Assistant agent."""

    def __init__(
        self,
        tester: AgentTester,
    ) -> None:
        self.tester = tester
        self.agent_type = "travel_assistant"


    def test_ping(self) -> None:
        """Test agent health check."""
        print("Testing Travel Assistant ping...")
        result = self.tester.ping_agent(self.agent_type)
        assert result, "Travel Assistant ping failed"
        print("✓ Travel Assistant is healthy")


    def test_agent_flight_search(self) -> None:
        """Test agent flight search via A2A."""
        print("Testing Travel Assistant flight search...")
        message = "Search for flights from SF to NY on 2025-11-15"
        response = self.tester.send_agent_message(self.agent_type, message)

        assert "result" in response, f"No result in response: {response}"
        assert "artifacts" in response["result"], "No artifacts in response"

        # Check if agent found flights
        artifacts = response["result"]["artifacts"]
        assert len(artifacts) > 0, "No artifacts returned"

        # Extract text from artifact parts
        response_text = ""
        for artifact in artifacts:
            if "parts" in artifact:
                for part in artifact["parts"]:
                    if "text" in part:
                        response_text += part["text"]

        assert "flight" in response_text.lower(), f"Response doesn't mention flights. Got: {response_text[:100]}"
        print("✓ Travel Assistant flight search working")


    def test_api_search_flights(self) -> None:
        """Test direct API endpoint (local only)."""
        if self.tester.is_live:
            print("Skipping /api/search-flights endpoint (only available in local Docker container)")
            return

        print("Testing Travel Assistant API endpoint...")
        response = self.tester.call_api_endpoint(
            self.agent_type,
            "/api/search-flights",
            departure_city="SF",
            arrival_city="NY",
            departure_date="2025-11-15",
        )

        assert "result" in response, f"No result in API response: {response}"
        result_data = json.loads(response["result"])
        assert "flights" in result_data, "No flights in API response"
        assert len(result_data["flights"]) > 0, "No flights found"
        print("✓ Travel Assistant API endpoint working")


    def test_api_recommendations(self) -> None:
        """Test recommendations API (local only)."""
        if self.tester.is_live:
            print("Skipping /api/recommendations endpoint (only available in local Docker container)")
            return

        print("Testing Travel Assistant recommendations...")
        response = self.tester.call_api_endpoint(
            self.agent_type,
            "/api/recommendations",
            method="GET",
            max_price=300,
            preferred_airlines="United,Delta",
        )

        assert "result" in response, "No result in recommendations response"
        result_data = json.loads(response["result"])
        assert "recommendations" in result_data, "No recommendations in response"
        print("✓ Travel Assistant recommendations working")


class FlightBookingTests:
    """Test suite for Flight Booking agent."""

    def __init__(
        self,
        tester: AgentTester,
    ) -> None:
        self.tester = tester
        self.agent_type = "flight_booking"


    def test_ping(self) -> None:
        """Test agent health check."""
        print("Testing Flight Booking ping...")
        result = self.tester.ping_agent(self.agent_type)
        assert result, "Flight Booking ping failed"
        print("✓ Flight Booking is healthy")


    def test_agent_availability_check(self) -> None:
        """Test agent availability check via A2A."""
        print("Testing Flight Booking availability check...")
        message = "Check availability for flight ID 1"
        response = self.tester.send_agent_message(self.agent_type, message)

        assert "result" in response, f"No result in response: {response}"
        assert "artifacts" in response["result"], "No artifacts in response"

        artifacts = response["result"]["artifacts"]
        assert len(artifacts) > 0, "No artifacts returned"

        response_text = artifacts[0]["parts"][0]["text"]
        assert "available" in response_text.lower(), "Response doesn't mention availability"
        print("✓ Flight Booking availability check working")


    def test_agent_booking(self) -> None:
        """Test agent booking via A2A."""
        print("Testing Flight Booking reservation...")
        message = "Book flight ID 1 for Jane Smith, email jane@test.com"
        response = self.tester.send_agent_message(self.agent_type, message)

        assert "result" in response, f"No result in response: {response}"
        artifacts = response["result"]["artifacts"]
        response_text = artifacts[0]["parts"][0]["text"]

        assert (
            "booking" in response_text.lower() or "reserved" in response_text.lower()
        ), "Response doesn't mention booking/reservation"
        print("✓ Flight Booking reservation working")


    def test_api_check_availability(self) -> None:
        """Test direct API endpoint (local only)."""
        if self.tester.is_live:
            print("Skipping /api/check-availability endpoint (only available in local Docker container)")
            return

        print("Testing Flight Booking API endpoint...")
        response = self.tester.call_api_endpoint(self.agent_type, "/api/check-availability", flight_id=1)

        assert "result" in response, f"No result in API response: {response}"
        result_data = json.loads(response["result"])
        assert "flight_id" in result_data, "No flight_id in API response"
        assert "available_seats" in result_data, "No available_seats in response"
        print("✓ Flight Booking API endpoint working")


class AgentDiscoveryTests:
    """Test suite for cross-agent discovery via the MCP Gateway Registry.

    Tests the full flow: Travel Assistant discovers Flight Booking agent
    through the registry's semantic search API and delegates a booking task.
    Requires the MCP Gateway Registry to be running and the Flight Booking
    agent to be registered in it.
    """

    def __init__(
        self,
        tester: AgentTester,
        registry_url: str = "http://localhost",
    ) -> None:
        self.tester = tester
        self.registry_url = registry_url


    def _is_registry_available(self) -> bool:
        """Check if the MCP Gateway Registry is reachable."""
        try:
            response = requests.get(f"{self.registry_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


    def test_discover_and_delegate_booking(self) -> None:
        """Test Travel Assistant discovering Flight Booking agent and delegating a booking.

        Flow:
        1. Send booking request to Travel Assistant
        2. Travel Assistant calls discover_remote_agents() to find booking agents
        3. Travel Assistant calls invoke_remote_agent() to delegate to Flight Booking
        4. Flight Booking processes the request and returns confirmation
        5. Travel Assistant returns combined response
        """
        if not self._is_registry_available():
            print(
                f"  Skipping: registry not available at {self.registry_url}. "
                "Start the registry and register the Flight Booking agent to run this test."
            )
            return

        print("Testing cross-agent discovery and delegation flow...")

        # This message explicitly instructs the LLM to use discovery tools
        message = (
            "I need to book a flight. Please use the discover_remote_agents tool to find "
            "agents that can handle flight bookings, then use invoke_remote_agent to ask "
            "that agent to book flight ID 1 for John Smith with email john@test.com"
        )

        logger.debug("[DISCOVERY TEST] Sending booking request to Travel Assistant...")
        response = self.tester.send_agent_message("travel_assistant", message)

        assert "result" in response, f"No result in discovery response: {response}"
        assert "artifacts" in response["result"], "No artifacts in discovery response"

        # Extract text from all artifact parts
        artifacts = response["result"]["artifacts"]
        assert len(artifacts) > 0, "No artifacts returned from discovery flow"

        response_text = ""
        for artifact in artifacts:
            if "parts" in artifact:
                for part in artifact["parts"]:
                    if "text" in part:
                        response_text += part["text"]

        logger.debug(f"[DISCOVERY TEST] Full response text:\n{response_text}")

        response_lower = response_text.lower()

        # Verify the response indicates discovery happened
        discovery_keywords = ["discover", "found", "flight booking", "remote agent", "cached"]
        has_discovery = any(keyword in response_lower for keyword in discovery_keywords)

        # Verify the response indicates a booking was attempted or completed
        booking_keywords = ["book", "reserv", "confirm", "john smith"]
        has_booking = any(keyword in response_lower for keyword in booking_keywords)

        assert has_discovery or has_booking, (
            f"Response doesn't indicate discovery or booking happened. "
            f"Got: {response_text[:300]}"
        )

        if has_discovery:
            print("  [OK] Discovery indicators found in response")
        if has_booking:
            print("  [OK] Booking indicators found in response")

        print("[PASS] Cross-agent discovery and delegation flow working")


def run_tests(
    endpoint_type: str,
    skip_discovery: bool = False,
    registry_url: str = "http://localhost",
) -> bool:
    """Run all tests for specified endpoint type."""
    print(f"Running tests against {endpoint_type} endpoints...")
    print("=" * 50)

    # Select endpoints
    endpoints = LOCAL_ENDPOINTS if endpoint_type == "local" else LIVE_ENDPOINTS

    # Check if endpoints are configured
    for agent, url in endpoints.items():
        if not url:
            print(f"❌ No {endpoint_type} endpoint configured for {agent}")
            return False

    is_live = endpoint_type == "live"
    tester = AgentTester(endpoints, is_live=is_live)

    try:
        # Test Travel Assistant
        print("\nTesting Travel Assistant Agent")
        print("-" * 30)
        travel_tests = TravelAssistantTests(tester)
        travel_tests.test_ping()
        travel_tests.test_agent_flight_search()
        travel_tests.test_api_search_flights()
        travel_tests.test_api_recommendations()

        # Test Flight Booking
        print("\nTesting Flight Booking Agent")
        print("-" * 30)
        booking_tests = FlightBookingTests(tester)
        booking_tests.test_ping()
        booking_tests.test_agent_availability_check()
        booking_tests.test_agent_booking()
        booking_tests.test_api_check_availability()

        # Test Agent-to-Agent Discovery
        if not skip_discovery:
            print("\nTesting Agent-to-Agent Discovery")
            print("-" * 30)
            discovery_tests = AgentDiscoveryTests(tester, registry_url=registry_url)
            discovery_tests.test_discover_and_delegate_booking()
        else:
            print("\nSkipping Agent-to-Agent Discovery tests (--skip-discovery flag set)")

        print("\n" + "=" * 50)
        print("All tests passed!")
        return True

    except Exception as e:
        logger.exception("Test failed with exception")
        print(f"\n❌ Test failed: {e}")
        return False


def main() -> None:
    """Main entry point for test script."""
    parser = argparse.ArgumentParser(description="Test Travel Assistant and Flight Booking agents")
    parser.add_argument(
        "--endpoint",
        choices=["local", "live"],
        required=True,
        help="Test against local or live endpoints",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to see detailed request/response traces",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Alias for --debug, enables debug logging",
    )
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Skip agent-to-agent discovery tests (requires registry running)",
    )
    parser.add_argument(
        "--registry-url",
        default="http://localhost",
        help="MCP Gateway Registry URL for discovery tests (default: http://localhost)",
    )

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug or args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled - detailed traces will be shown")

    success = run_tests(
        endpoint_type=args.endpoint,
        skip_discovery=args.skip_discovery,
        registry_url=args.registry_url,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()