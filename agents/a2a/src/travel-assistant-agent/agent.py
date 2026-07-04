"""Tools for Travel Assistant Agent - Flight search and trip planning utilities."""

import json
import logging

from dependencies import get_db_manager, get_registry_client, get_remote_agent_cache
from strands import Agent, tool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


@tool
def search_flights(
    departure_city: str,
    arrival_city: str,
    departure_date: str,
) -> str:
    """Search for available flights between cities on a specific date."""
    logger.info(
        f"Tool called: search_flights(departure_city={departure_city}, arrival_city={arrival_city}, departure_date={departure_date})"
    )
    try:
        flights = get_db_manager().search_flights(departure_city, arrival_city, departure_date)

        result = {
            "query": {
                "departure_city": departure_city,
                "arrival_city": arrival_city,
                "departure_date": departure_date,
            },
            "flights": flights,
            "count": len(flights),
        }

        logger.debug(f"Flight search result:\n{json.dumps(result, indent=2)}")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.exception(f"Database error in search_flights: {e}")
        return json.dumps({"error": "An internal database error occurred"})


@tool
def check_prices(
    flight_id: int,
) -> str:
    """Get pricing and seat availability for a specific flight."""
    logger.info(f"Tool called: check_prices(flight_id={flight_id})")
    try:
        flight_details = get_db_manager().get_flight_details(flight_id)

        if not flight_details:
            error_msg = f"Flight with ID {flight_id} not found"
            logger.warning(error_msg)
            return json.dumps({"error": error_msg})

        logger.debug(f"Flight details result:\n{json.dumps(flight_details, indent=2)}")
        return json.dumps(flight_details, indent=2)

    except Exception as e:
        logger.exception(f"Database error in check_prices: {e}")
        return json.dumps({"error": "An internal database error occurred"})


@tool
def get_recommendations(
    max_price: float,
    preferred_airlines: list[str] | None = None,
) -> str:
    """Get flight recommendations based on customer preferences."""
    logger.info(
        f"Tool called: get_recommendations(max_price={max_price}, preferred_airlines={preferred_airlines})"
    )
    try:
        recommendations = get_db_manager().get_recommendations(max_price, preferred_airlines)

        result = {
            "criteria": {"max_price": max_price, "preferred_airlines": preferred_airlines or "Any"},
            "recommendations": recommendations,
            "count": len(recommendations),
        }

        logger.debug(f"Recommendations result:\n{json.dumps(result, indent=2)}")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.exception(f"Database error in get_recommendations: {e}")
        return json.dumps({"error": "An internal database error occurred"})


@tool
def create_trip_plan(
    departure_city: str,
    arrival_city: str,
    departure_date: str,
    return_date: str | None = None,
    budget: float | None = None,
) -> str:
    """Create and save a trip planning record."""
    logger.info(
        f"Tool called: create_trip_plan(departure_city={departure_city}, arrival_city={arrival_city}, departure_date={departure_date})"
    )
    logger.debug(f"Return date: {return_date}, Budget: {budget}")
    try:
        db_manager = get_db_manager()
        trip_plan_id = db_manager.create_trip_plan(
            departure_city, arrival_city, departure_date, return_date, budget
        )

        # Get available flights for the trip
        outbound_flights = db_manager.search_flights(departure_city, arrival_city, departure_date)
        return_flights = []

        if return_date:
            return_flights = db_manager.search_flights(arrival_city, departure_city, return_date)

        result = {
            "trip_plan_id": trip_plan_id,
            "trip_details": {
                "departure_city": departure_city.upper(),
                "arrival_city": arrival_city.upper(),
                "departure_date": departure_date,
                "return_date": return_date,
                "budget": budget,
                "status": "planning",
            },
            "outbound_flights": outbound_flights,
            "return_flights": return_flights,
            "next_steps": [
                "Review available flights",
                "Select preferred flights",
                "Contact Flight Booking Agent for reservation",
            ],
        }

        logger.debug(f"Trip plan result:\n{json.dumps(result, indent=2)}")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.exception(f"Database error in create_trip_plan: {e}")
        return json.dumps({"error": "An internal database error occurred"})


@tool
async def discover_remote_agents(query: str, max_results: int = 5) -> str:
    """
    Discover remote agents from the mcp-registry with natural language query.
    Cache them for visibility and invocation for later tool calls from LLM
    """
    logger.info(f"Tool called: discover_remote_agents(query='{query}', max_results={max_results})")

    try:
        registry_client = get_registry_client()
        if not registry_client:
            return json.dumps(
                {
                    "error": "Registry discovery not configured",
                    "message": "Set M2M_CLIENT_ID and M2M_CLIENT_SECRET environment variables",
                }
            )

        # Search registry
        discovered = await registry_client.discover_by_semantic_search(
            query=query,
            max_results=max_results,
        )

        if not discovered:
            return json.dumps(
                {
                    "query": query,
                    "agents_found": 0,
                    "message": "No agents found matching your query",
                }
            )

        # Cache the discovered agents. SECURITY: the registry-scoped token is
        # NOT forwarded to remote agents -- those endpoints are registrant-
        # controlled and not fully trusted, and the registry token could be
        # replayed against the registry API. Outbound A2A calls carry no
        # credential in this sample; a production deployment should supply a
        # delegation_token_provider that mints an audience-restricted, short-
        # lived token bound to each specific target agent.
        cache = get_remote_agent_cache()
        newly_cached = cache.cache_discovered_agents(discovered)

        result = {
            "query": query,
            "agents_found": len(discovered),
            "newly_cached": len(newly_cached),
            "total_cached": len(cache),
            "agents": [
                {
                    "id": agent.path,
                    "name": agent.name,
                    "description": agent.description,
                    "url": agent.url,
                    "skills": agent.skill_names,
                    "tags": agent.tags,
                    "relevance_score": agent.relevance_score,
                    "trust_level": agent.trust_level,
                }
                for agent in discovered
            ],
            "next_steps": [
                "Use view_cached_remote_agents() to see all cached agents",
                "Use invoke_remote_agent(agent_id, message) to call a specific agent",
            ],
        }

        logger.info(
            f"Discovery successful: found {len(discovered)} agents, cached {len(newly_cached)} new"
        )
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Discovery error in discover_remote_agents: {e}", exc_info=True)
        return json.dumps(
            {
                "error": "Discovery failed",
                "message": "An internal error occurred during agent discovery",
            }
        )


@tool
async def view_cached_remote_agents() -> str:
    """View all cached remote agents available for invocation."""
    logger.info("Tool called: view_cached_remote_agents()")

    try:
        cache = get_remote_agent_cache()

        if len(cache) == 0:
            return json.dumps(
                {
                    "total": 0,
                    "message": "No agents cached. Use discover_remote_agents() to find and cache agents.",
                }
            )

        all_agents = cache.get_all()
        result = {
            "total": len(cache),
            "agents": [
                {
                    "id": agent_id,
                    "name": agent_client.agent_name,
                    "url": agent_client.agent_url,
                    "skills": agent_client.skills,
                }
                for agent_id, agent_client in all_agents.items()
            ],
            "usage": "Use invoke_remote_agent(agent_id, message) to call any of these agents",
        }

        logger.info(f"Returning {len(cache)} cached agents")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error in view_cached_remote_agents: {e}", exc_info=True)
        return json.dumps(
            {
                "error": "Failed to view cached agents",
                "message": "An internal error occurred while viewing cached agents",
            }
        )


@tool
async def invoke_remote_agent(agent_id: str, message: str) -> str:
    """Invoke a cached remote agent by ID with a natural language message."""
    logger.info(
        f"Tool called: invoke_remote_agent(agent_id='{agent_id}', message='{message[:100]}...')"
    )

    try:
        cache = get_remote_agent_cache()

        if agent_id not in cache:
            all_agents = cache.get_all()
            available_ids = list(all_agents.keys())
            return json.dumps(
                {
                    "error": f"Agent '{agent_id}' not found in cache",
                    "available_agents": available_ids,
                    "hint": "Use discover_remote_agents() to find and cache agents, or view_cached_remote_agents() to see what's available",
                }
            )

        # Get the cached agent client and invoke it
        agent_client = cache.get(agent_id)
        logger.info(f"Invoking agent: {agent_client.agent_name}")

        response = await agent_client.send_message(message)

        logger.info(f"Successfully invoked {agent_client.agent_name}")
        return response

    except Exception as e:
        logger.error(f"Error in invoke_remote_agent: {e}", exc_info=True)
        return json.dumps(
            {
                "error": "Failed to invoke remote agent",
                "agent_id": agent_id,
                "message": "An internal error occurred while invoking the remote agent",
            }
        )


TRAVEL_ASSISTANT_TOOLS = [
    search_flights,
    check_prices,
    get_recommendations,
    create_trip_plan,
    discover_remote_agents,
    view_cached_remote_agents,
    invoke_remote_agent,
]

strands_agent = Agent(
    name="Travel Assistant Agent",
    description="Flight search and trip planning agent with dynamic agent discovery",
    tools=TRAVEL_ASSISTANT_TOOLS,
    callback_handler=None,
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
)


def get_agent_instance():
    return strands_agent
