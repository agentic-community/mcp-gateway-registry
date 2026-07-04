"""Travel Assistant Agent - Main application module."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from agent import (
    check_prices,
    create_trip_plan,
    get_recommendations,
    search_flights,
    strands_agent,
)
from auth_middleware import install_agent_auth
from dependencies import (
    get_db_manager,
    get_env,
)
from fastapi import FastAPI
from strands.multiagent.a2a import A2AServer

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

env_settings = get_env()

# Use agent instance from tools module
a2a_server = A2AServer(agent=strands_agent, http_url=env_settings.agent_url, serve_at_root=True)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """Application lifespan manager."""
    # Setups before server startup
    get_db_manager()
    logger.info("Travel Assistant Agent starting up")
    logger.info(f"Agent URL: {env_settings.agent_url}")
    logger.info(f"Listening on {env_settings.host}:{env_settings.port}")

    yield
    # Triggered after server shutdown
    logger.info("Travel Assistant Agent shutting down")


app = FastAPI(title="Travel Assistant Agent", lifespan=lifespan)

# Authenticate every request (except health probes) before it reaches the A2A
# mount or the /api endpoints, which drive the LLM tool loop. Fails closed.
install_agent_auth(
    app,
    keycloak_url=env_settings.keycloak_url,
    realm=env_settings.keycloak_realm,
    audience=env_settings.agent_audience,
    bind_host=env_settings.host,
)


@app.get("/ping")
def ping():
    """Health check endpoint."""
    logger.debug("Ping endpoint called")
    return {"status": "healthy"}


@app.get("/api/health")
def health():
    """Health status endpoint."""
    logger.debug("Health endpoint called")
    return {"status": "healthy", "agent": "travel_assistant"}


@app.post("/api/search-flights")
def api_search_flights(
    departure_city: str,
    arrival_city: str,
    departure_date: str,
):
    """Search flights API endpoint."""
    logger.info(f"Searching flights: {departure_city} to {arrival_city} on {departure_date}")
    result = search_flights(departure_city, arrival_city, departure_date)
    logger.debug(f"Flight search result: {result}")
    return {"result": result}


@app.post("/api/check-prices")
def api_check_prices(
    flight_id: int,
):
    """Check prices API endpoint."""
    logger.info(f"Checking prices for flight_id: {flight_id}")
    result = check_prices(flight_id)
    logger.debug(f"Price check result: {result}")
    return {"result": result}


@app.get("/api/recommendations")
def api_recommendations(
    max_price: float,
    preferred_airlines: str | None = None,
):
    """Get recommendations API endpoint."""
    logger.info(
        f"Getting recommendations: max_price={max_price}, preferred_airlines={preferred_airlines}"
    )
    airlines = preferred_airlines.split(",") if preferred_airlines else None
    result = get_recommendations(max_price, airlines)
    logger.debug(f"Recommendations result: {result}")
    return {"result": result}


@app.post("/api/create-trip-plan")
def api_create_trip_plan(
    departure_city: str,
    arrival_city: str,
    departure_date: str,
    return_date: str | None = None,
    budget: float | None = None,
):
    """Create trip plan API endpoint."""
    logger.info(
        f"Creating trip plan: {departure_city} to {arrival_city}, dates: {departure_date} - {return_date}"
    )
    logger.debug(f"Budget: {budget}")
    result = create_trip_plan(departure_city, arrival_city, departure_date, return_date, budget)
    logger.debug(f"Trip plan result: {result}")
    return {"result": result}


@app.post("/api/discover-agents")
async def api_discover_agents(query: str):
    """Discover agents through registry using semantic search."""
    logger.info(f"Agent discovery request: query='{query}'")

    from dependencies import get_registry_client

    registry_client = get_registry_client()
    if not registry_client:
        return {"error": "Discovery not configured"}

    try:
        agents = await registry_client.discover_by_semantic_search(
            query=query,
            max_results=5,
        )
        return {
            "query": query,
            "agents_found": len(agents),
            "agents": [agent.model_dump() for agent in agents],
        }
    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=True)
        return {"error": "An internal error occurred during agent discovery"}


app.mount("/", a2a_server.to_fastapi_app())


def main() -> None:
    """Main entry point for the application."""
    logger.info("Starting Travel Assistant Agent server")
    uvicorn.run(app, host=env_settings.host, port=env_settings.port)


if __name__ == "__main__":
    main()
