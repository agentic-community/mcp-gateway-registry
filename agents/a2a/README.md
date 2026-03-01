# Travel Booking Agents

Two AI agents built with AWS Bedrock AgentCore and the Strands framework for flight search and booking.

## Agents

**Travel Assistant Agent** (`travel_assistant_agent`)
- Searches for available flights between cities
- Provides flight recommendations based on price and preferences
- Returns detailed flight information (times, prices, airlines)
- **Discovers other agents** through the MCP Gateway Registry and dynamically adds them as tools
- [Full specification](https://github.com/agentic-community/mcp-gateway-registry/issues/196)

**Flight Booking Agent** (`flight_booking_agent`)
- Checks flight availability and seat counts
- Creates flight reservations
- Manages booking database
- [Full specification](https://github.com/agentic-community/mcp-gateway-registry/issues/197)

## Deployment Options

### Local Docker Container

Run agents locally with full FastAPI server including custom API endpoints.

**Prerequisites:**
- Docker and Docker Compose
- AWS credentials configured (via AWS_PROFILE, EC2 IAM role, or ~/.aws/credentials)
- `uv sync --extra dev` to install main dependencies and development ones

**Deploy:**
```bash
# 1. Get AWS credentials (for Isengard users)
isengard credentials --account YOUR_ACCOUNT --role YOUR_ROLE --export

# This exports: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
# Docker will automatically pick these up from your environment

# 2. Deploy (auto-detects your system architecture)
# From repo root:
agents/a2a/deploy_local.sh

# Or from agents/a2a directory:
./deploy_local.sh
```

**Architecture Support:**
The script automatically detects your system architecture:
- **Intel/AMD Macs and Linux:** Uses `docker-compose.local.yml` (x86_64)
- **Apple Silicon Macs:** Uses `docker-compose.arm.yml` (ARM64)

To override auto-detection:
```bash
# Force ARM64 (Apple Silicon) - from repo root
agents/a2a/deploy_local.sh --arm64

# Force x86_64 (Intel/AMD) - from repo root
agents/a2a/deploy_local.sh --x86_64

# Show help - from repo root
agents/a2a/deploy_local.sh --help

# Or from agents/a2a directory:
./deploy_local.sh --arm64
./deploy_local.sh --x86_64
./deploy_local.sh --help
```

**Endpoints:**
- Travel Assistant: `http://localhost:9001`
- Flight Booking: `http://localhost:9002`
- Custom APIs: `/api/search-flights`, `/api/recommendations`, `/api/check-availability`
- Health check: `/ping`

### AgentCore Runtime (AWS)

Deploy agents to AWS managed infrastructure with automatic scaling.

**Prerequisites:**
- AWS credentials configured (via AWS_PROFILE, EC2 IAM role, or ~/.aws/credentials)
- AgentCore CLI: `pip install bedrock-agentcore-starter-toolkit`

**Deploy:**
```bash
# Configure AWS credentials (one of these methods)
export AWS_PROFILE=your_profile_name

# Or use EC2 IAM role (no export needed)

# Then deploy
./deploy_live.sh
```

**Note:** The deployment script automatically builds ARM64 images for AgentCore Runtime compatibility. The `docker-compose.arm.yml` file defines the ARM64 build targets used during deployment.

**Access:**
- Agents accessible via A2A protocol only
- ARNs shown in deployment output
- CloudWatch logs for monitoring

## Testing

### Agent Card Endpoint (Local)

Test the agent card endpoint locally to verify agent metadata. The script retrieves and displays agent card information, and saves JSON files locally for reference.

**Run the check:**

```bash
# From repo root
agents/a2a/test/check_agent_cards.sh

# Or from agents/a2a directory
cd agents/a2a
./test/check_agent_cards.sh
```

**Output Files:**

Agent cards are saved to the `agents/a2a/test/` directory:
- `travel_assistant_agent_card.json` - Travel Assistant agent metadata
- `flight_booking_agent_card.json` - Flight Booking agent metadata

These files contain:
- Agent name and description
- Available tools and capabilities
- API endpoints and methods
- Input/output schemas

> **Next Steps:** For remote testing of deployed agents, consider using the [A2A Inspector](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore-testing.html) to interact with and debug your AgentCore Runtime deployments.

### Agent and API Tests

Run comprehensive tests against local or live deployments to verify agent functionality:

**Test Coverage:**
- **Health Checks:** Verify agents are responsive via `/ping` endpoint
- **Agent Communication (A2A Protocol):** Send natural language requests to agents and verify responses
  - Travel Assistant: Flight search queries
  - Flight Booking: Availability checks and reservations
- **Direct API Endpoints:** Test custom FastAPI endpoints (local only)
  - `/api/search-flights` - Flight search with parameters
  - `/api/recommendations` - Price-based recommendations
  - `/api/check-availability` - Seat availability checks
- **Response Validation:** Verify response structure and content accuracy

**Run Tests:**

```bash
# Test local Docker containers (from repo root)
uv run python agents/a2a/test/simple_agents_test.py --endpoint local

# Test local Docker containers (from agents/a2a directory)
cd agents/a2a
uv run python test/simple_agents_test.py --endpoint local

# Test AgentCore Runtime (from repo root)
uv run python agents/a2a/test/simple_agents_test.py --endpoint live
```

**Debug Mode:**

For detailed request/response tracing, use the `--debug` flag:

```bash
# View full JSON-RPC payloads, response bodies, and timing (from repo root)
uv run python agents/a2a/test/simple_agents_test.py --endpoint local --debug

# Or from agents/a2a directory:
uv run python test/simple_agents_test.py --endpoint local --debug
```

This displays:
- Complete JSON-RPC request payloads
- Full agent response bodies with artifacts
- Response timing and HTTP status codes
- Streaming data for agent reasoning

## Deployment Scripts

### deploy_local.sh
Deploys and starts the agents locally in Docker containers.

**Features:**
- Auto-detects your system architecture (x86_64 or ARM64)
- Validates AWS credentials using the credential chain
- Removes and recreates containers and volumes for a clean deployment
- Builds Docker images locally before starting

**Usage (from repo root):**
```bash
agents/a2a/deploy_local.sh                 # Auto-detect architecture
agents/a2a/deploy_local.sh --arm64         # Force ARM64 (Apple Silicon)
agents/a2a/deploy_local.sh --x86_64        # Force x86_64 (Intel/AMD)
agents/a2a/deploy_local.sh --help          # Show usage options
```

**Usage (from agents/a2a directory):**
```bash
./deploy_local.sh                 # Auto-detect architecture
./deploy_local.sh --arm64         # Force ARM64 (Apple Silicon)
./deploy_local.sh --x86_64        # Force x86_64 (Intel/AMD)
./deploy_local.sh --help          # Show usage options
```

### shutdown_local.sh
Stops and removes all containers, networks, and volumes.

**Usage (from repo root):**
```bash
agents/a2a/shutdown_local.sh
```

**Usage (from agents/a2a directory):**
```bash
./shutdown_local.sh
```

This is useful when you want to completely clean up before redeploying or when done testing locally.

## Agent-to-Agent Discovery

The Travel Assistant Agent can discover and invoke other agents at runtime using the MCP Gateway Registry's semantic search API. This enables dynamic agent composition where agents find collaborators based on capabilities.

### Configuration

1. Copy the example environment file:
```bash
cp agents/a2a/.env.example agents/a2a/.env
```

2. Edit `agents/a2a/.env` and set your JWT token:
```bash
# Registry URL (default works for Docker network)
MCP_REGISTRY_URL=http://registry

# Paste a valid JWT token from the registry UI or API
REGISTRY_JWT_TOKEN=<your-jwt-token>
```

The agent uses this token to authenticate with the registry's semantic search API. The `deploy_local.sh` script automatically loads `.env` before starting containers.

### Prerequisites

- MCP Gateway Registry running (`docker-compose up -d`)
- Flight Booking Agent registered in the registry (via UI or CLI)
- Valid JWT token configured in `agents/a2a/.env`

### Discovery Tools

The Travel Assistant Agent provides three tools for agent discovery:

| Tool | Description |
|------|-------------|
| `discover_remote_agents` | Search registry for agents by natural language query |
| `view_cached_remote_agents` | List all discovered agents in cache |
| `invoke_remote_agent` | Send a message to a cached agent via A2A protocol |

### Testing Discovery

1. Start the MCP Gateway Registry and register the Flight Booking Agent

2. Deploy agents locally:
```bash
agents/a2a/deploy_local.sh
```

3. Run the test suite (includes discovery test):
```bash
uv run python agents/a2a/test/simple_agents_test.py --endpoint local --debug
```

4. View agent logs to see discovery in action:
```bash
docker logs -f travel-assistant-agent
```

You should see logs like:
```
RegistryDiscoveryClient initialized with direct JWT token for http://registry
Tool called: discover_remote_agents(query='book flights', max_results=5)
Found 1 agents
Cached agent: Flight Booking Agent (ID: /flight-booking-agent)
```

### Example Flow

```
User: "I need to book a flight from NYC to LA"

Travel Assistant Agent:
  1. discover_remote_agents("agent that can book flights")
     -> Returns: Flight Booking Agent (score: 0.85)

  2. invoke_remote_agent("/flight-booking-agent", "Book flight NYC to LA")
     -> Flight Booking Agent processes request
     -> Returns booking confirmation

  3. Returns combined response to user
```

---

## Key Differences

| Feature | Local Docker | AgentCore Runtime |
|---------|-------------|-------------------|
| A2A Protocol | ✅ | ✅ |
| Custom API Endpoints | ✅ | ❌ |
| Health Check `/ping` | ✅ | ❌ |
| Agent Discovery | ✅ | ✅ |
| Deployment | Docker Compose | AgentCore CLI |

**Note:** Custom FastAPI endpoints (like `/api/search-flights`) are only available in local Docker deployments. **AgentCore Runtime only wraps the container and exposes the standard A2A conversational interface.**
