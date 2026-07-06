<div align="center">
<img src="docs/img/mcp_gateway_horizontal_white_logo.png" alt="MCP Gateway & Registry Logo" width="100%">

**Unified Agent & MCP Server Registry – Gateway for AI Development Tools**

[![GitHub stars](https://img.shields.io/github/stars/agentic-community/mcp-gateway-registry?style=flat&logo=github)](https://github.com/agentic-community/mcp-gateway-registry/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/agentic-community/mcp-gateway-registry?style=flat&logo=github)](https://github.com/agentic-community/mcp-gateway-registry/network)
[![GitHub issues](https://img.shields.io/github/issues/agentic-community/mcp-gateway-registry?style=flat&logo=github)](https://github.com/agentic-community/mcp-gateway-registry/issues)
[![License](https://img.shields.io/github/license/agentic-community/mcp-gateway-registry?style=flat)](https://github.com/agentic-community/mcp-gateway-registry/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/agentic-community/mcp-gateway-registry?style=flat&logo=github)](https://github.com/agentic-community/mcp-gateway-registry/releases)

[🚀 Get Running Now](#option-a-pre-built-images-instant-setup) | [macOS Setup Skill](.claude/skills/macos-setup/SKILL.md) | [AWS Workshop Studio](https://catalog.us-east-1.prod.workshops.aws/workshops/0c3265a6-1a4a-467b-ae56-e4d019184b0e/en-US) | [AWS Deployment](terraform/aws-ecs/README.md) | [Quick Start](#quick-start) | [Documentation](docs/) | [Community](#community)

**Demo Videos:** 📺 [All Demo Videos](docs/demo-videos.md) | 🎥 [AWS Show & Tell](https://www.youtube.com/watch?v=dk0qVukHLGU) | ⭐ [MCP Registry CLI Demo](https://github.com/user-attachments/assets/98200866-e8bd-4ac3-bad6-c6d42b261dbe) | [Full End-to-End Functionality](https://github.com/user-attachments/assets/5ffd8e81-8885-4412-a4d4-3339bbdba4fb) | [OAuth 3-Legged Authentication](https://github.com/user-attachments/assets/3c3a570b-29e6-4dd3-b213-4175884396cc) | [Dynamic Tool Discovery](https://github.com/user-attachments/assets/cee25b31-61e4-4089-918c-c3757f84518c) | [Agent Skills](https://github.com/user-attachments/assets/5d1f227a-25f8-480d-9ff9-acba2498844b) | [Virtual MCP Servers](https://app.vidcast.io/share/954e6296-f217-4559-8d86-88cec25af763) | [Slide Deck](docs/slides/mcp-gateway-registry-presentation.pdf)

</div>

---

## What is MCP Gateway & Registry?

The **MCP Gateway & Registry** is a single, governed control plane for every AI asset in your organization, from MCP servers and AI agents to skills and any custom asset your teams build.

It began as a gateway and registry for the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction): one secure entry point to many MCP servers, with centralized discovery and governance. As teams started registering agents, skills, and other assets alongside their servers, it grew into a general-purpose **AI asset registry**, while keeping the same gateway, access control, and audit model it started with.

Today it serves these core functions:

1. **Unified MCP Server Gateway** – Centralized access point for multiple MCP servers
2. **MCP Servers Registry** – Register, discover, and manage access to MCP servers with unified governance
3. **Agent Registry & A2A Communication Hub** – Agent registration, discovery, governance, and direct agent-to-agent communication through the [A2A (Agent-to-Agent) Protocol](https://a2a-protocol.org/latest/specification/)
4. **Skills Registry** – Register, version, and discover reusable AI skills (`SKILL.md`) with security scanning
5. **Custom Entity Registry** – Administrators define their own entity types with custom schemas, so the registry can hold *any* AI asset: n8n workflows, policies, prompt templates, model cards, datasets, evaluations, and more

Every asset type, built-in or custom, shares the same control plane: schema-driven registration, semantic and lexical search, discovery APIs, voting, registry cards, and one access-control and audit model. The platform also integrates with external registries such as Anthropic's MCP Registry (and more to come), giving you a single surface for tool access, agent orchestration, and agent-to-agent communication.

**Why unified?** Instead of managing hundreds of individual MCP server configurations, agent connections, and separate governance systems across your teams, this platform provides secure, governed access to every curated AI asset through one control plane.

**Transform this chaos:**
```
❌ AI agents require separate connections to each MCP server
❌ Each developer configures VS Code, Cursor, Claude Code individually
❌ Developers must install and manage MCP servers locally
❌ No standard authentication flow for enterprise tools
❌ Scattered API keys and credentials across tools
❌ No visibility into what tools teams are using
❌ Security risks from unmanaged tool sprawl
❌ No dynamic tool discovery for autonomous agents
❌ No curated tool catalog for multi-tenant environments
❌ A2A provides agent cards but no way for agents to discover other agents
❌ Maintaining separate MCP server and agent registries is a non-starter for governance
❌ Impossible to maintain unified policies across server and agent access
```

**Into this organized approach:**
```
✅ AI agents connect to one gateway, access multiple MCP servers
✅ Single configuration point for VS Code, Cursor, Claude Code
✅ Central IT manages cloud-hosted MCP infrastructure via streamable HTTP
✅ Developers use standard OAuth 2LO/3LO flows for enterprise MCP servers
✅ Centralized credential management with secure vault integration
✅ Complete visibility and audit trail for all tool usage
✅ Security features with governed tool access
✅ Dynamic tool discovery and invocation for autonomous workflows
✅ Registry provides discoverable, curated MCP servers for multi-tenant use
✅ Agents can discover and communicate with other agents through unified Agent Registry
✅ Single control plane for both MCP servers and agent governance
✅ Unified policies and audit trails for both server and agent access
```

```
┌─────────────────────────────────────┐     ┌──────────────────────────────────────────────────────┐
│          BEFORE: Chaos              │     │    AFTER: MCP Gateway & Registry                     │
├─────────────────────────────────────┤     ├──────────────────────────────────────────────────────┤
│                                     │     │                                                      │
│  Developer 1 ──┬──► MCP Server A    │     │  Developer 1 ──┐                  ┌─ MCP Server A    │
│                ├──► MCP Server B    │     │                │                  ├─ MCP Server B    │
│                └──► MCP Server C    │     │  Developer 2 ──┼──► MCP Gateway   │                  │
│                                     │     │                │    & Registry ───┼─ MCP Server C    │
│  Developer 2 ──┬──► MCP Server A    │ ──► │  AI Agent 1 ───┘         │        │                  │
│                ├──► MCP Server D    │     │                          │        ├─ AI Agent 1      │
│                └──► MCP Server E    │     │  AI Agent 2 ──────────────┤        ├─ AI Agent 2     │
│                                     │     │                          │        │                  │
│  AI Agent 1 ───┬──► MCP Server B    │     │  AI Agent 3 ──────────────┘        └─ AI Agent 3     │
│                ├──► MCP Server C    │     │                                                      │
│                └──► MCP Server F    │     │              Single Connection Point                 │
│                                     │     │                                                      │
│  ❌ Multiple connections per user  │     │         ✅ One gateway for all                      │
│  ❌ No centralized control         │     │         ✅ Unified server & agent access            │
│  ❌ Credential sprawl              │     │         ✅ Unified governance & audit trails        │
└─────────────────────────────────────┘     └──────────────────────────────────────────────────────┘
```

> **Note on Agent-to-Agent Communication:** AI Agents discover other AI Agents through the unified Agent Registry and communicate with them **directly** (peer-to-peer) without routing through the MCP Gateway. The Registry handles discovery, authentication, and access control, while agents maintain direct connections for efficient, low-latency communication.

## Unified Agent & Server Registry

This platform serves as a comprehensive, unified registry supporting:

- ✅ **MCP Server Registration & Discovery** – Register, discover, and manage access to MCP servers
- ✅ **AI Agent Registration & Discovery** – Register agents and enable them to discover other agents
- ✅ **Agent-to-Agent (A2A) Communication** – Direct agent-to-agent communication patterns using the A2A protocol
- ✅ **Multi-Protocol Support** – Support for various agent communication protocols and patterns
- ✅ **Unified Governance** – Single policy and access control system for both agents and servers
- ✅ **Cross-Protocol Agent Discovery** – Agents can discover each other regardless of implementation
- ✅ **Integrated External Registries** – Connect with Anthropic's MCP Registry and other external sources
- ✅ **Agent Cards & Metadata** – Rich metadata for agent capabilities, skills, and authentication schemes

Key distinction: **Unlike separate point solutions, this unified registry eliminates the need to maintain separate MCP server and agent systems**, providing a single control plane for agent orchestration, MCP server access, and agent-to-agent communication.

## MCP Servers, Agents and Skills Registry

Watch how MCP Servers, A2A Agents, and External Registries work together for dynamic tool discovery:

https://github.com/user-attachments/assets/97c640db-f78b-4a6c-9662-894f975f66e2

---

## MCP Tools in Action

[View MCP Tools Demo](docs/img/MCP_tools.gif)

---

## MCP Registry CLI

Interactive terminal interface for chatting with AI models and discovering MCP tools in natural language. Talk to the registry using a Claude Code-like conversational interface with real-time token status, cost tracking, and AI model selection.

<div align="center">
<img src="docs/img/mcp-registry-cli.png" alt="MCP Registry CLI Screenshot" width="800"/>
</div>

**Quick Start:** `registry --url https://mcpgateway.ddns.net` | [Full Guide](docs/mcp-registry-cli.md)

---

## A2A Agents - Example Implementations

The registry includes two example A2A agents that demonstrate how both human developers and autonomous AI agents can discover, register, and use agents through the unified Agent Registry. Agents can programmatically discover other agents via semantic search and use them through the A2A protocol, enabling dynamic agent composition and autonomous agent orchestration.

### Example Agents

| Agent | Path | Skills |
|-------|------|--------|
| **Travel Assistant Agent** | `/travel-assistant-agent` | Flight search, pricing checks, recommendations, trip planning |
| **Flight Booking Agent** | `/flight-booking-agent` | Availability checks, flight reservations, payments, reservation management |

### Agent Discovery

**View in Registry UI:**
Open the registry and navigate to the **A2A Agents** tab to browse registered agents with their full metadata, capabilities, and skills.

**Search via CLI:**
Developers can search for agents by natural language description:

```bash
# Search for agents that can help book a trip
cli/agent_mgmt.sh search "need an agent to book a trip"
```

**Example Output:**
```
Found 4 agent(s) matching 'need an agent to book a trip':
--------------------------------------------------------------------------------------------------------------
Agent Name                               | Path                      | Score
--------------------------------------------------------------------------------------------------------------
Travel Assistant Agent                   | /travel-assistant-agent   |  0.8610
Flight Booking Agent                     | /flight-booking-agent     |  1.2134
--------------------------------------------------------------------------------------------------------------
```

### Agent-to-Agent Discovery API

The registry provides a **semantic search API** that agents can use as a tool to discover other A2A agents at runtime. This API enables dynamic agent composition where agents find collaborators based on capabilities rather than hardcoded references.

**Discovery API Endpoint:**
```
POST /api/agents/discover/semantic?query=<natural-language-query>&max_results=5
Authorization: Bearer <jwt-token>
```

**Response includes:**
- Agent name, description, and endpoint URL
- Agent card metadata with skills and capabilities
- Relevance score for ranking matches
- Trust level and visibility settings

**How agents use it:**
1. An agent calls the registry's semantic search API with a natural language query (e.g., "agent that can book flights")
2. The registry returns matching agents with their endpoint URLs and full agent card metadata
3. The agent uses the agent card to understand capabilities and invokes the discovered agent via A2A protocol

**Example - Travel Assistant discovering and invoking Flight Booking Agent:**
```
User: "I need to book a flight from NYC to LA"

Travel Assistant:
  1. Calls registry API: POST /api/agents/discover/semantic?query="book flights"
  2. Registry returns Flight Booking Agent with endpoint URL and agent card
  3. Uses agent card to understand capabilities, then sends A2A message to Flight Booking Agent
  4. Returns booking confirmation to user
```

This pattern enables agents to dynamically extend their capabilities by discovering specialized agents for tasks they cannot handle directly.

**Agent Cards:** View the agent card metadata at [agents/a2a/test/](agents/a2a/test/) to see the complete agent definitions including skills, protocols, and capabilities.

For complete agent deployment and testing documentation, see [agents/a2a/README.md](agents/a2a/README.md).

---

## Core Use Cases

### AI Agent & Coding Assistant Governance
Provide both autonomous AI agents and human developers with secure access to approved tools through AI coding assistants (VS Code, Cursor, Claude Code) while maintaining IT oversight and compliance.

### Enterprise Security & Compliance
Centralized authentication, fine-grained permissions, and comprehensive audit trails for SOX/GDPR compliance pathways across both human and AI agent access patterns.

### Dynamic Tool Discovery
AI agents can autonomously discover and execute specialized tools beyond their initial capabilities using intelligent semantic search, while developers get guided tool discovery through their coding assistants.

### Unified Access Gateway
Single gateway supporting both autonomous AI agents (machine-to-machine) and AI coding assistants (human-guided) with consistent authentication and tool access patterns.

---

## Architecture

The MCP Gateway & Registry provides a unified platform for both autonomous AI agents and AI coding assistants to access enterprise-curated tools through a centralized gateway with comprehensive authentication and governance.

The solution consists of four core components: an **Nginx reverse proxy** that handles TLS termination, auth validation, and dynamic routing to backend MCP servers; a **Registry API** (FastAPI) that manages the catalog of servers, agents, and skills with semantic search; an **Auth Server** that integrates with enterprise identity providers (Keycloak, Entra ID, Okta, Auth0, Cognito, PingFederate) for OAuth2/OIDC flows; and a **MongoDB/DocumentDB data layer** that stores all configuration, embeddings, sessions, and audit records.

For detailed architecture diagrams at multiple levels of depth (executive overview through security threat model), see [Architecture Diagrams](docs/architecture-diagrams.md).

```mermaid
flowchart TB
    subgraph Human_Users["Human Users"]
        User1["Human User 1"]
        User2["Human User 2"]
        UserN["Human User N"]
    end

    subgraph AI_Agents["AI Agents"]
        Agent1["AI Agent 1"]
        Agent2["AI Agent 2"]
        Agent3["AI Agent 3"]
        AgentN["AI Agent N"]
    end

    subgraph EC2_Gateway["<b>MCP Gateway & Registry</b> (Amazon EC2 Instance)"]
        subgraph NGINX["NGINX Reverse Proxy"]
            RP["Reverse Proxy Router"]
        end

        subgraph AuthRegistry["Authentication & Registry Services"]
            AuthServer["Auth Server<br/>(Dual Auth)"]
            Registry["Registry<br/>Web UI"]
            RegistryMCP["Registry<br/>MCP Server"]
        end

        subgraph LocalMCPServers["Local MCP Servers"]
            MCP_Local1["MCP Server 1"]
            MCP_Local2["MCP Server 2"]
        end
    end

    %% Identity Provider
    IdP[Identity Provider<br/>Keycloak/Cognito]

    subgraph EKS_Cluster["Amazon EKS/EC2 Cluster"]
        MCP_EKS1["MCP Server 3"]
        MCP_EKS2["MCP Server 4"]
    end

    subgraph APIGW_Lambda["Amazon API Gateway + AWS Lambda"]
        API_GW["Amazon API Gateway"]
        Lambda1["AWS Lambda Function 1"]
        Lambda2["AWS Lambda Function 2"]
    end

    subgraph External_Systems["External Data Sources & APIs"]
        DB1[(Database 1)]
        DB2[(Database 2)]
        API1["External API 1"]
        API2["External API 2"]
        API3["External API 3"]
    end

    %% Connections from Human Users
    User1 -->|Web Browser<br>Authentication| IdP
    User2 -->|Web Browser<br>Authentication| IdP
    UserN -->|Web Browser<br>Authentication| IdP
    User1 -->|Web Browser<br>HTTPS| Registry
    User2 -->|Web Browser<br>HTTPS| Registry
    UserN -->|Web Browser<br>HTTPS| Registry

    %% Connections from Agents to Gateway
    Agent1 -->|MCP Protocol<br>SSE with Auth| RP
    Agent2 -->|MCP Protocol<br>SSE with Auth| RP
    Agent3 -->|MCP Protocol<br>Streamable HTTP with Auth| RP
    AgentN -->|MCP Protocol<br>Streamable HTTP with Auth| RP

    %% Auth flow connections
    RP -->|Auth validation| AuthServer
    AuthServer -.->|Validate credentials| IdP
    Registry -.->|User authentication| IdP
    RP -->|Tool discovery| RegistryMCP
    RP -->|Web UI access| Registry

    %% Connections from Gateway to MCP Servers
    RP -->|SSE| MCP_Local1
    RP -->|SSE| MCP_Local2
    RP -->|SSE| MCP_EKS1
    RP -->|SSE| MCP_EKS2
    RP -->|Streamable HTTP| API_GW

    %% Connections within API GW + Lambda
    API_GW --> Lambda1
    API_GW --> Lambda2

    %% Connections to External Systems
    MCP_Local1 -->|Tool Connection| DB1
    MCP_Local2 -->|Tool Connection| DB2
    MCP_EKS1 -->|Tool Connection| API1
    MCP_EKS2 -->|Tool Connection| API2
    Lambda1 -->|Tool Connection| API3

    %% Style definitions
    classDef user fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef agent fill:#e1f5fe,stroke:#29b6f6,stroke-width:2px
    classDef gateway fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
    classDef nginx fill:#f3e5f5,stroke:#ab47bc,stroke-width:2px
    classDef mcpServer fill:#fff3e0,stroke:#ffa726,stroke-width:2px
    classDef eks fill:#ede7f6,stroke:#7e57c2,stroke-width:2px
    classDef apiGw fill:#fce4ec,stroke:#ec407a,stroke-width:2px
    classDef lambda fill:#ffebee,stroke:#ef5350,stroke-width:2px
    classDef dataSource fill:#e3f2fd,stroke:#2196f3,stroke-width:2px

    %% Apply styles
    class User1,User2,UserN user
    class Agent1,Agent2,Agent3,AgentN agent
    class EC2_Gateway,NGINX gateway
    class RP nginx
    class AuthServer,Registry,RegistryMCP gateway
    class IdP apiGw
    class MCP_Local1,MCP_Local2 mcpServer
    class EKS_Cluster,MCP_EKS1,MCP_EKS2 eks
    class API_GW apiGw
    class Lambda1,Lambda2 lambda
    class DB1,DB2,API1,API2,API3 dataSource
```

**Key Architectural Benefits:**
- **Unified Gateway**: Single point of access for both AI agents and human developers through coding assistants
- **Dual Authentication**: Supports both human user authentication and machine-to-machine agent authentication
- **Scalable Infrastructure**: Nginx reverse proxy with horizontal scaling capabilities
- **Multiple Transports**: SSE and Streamable HTTP support for different client requirements

---

## Key Advantages

### **Security Features**
- OAuth 2.0/3.0 compliance with IdP integration
- Fine-grained access control at tool and method level
- Zero-trust network architecture
- Complete audit trails and comprehensive analytics for compliance

### **AI Agent & Developer Experience**
- Single configuration works across autonomous AI agents and AI coding assistants (VS Code, Cursor, Claude Code, Cline)
- Dynamic tool discovery with natural language queries for both agents and humans
- Instant onboarding for new team members and AI agent deployments
- Unified governance for both AI agents and human developers

### **Deployment Features**
- Container-native (Docker/Kubernetes)
- Real-time health monitoring and alerting
- Dual authentication supporting both human and machine authentication

---
## Quick Start

There are 4 options for setting up the MCP Gateway & Registry:

- **Option A: AI-Assisted macOS Setup** — The absolute fastest way to get started on macOS. Ask your AI coding assistant to use the [macOS Setup Skill](.claude/skills/macos-setup/SKILL.md) for fully automated one-command setup. Perfect for experimentation.
- **Option B: Pre-built Images** — Fast setup using pre-built Docker or Podman containers. Recommended for most users.
- **Option C: Podman (Rootless)** — Detailed Podman-specific instructions for macOS and rootless Linux environments.
- **Option D: Build from Source** — Full source build for customization or development.

### Option A: AI-Assisted macOS Setup (Fastest)

**The easiest way to get started on macOS.** Simply ask Claude Code or your AI coding assistant:

> "Use the macOS setup skill to install and configure the MCP Gateway & Registry"

The [macOS Setup Skill](.claude/skills/macos-setup/SKILL.md) will automatically:
- ✅ Clone the repository and install all dependencies (Homebrew, Python, UV, Docker, Node.js)
- ✅ Configure and start MongoDB with replica set
- ✅ Set up and initialize Keycloak with admin user
- ✅ Start the registry and auth server
- ✅ Register the Cloudflare MCP docs server
- ✅ Verify the complete stack is operational

**Perfect for:** Single-developer experimentation, quick demos, hands-on exploration

**What you need:** macOS with an AI coding assistant (Claude Code, Cursor, etc.)

**Clean up:** When done, ask your AI assistant to "teardown the MCP Gateway setup" for complete removal.

*Note: ECS/EKS deployment skill coming very soon for production deployments.*

---

### Option B: Pre-built Images (Instant Setup)

Get running with pre-built Docker containers in minutes. This is the recommended approach for most users.

```bash
# Clone and configure
git clone https://github.com/agentic-community/mcp-gateway-registry.git
cd mcp-gateway-registry
cp .env.example .env

# Edit .env with your passwords (KEYCLOAK_ADMIN_PASSWORD, etc.)
nano .env

# Deploy with pre-built images (pulled from Amazon ECR Public by default)
./build_and_run.sh --prebuilt

# Access the Registry UI (served by nginx on port 80)
open http://localhost  # macOS
# xdg-open http://localhost  # Linux
```

**[Complete Quick Start Guide](docs/quickstart.md)** - Full step-by-step instructions including:
- Prerequisites installation (Docker, Python, UV)
- Environment configuration
- MongoDB and Keycloak initialization
- User and service account setup
- Server and agent registration
- Testing the gateway functionality

**Benefits:** No build time | No Node.js required | No frontend compilation | Consistent tested images

---

### Option C: Podman (Rootless Container Deployment)

**Perfect for macOS and rootless Linux environments**

Podman provides rootless container execution without requiring privileged ports, making it ideal for:
- **macOS** users with Podman Desktop
- **Linux** users preferring rootless containers
- **Development** environments where Docker daemon isn't available

**Quick Podman Setup (macOS non-Apple Silicon):**

```bash
# Install Podman Desktop
brew install podman-desktop
# OR download from: https://podman-desktop.io/
```

Inside Podman Desktop, go to Preferences > Podman Machine and create a new machine with at least 4 CPUs and 8GB RAM. Alternatively, see more detailed [Podman installation guide](docs/installation.md#podman-installation) for instructions on setting this up on CLI.

```bash
# Initialize Podman machine
podman machine init
podman machine start

# Verify installation
podman --version
podman compose version

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

**Deploy with Podman** see full Podman setup instructions (downloading, installing, and initializing a first Podman container, as well as troubleshooting) in our [Installation Guide](docs/installation.md#podman-installation).

**Build with Podman:**

```bash
# Auto-detect (will use Podman if Docker not available)
./build_and_run.sh --prebuilt

# Explicit Podman mode (only non-Apple Silicon)
./build_and_run.sh --prebuilt --podman

# Access registry at non-privileged ports
# On macOS:
open http://localhost:8080
# On Linux: xdg-open http://localhost:8080
```

> Note: **Apple Silicon (M1/M2/M3)?** Don't use `--prebuilt` with Podman on ARM64. This will cause a "proxy already running" error. See [Podman on Apple Silicon Guide](docs/podman-apple-silicon.md).

```bash
# To run on Apple Silicon Macs:
./build_and_run.sh --podman
```

**Key Differences vs. Docker:**
- No root/sudo required
- Works on macOS without privileged port access
- HTTP port: `8080` (instead of `80`)
- HTTPS port: `8443` (instead of `443`)
- All other service ports unchanged

For detailed Podman setup instructions, see [Installation Guide](docs/installation.md#podman-installation) and [macOS Setup Guide](docs/macos-setup-guide.md#podman-deployment).

### Option D: Build from Source

**New to MCP Gateway?** Start with our [Complete Setup Guide](docs/complete-setup-guide.md) for detailed step-by-step instructions from scratch on AWS EC2.

**Running on macOS?** See our [macOS Setup Guide](docs/macos-setup-guide.md) for platform-specific instructions and optimizations.

### Testing & Integration Options

**Test Suite:**
The project includes comprehensive automated testing with pytest:

```bash
# Run all tests
make test

# Run only unit tests (fast)
make test-unit

# Run with coverage report
make test-coverage

# Run specific test categories
uv run pytest -m unit           # Unit tests only
uv run pytest -m integration    # Integration tests
uv run pytest -m "not slow"     # Skip slow tests
```

**Test Structure:**
- **Unit Tests** (`tests/unit/`) - Fast, isolated component tests
- **Integration Tests** (`tests/integration/`) - Component interaction tests
- **E2E Tests** (`tests/integration/test_e2e_workflows.py`) - Complete workflow tests

**Python Agent:**
- `agents/agent.py` - Full-featured Python agent with advanced AI capabilities

**Testing Documentation:**
- [Testing Guide](docs/testing/README.md) - Comprehensive testing documentation
- [Writing Tests](docs/testing/WRITING_TESTS.md) - How to write effective tests
- [Test Maintenance](docs/testing/MAINTENANCE.md) - Maintaining test suite health

**Pre-commit Hooks:**
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

**Next Steps:** [Complete Installation Guide](docs/installation.md) | [Authentication Setup](docs/auth.md) | [AI Assistant Integration](docs/ai-coding-assistants-setup.md)

---

## Enterprise Features

### AI Agents & Coding Assistants Integration

Transform how both autonomous AI agents and development teams access enterprise tools with centralized governance:

<table>
<tr>
<td width="50%">
<img src="docs/img/roo.png" alt="Roo Code MCP Configuration" />
<p><em>Enterprise-curated MCP servers accessible through unified gateway</em></p>
</td>
<td width="50%">
<img src="docs/img/roo_agent.png" alt="Roo Code Agent in Action" />
<p><em>AI assistants executing approved enterprise tools with governance</em></p>
</td>
</tr>
<tr>
<td colspan="2">

### Observability

Comprehensive real-time metrics and monitoring through Grafana dashboards with dual-path storage: SQLite for detailed historical analysis and OpenTelemetry (OTEL) export for integration with Prometheus, CloudWatch, Datadog, and other monitoring platforms. Track authentication events, tool executions, discovery queries, and system performance metrics. [Learn more](docs/OBSERVABILITY.md)

**Log files**: registry, auth-server, and mcpgw write JSONL-formatted log files to `/var/log/containers/ai-registry/` on the Docker host, making them easy to ingest with Splunk Universal Forwarders or similar log shippers. See the [Logging Standard](docs/logging-standard.md) for the schema and Splunk props.conf recipe.

<img src="docs/img/dashboard.png" alt="Grafana Metrics Dashboard" />
<p><em>Real-time metrics and observability dashboard tracking server health, tool usage, and authentication events</em></p>
</td>
</tr>
</table>

### Anthropic MCP Registry Integration

Seamlessly integrate with Anthropic's official MCP Registry to import and access curated MCP servers through your gateway:

- **Import Servers**: Select and import desired servers from Anthropic's registry with a single command
- **Unified Access**: Access imported servers through your gateway with centralized authentication and governance
- **API Compatibility**: Full support for Anthropic's Registry REST API specification - point your Anthropic API clients to this registry to discover available servers

<img src="docs/img/registry_w_a.png" alt="Anthropic Registry Integration" />
<p><em>Import and access curated MCP servers from Anthropic's official registry</em></p>

[Import Guide](docs/anthropic-registry-import.md) | [Registry API Documentation](docs/anthropic_registry_api.md)

### Federation - External Registry Integration

**Unified Multi-Registry Access:**
- **Anthropic MCP Registry** - Import curated MCP servers with purple `ANTHROPIC` visual tags
- **Workday ASOR** - Import AI agents from Agent System of Record with orange `ASOR` visual tags
- **Automatic Sync** - Scheduled synchronization with external registries
- **Visual Identification** - Clear visual tags distinguish federation sources in the UI
- **Centralized Management** - Single control plane for all federated servers and agents

**Quick Setup:**
```bash
# Configure federation sources
echo 'ASOR_ACCESS_TOKEN=your_token' >> .env

# Update federation.json with your sources
# Restart services
./build_and_run.sh
```

[**📖 Complete Federation Guide**](docs/federation.md) - Environment setup, authentication, configuration, and troubleshooting

### Security Scanning

**Integrated Vulnerability Detection:**
- **Automated Security Scanning** - Integrated vulnerability scanning for MCP servers using [Cisco AI Defence MCP Scanner](https://github.com/cisco-ai-defense/mcp-scanner), with automatic scans during registration and support for periodic registry-wide scans
- **Detailed Security Reports** - Comprehensive markdown reports with vulnerability details, severity assessments, and remediation recommendations
- **Automatic Protection** - Servers with security issues are automatically disabled with security-pending status to protect your infrastructure
- **Compliance Ready** - Security audit trails and vulnerability tracking for enterprise compliance requirements

### Authentication & Authorization

**Multiple Identity Modes:**
- **Machine-to-Machine (M2M)** - For autonomous AI agents and automated systems
- **Three-Legged OAuth (3LO)** - For external service integration (Atlassian, Google, GitHub)
- **Session-Based** - For human developers using AI coding assistants and web interface

**Supported Identity Providers:** Keycloak, Microsoft Entra ID, Okta, Auth0, Amazon Cognito, and any OAuth 2.0 compatible provider. [Learn more](docs/auth.md)

**Fine-Grained Permissions:** Tool-level, method-level, team-based, and temporary access controls. [Learn more](docs/scopes.md)

### Deployment Options

**Cloud Platforms:** Amazon EC2, Amazon EKS

---

## Telemetry

The registry collects **anonymous, non-sensitive** usage telemetry to help us understand adoption patterns and improve the product. Both tiers are **opt-out** and **on by default**.

**What is sent (Tier 1 -- startup ping):** Registry version, Python version, OS, CPU architecture, cloud provider, cloud detection method (added in schema v3, see below), storage backend, auth provider, deployment mode, embeddings provider (`sentence-transformers` or `litellm`), and embeddings backend kind (a derived coarse category: `sentence-transformers`, `bedrock`, `openai`, `azure-openai`, `voyage`, `cohere`, `other`, or `unknown`). No IP addresses, hostnames, file paths, user data, or any PII.

**Also sent by default (Tier 2 -- daily heartbeat):** Aggregate counts (number of servers, agents, skills, peers), search backend, embeddings provider, embeddings backend kind, cloud detection method, and uptime. Same privacy guarantees as Tier 1. Disable heartbeat only: `MCP_TELEMETRY_OPT_OUT=1`.

**What is never sent:** The raw `EMBEDDINGS_MODEL_NAME`, `EMBEDDINGS_MODEL_DIMENSIONS`, credentials (API keys, AWS secrets), endpoints (`EMBEDDINGS_API_BASE`), and regions (`EMBEDDINGS_AWS_REGION`). Only the derived backend-kind category is reported. See [docs/TELEMETRY.md](docs/TELEMETRY.md) for the complete schema and privacy guarantees.

**Cloud detection method (schema v3, issue [#986](https://github.com/agentic-community/mcp-gateway-registry/issues/986)):** Cloud classification cascades through env vars -> DMI files -> `ECS_CONTAINER_METADATA_URI` -> Kubernetes node-name heuristics (`*.compute.internal`, `gke-*`, `aks-*`) -> live IMDS probe at 169.254.169.254 (AWS IMDSv2, Azure) and `metadata.google.internal` (GCP), with a 300 ms per-provider timeout and fail-silent behavior. First match wins. The `cloud_detection_method` field in the payload reports which tier fired (`env`, `dmi`, `ecs_meta`, `k8s_heuristic`, `imds`, or `unknown`) so rollups can diagnose why any instance classifies as `unknown`. IMDSv2 tokens have a 1-second TTL and are never logged; response bodies are discarded immediately. `httpx.Client(trust_env=False)` bypasses `HTTP_PROXY`/`HTTPS_PROXY` so link-local probes never route through a corporate proxy. Operators can see how their instance was classified via the System Health hover panel or `GET /api/system/telemetry-detection`. Disable IMDS probing while keeping env/DMI/ECS/k8s tiers with `MCP_TELEMETRY_IMDS_PROBE_DISABLED=1`. See [Cloud Provider Detection](docs/TELEMETRY.md#cloud-provider-detection) for the full cascade, the **EKS IMDS hop-limit gotcha** (the #1 operator trap), and zero-cost opt-in env vars.

> **Behavior change (post v1.0.18):** The daily heartbeat was previously opt-in (`MCP_TELEMETRY_OPT_IN=1`). It is now opt-out and sent by default. Since the heartbeat contains only aggregate counts (no PII), this aligns it with the startup ping behavior.

**To opt out completely:**

```bash
export MCP_TELEMETRY_DISABLED=1   # Disables both startup ping and heartbeat
```

**To disable heartbeat only (startup ping still sent):**

```bash
export MCP_TELEMETRY_OPT_OUT=1
```

**To disable IMDS probes only (keep env/DMI/ECS/k8s detection tiers):**

```bash
export MCP_TELEMETRY_IMDS_PROBE_DISABLED=1
```

All requests are HMAC-signed, rate-limited, and schema-validated. Telemetry is fail-silent and never impacts registry operation. Full details in the [Telemetry Documentation](docs/TELEMETRY.md).

---

## Deployments

### AWS Elastic Container Service (ECS)

<div align="center">
<img src="terraform/aws-ecs/img/MCP-Gateway-Registry-first-login.png" alt="MCP Gateway Registry on AWS ECS" width="800"/>
</div>

**Deployment configuration** on Amazon ECS Fargate with comprehensive enterprise features:

- **Multi-AZ Architecture** - Redundancy across multiple availability zones
- **Application Load Balancer** - HTTPS/SSL termination with automatic certificate management via ACM
- **Auto-scaling** - Dynamic scaling based on CPU and memory utilization
- **CloudWatch Integration** - Comprehensive monitoring, logging, and alerting
- **NAT Gateway HA** - Redundant NAT gateway configuration for secure outbound connectivity
- **Keycloak Integration** - Enterprise authentication with RDS Aurora PostgreSQL backend
- **EFS Shared Storage** - Persistent storage for models, logs, and configuration
- **Service Discovery** - AWS Cloud Map for service-to-service communication

**[Complete ECS Deployment Guide](terraform/aws-ecs/README.md)** - Step-by-step instructions for deploying the entire stack with Terraform.

### Amazon EKS (Kubernetes)

**Coming Soon** - Kubernetes deployment on Amazon EKS with Helm charts for container orchestration at scale.

---

## Documentation

| Getting Started | Enterprise Setup | Developer & Operations |
|------------------|-------------------|------------------------|
| [Complete Setup Guide](docs/complete-setup-guide.md)<br/>**NEW!** Step-by-step from scratch on AWS EC2 | [Authentication Guide](docs/auth.md)<br/>OAuth and identity provider integration | [AI Coding Assistant Registry Integration](docs/ai-coding-assistant-registry-integration.md)<br/>**NEW!** One-command DCR + OAuth setup for Claude Code and Claude.ai<br/>[AI Coding Assistants Setup](docs/ai-coding-assistants-setup.md)<br/>VS Code, Cursor, Claude Code static-token configuration |
| [Installation Guide](docs/installation.md)<br/>Complete setup instructions for EC2 and EKS | [AWS ECS Deployment](terraform/aws-ecs/README.md)<br/>Deployment guide for AWS ECS Fargate | [API Reference](docs/registry_api.md)<br/>Programmatic registry management |
| [Keycloak: Agent M2M & Operations](docs/keycloak-agent-m2m.md)<br/>Service accounts, ops, agent audit trails<br/>[Keycloak: MCP Client Guide](docs/keycloak-mcp-clients.md)<br/>DCR and OAuth flows for end-user MCP clients | [Token Refresh Service](docs/token-refresh-service.md)<br/>Automated token refresh and lifecycle management | [MCP Registry CLI](docs/mcp-registry-cli.md)<br/>Command-line client for registry management |
| [Configuration Reference](docs/configuration.md)<br/>Environment variables and settings | [Amazon Cognito Setup](docs/cognito.md)<br/>Step-by-step IdP configuration | [Observability Guide](docs/OBSERVABILITY.md)<br/>**NEW!** Metrics, monitoring, and OpenTelemetry setup |
| [Auth0 Integration](docs/auth0.md)<br/>Auth0 SSO with M2M support | [Okta Setup](docs/okta-setup.md)<br/>Okta IdP configuration | [Entra ID Setup](docs/entra-id-setup.md)<br/>Microsoft Entra ID integration |
| | [Anthropic Registry Import](docs/anthropic-registry-import.md)<br/>**NEW!** Import servers from Anthropic MCP Registry | [Federation Guide](docs/federation.md)<br/>External registry integration (Anthropic, ASOR) |
| | | [P2P Federation Guide](docs/federation-operational-guide.md)<br/>**NEW!** Peer-to-peer registry federation |
| | [Service Management](docs/service-management.md)<br/>Server lifecycle and operations | [Anthropic Registry API](docs/anthropic_registry_api.md)<br/>**NEW!** REST API compatibility |
| | | [Fine-Grained Access Control](docs/scopes.md)<br/>Permission management and security |
| | | [Dynamic Tool Discovery](docs/dynamic-tool-discovery.md)<br/>Autonomous agent capabilities |
| | | [Deployment Guide](docs/installation.md)<br/>Complete setup for deployment environments |
| | | [Troubleshooting Guide](docs/faq/index.md)<br/>Common issues and solutions |

---

## Community

### Get Involved

**Join the Discussion**
- [GitHub Discussions](https://github.com/agentic-community/mcp-gateway-registry/discussions) - Feature requests and general discussion
- [GitHub Issues](https://github.com/agentic-community/mcp-gateway-registry/issues) - Bug reports and feature requests

**Contributing**
- [Contributing Guide](CONTRIBUTING.md) - How to contribute code and documentation
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community guidelines
- [Security Policy](SECURITY.md) - Responsible disclosure process

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=agentic-community/mcp-gateway-registry&type=Date)](https://star-history.com/#agentic-community/mcp-gateway-registry&Date)

### Roadmap

Our development roadmap is organized into release milestones with clear deliverables and progress tracking:

| Milestone | Progress | Status | Key Issues |
|-----------|----------|--------|------------|
| **1.24.0** | 100% | Complete (May 2026) | Federation 2.0, skills phase 1. See [release-notes/1.24.0.md](docs/release-notes/1.24.0.md). |
| **1.24.1** | 100% (21/21) | Complete (May 2026) | [#1042 - Server-side OAuth session store](https://github.com/agentic-community/mcp-gateway-registry/issues/971), [#1061 - CVE-2026-4438 (glibc) remediation](https://github.com/agentic-community/mcp-gateway-registry/issues/1061), [#1044 - Nginx config race fix](https://github.com/agentic-community/mcp-gateway-registry/issues/1044), [#1056 - Operations runbooks](https://github.com/agentic-community/mcp-gateway-registry/issues/1056), [#1066 - HealthStatus enum sync](https://github.com/agentic-community/mcp-gateway-registry/issues/1066) and 16 more. See [release-notes/1.24.1.md](docs/release-notes/1.24.1.md). |
| **1.24.4** | 100% (25/25) | Complete (Jun 2026) | [#1173 - Custom entities & schemas](https://github.com/agentic-community/mcp-gateway-registry/issues/1172), [#1122 - Native OpenTelemetry metrics](https://github.com/agentic-community/mcp-gateway-registry/issues/1122), [#1157 - RRF hybrid search scoring](https://github.com/agentic-community/mcp-gateway-registry/issues/1157), [#1155 - search_registry tool](https://github.com/agentic-community/mcp-gateway-registry/issues/1155), [#1160 - per-entity vector search](https://github.com/agentic-community/mcp-gateway-registry/issues/1160), Amazon Cognito IdP, [#1128 - GitLab private repo support](https://github.com/agentic-community/mcp-gateway-registry/issues/1128), [#1127 - PingFederate IdP](https://github.com/agentic-community/mcp-gateway-registry/issues/1127) and more |
| **1.24.5 - Registry Hardening & Observability** | 55% (11/20) | In progress (Jun 2026) | [#1134 - Secrets Manager for ECS env vars](https://github.com/agentic-community/mcp-gateway-registry/issues/1134), [#1101 - MCP proxy session affinity](https://github.com/agentic-community/mcp-gateway-registry/issues/1101), [#1063 - CDK deployment](https://github.com/agentic-community/mcp-gateway-registry/issues/1063), [#862 - CDK deployment support](https://github.com/agentic-community/mcp-gateway-registry/issues/862), [#844 - Card dependency management](https://github.com/agentic-community/mcp-gateway-registry/issues/844), [#1216 - Internal/workshop deployment telemetry](https://github.com/agentic-community/mcp-gateway-registry/issues/1216) and more |
| **1.24.6 - Token Mint Audit & Observability** | 0% (0/1) | Planned (Jun 2026) | Token mint audit and observability improvements |
| **1.25.0 - Coding-Assistant OAuth Integration** | 22% (2/9) | In progress (Jun 2026) | [#988 - Umbrella issue](https://github.com/agentic-community/mcp-gateway-registry/issues/988): PRM + AS metadata + WWW-Authenticate ([#989](https://github.com/agentic-community/mcp-gateway-registry/issues/989) done), Entra v1 scope pass-through ([#990](https://github.com/agentic-community/mcp-gateway-registry/issues/990)), RFC 8707 resource enforcement ([#991](https://github.com/agentic-community/mcp-gateway-registry/issues/991)), CIMD publisher/consumer ([#992](https://github.com/agentic-community/mcp-gateway-registry/issues/992), [#993](https://github.com/agentic-community/mcp-gateway-registry/issues/993)), ID-JAG receiver ([#994](https://github.com/agentic-community/mcp-gateway-registry/issues/994)), no-DCR decision doc ([#995](https://github.com/agentic-community/mcp-gateway-registry/issues/995)) |
| **1.27.0 - A2A Gateway & Registry Copilot** | 0% (0/2) | Planned | [#847 - A2A reverse proxy gateway](https://github.com/agentic-community/mcp-gateway-registry/issues/847), [#744 - Registry Copilot embedded chat](https://github.com/agentic-community/mcp-gateway-registry/issues/744) |
| **Parking Lot** | Backlog | Backlog | 25 open issues awaiting prioritization |

**Status Legend:** Complete, In progress, Planned, Backlog

---

#### Major Features

The following major features span multiple milestones and represent significant architectural improvements:

- **[#1172 - Custom Entities & Schemas](https://github.com/agentic-community/mcp-gateway-registry/issues/1172)** **COMPLETED** (1.24.4)
  Turns the registry into a true AI asset registry. Admins define custom entity types with their own schema, and admins and users register entities against that schema. Custom entities create their own dynamic tabs, support lexical and semantic search and voting, and register their own cards, with the same governance and audit trail as MCP servers, agents, and skills. Store n8n workflows, policies, prompt templates, model cards, datasets, evaluations, and more.

- **[#988 - Coding-Assistant OAuth Integration](https://github.com/agentic-community/mcp-gateway-registry/issues/988)** **IN PROGRESS** (1.25.0)
  Umbrella for the seven-phase OAuth 2.1 / RFC 8707 / CIMD / ID-JAG implementation that brings native coding-assistant token flows to the gateway. Phase 1 (PRM + AS metadata) complete.

- **[#1063 - CDK-based Deployment](https://github.com/agentic-community/mcp-gateway-registry/issues/1063)** **PLANNED** (1.24.5)
  CDK deployment option alongside the existing Terraform module for AWS-native installs.

- **[#1020 - Rename `mcpgw` to `airegistry-tools`](https://github.com/agentic-community/mcp-gateway-registry/issues/1020)** **BACKLOG**
  Naming alignment to make the proxy server's role clearer.

- **[#847 - A2A Reverse Proxy Gateway](https://github.com/agentic-community/mcp-gateway-registry/issues/847)** **PLANNED** (1.27.0)
  Add reverse proxy gateway support for A2A agents.

- **[#744 - Registry Copilot](https://github.com/agentic-community/mcp-gateway-registry/issues/744)** **PLANNED** (1.27.0)
  Embedded AI chat assistant for registry operations, discovery, and agent design.

- **[#665 - Agent-to-Agent Knowledge Sharing](https://github.com/agentic-community/mcp-gateway-registry/issues/665)** **BACKLOG**
  Enable agents to share and discover knowledge through the AI Registry, forming a collaborative knowledge network.

- **[#666 - Context Hub MVP](https://github.com/agentic-community/mcp-gateway-registry/issues/666)** **BACKLOG**
  Implement Context Hub with card creation, search, and auto-discovery for agent knowledge management.

- **[#614 - MCP OAuth 2.1 Authorization Spec](https://github.com/agentic-community/mcp-gateway-registry/issues/614)** **BACKLOG**
  Implement RFC 9728 Protected Resource Metadata with native IDE support for MCP OAuth 2.1 authorization.

- **[#556 - AI Gateway & Registry Rebrand](https://github.com/agentic-community/mcp-gateway-registry/issues/556)** **BACKLOG**
  Rename "MCP Gateway Registry" to "AI Gateway & Registry" to reflect expanded support for agents and tools beyond MCP.

- **[#605 - AgentCore Auto-Registration](https://github.com/agentic-community/mcp-gateway-registry/issues/605)** **COMPLETED** (April 2026)
  Automated discovery and registration of Bedrock AgentCore gateways with credential management integration. Full `cli/agentcore/` module with boto3 discovery, registration, token refresh, and security scheme support.

- **[#641 - Okta Identity Provider](https://github.com/agentic-community/mcp-gateway-registry/issues/641)** **COMPLETED**
  Added Okta as an identity provider option alongside Keycloak, Entra ID, Auth0, GitHub, and Google OAuth2.

- **[#557-559 - Observability & Telemetry Suite](https://github.com/agentic-community/mcp-gateway-registry/issues/557)** **COMPLETED**
  Comprehensive telemetry infrastructure with server-side collector, client-side instrumentation, and end-to-end enhancements. [Telemetry docs](docs/TELEMETRY.md).

- **[#129 - Virtual MCP Server Support](https://github.com/agentic-community/mcp-gateway-registry/issues/129)** **COMPLETED**
  Dynamic tool aggregation and intelligent routing using Lua scripting. Enables logical grouping of tools from multiple backend servers into a single virtual endpoint.

- **[#232 - A2A Curated Registry Discovery](https://github.com/agentic-community/mcp-gateway-registry/issues/232)** **COMPLETED**
  Enable agent-to-agent discovery and tool invocation through curated registry patterns.

- **[#260 - Federation Between MCP Registry Instances](https://github.com/agentic-community/mcp-gateway-registry/issues/260)** **COMPLETED**
  Federated registry with bi-directional sync, peer management, chain prevention, orphan detection, and security scan propagation across registries.

- **[#297 - Unified UI Registration Flow](https://github.com/agentic-community/mcp-gateway-registry/issues/297)** **COMPLETED**
  Streamlined registration experience for both MCP servers and A2A agents through a unified interface.

- **[#295 - Multi-Level Tool Usage Rate Limiting](https://github.com/agentic-community/mcp-gateway-registry/issues/295)** **BACKLOG**
  Comprehensive rate limiting architecture with detailed implementation guide for tool usage control.

---

#### Recently Completed (June 2026)

**1.24.4 (June 2026)**

- **[#1172 - Custom entities & schemas](https://github.com/agentic-community/mcp-gateway-registry/issues/1172)** - Admins define custom entity types with their own schema, turning the registry into a true AI asset registry that can store any AI asset (n8n workflows, policies, prompt templates, model cards, and more). Each type gets a dynamic tab, schema-driven registration, semantic and lexical search, voting, and cards.
- **[#1122 - Native OpenTelemetry metrics](https://github.com/agentic-community/mcp-gateway-registry/issues/1122)** - Registry, auth-server, and mcpgw emit application metrics natively via OpenTelemetry, retiring the metrics-service POST path.
- **[#1157 - RRF hybrid search](https://github.com/agentic-community/mcp-gateway-registry/issues/1157) / [#1155 - search_registry tool](https://github.com/agentic-community/mcp-gateway-registry/issues/1155)** - Reciprocal Rank Fusion scoring, per-entity vector pipelines, and a `search_registry` MCP tool for coding assistants.
- **Amazon Cognito IdP** - Added Amazon Cognito as an identity provider on Terraform/ECS with IAM manager support, alongside Keycloak, Entra, Okta, Auth0, GitHub, Google, and PingFederate.

**1.24.1 (May 2026)**

- **[#971 / #399 - Server-side OAuth session store](https://github.com/agentic-community/mcp-gateway-registry/issues/971)** - Moved the OAuth session payload (username, groups, encrypted `id_token`) out of the browser cookie into a server-side MongoDB / DocumentDB store. Eliminates the cookie-size bug class that broke login for Entra users with large group memberships. AES-GCM encryption at rest with a key derived from `SECRET_KEY` via HKDF-SHA256.
- **[#1061 - CVE-2026-4438 (glibc) remediation](https://github.com/agentic-community/mcp-gateway-registry/issues/1061)** - Added `apt-get upgrade -y` to all eight Dockerfiles built on `python:3.14-slim` to pull in the patched libc6 `2.41-12+deb13u3`.
- **[#1044 - Nginx config regeneration race fix](https://github.com/agentic-community/mcp-gateway-registry/issues/1044)** - Serialized config regeneration with a per-process lock and atomic temp-file writes; eliminates corrupt configs on concurrent registration / removal. Also fixed `auth_server` forcing `application/json` and breaking MCP SSE clients.
- **[#1056 - Operations runbooks](https://github.com/agentic-community/mcp-gateway-registry/issues/1056)** - Established a `docs/operations/` namespace with runbooks for incident response, MongoDB export/import, audit log export, and secret rotation.
- **[#1059 - Heartbeat schema v4](https://github.com/agentic-community/mcp-gateway-registry/issues/1059)** - Added deployment-shape fields to the telemetry heartbeat to fix mis-bucketing of long-lived instances in usage analytics.
- **[#1066 - HealthStatus enum sync](https://github.com/agentic-community/mcp-gateway-registry/issues/1066)** - Loosened client-side enum so the CLI list command and frontend handle backend states like `local`, `checking`, and granular `unhealthy: <reason>` strings.

**1.24.0 (May 2026)** — Federation 2.0 and skills phase 1. See [release-notes/1.24.0.md](docs/release-notes/1.24.0.md) for the full list.

**Earlier (1.23.0 and v1.0.x line, April 2026 and prior)**

- **[#906 - Admin Data Export](https://github.com/agentic-community/mcp-gateway-registry/issues/906)** - Admin-only Data Export page for downloading registry collections as JSON.
- **[#897 - Per-skill Auth Credentials UI](https://github.com/agentic-community/mcp-gateway-registry/issues/897)** - Frontend UI for managing per-skill authentication credentials.
- **[#886 - Centralized Log Rotation](https://github.com/agentic-community/mcp-gateway-registry/issues/886)** - Centralized log rotation, auth-server file logging, and log retrieval via MongoDB storage.
- **[#871 - Unified JWT and Static Token Auth](https://github.com/agentic-community/mcp-gateway-registry/issues/871)** - JWT/session auth coexists with static token auth, supporting four credential types concurrently.
- **[#856 - ARM64 Docker Images](https://github.com/agentic-community/mcp-gateway-registry/issues/856)** - Multi-architecture Docker images with ARM64 support.
- **[#851 - Direct M2M Client Registration](https://github.com/agentic-community/mcp-gateway-registry/issues/851)** - Direct machine-to-machine client registration API that bypasses IdP sync.
- **[#824 - Python 3.14 Runtime Upgrade](https://github.com/agentic-community/mcp-gateway-registry/issues/824)** - Upgraded Python runtime from 3.12 to 3.14 to resolve CVE-2025-13836.
- **[#809 - Registration Gate Admission Control](https://github.com/agentic-community/mcp-gateway-registry/issues/809)** - Admission control webhook for agent, server, and skill registration.
- **[#779 - Multiple Static API Keys](https://github.com/agentic-community/mcp-gateway-registry/issues/779)** - Multiple static API keys with per-key group and scope assignments.
- **[#742 - Webhook Notifications](https://github.com/agentic-community/mcp-gateway-registry/issues/742)** - Configurable webhook notification on server, agent, and skill registration events.

For the complete list of all issues, feature requests, and detailed release history, visit:
- [All GitHub Issues](https://github.com/agentic-community/mcp-gateway-registry/issues)
- [All GitHub Milestones](https://github.com/agentic-community/mcp-gateway-registry/milestones)
- [Release Notes](docs/release-notes/)

---

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if it helps your organization!**

[Get Started](docs/installation.md) | [Documentation](docs/) | [Contribute](CONTRIBUTING.md)

</div>
