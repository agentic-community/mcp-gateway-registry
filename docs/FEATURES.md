# MCP Gateway & Registry - Feature Overview

This document provides a comprehensive overview of the MCP Gateway & Registry solution capabilities, designed for stakeholder presentations, marketing materials, and solution demonstrations.

## Core Problem Solved
- **Multi-Platform AI Tool Integration**: Unified gateway for accessing tools across different MCP servers, eliminating the need to manage multiple connections and authentication schemes
- **Centralized Tool Catalog**: Registry acts as a comprehensive catalog of available tools for developers, AI agents, and knowledge workers
- **Dynamic Tool Discovery**: Intelligent routing based on natural language queries and semantic matching, reducing configuration overhead

## Registry & Management
- **Centralized Server Registry**: MongoDB/DocumentDB-backed configuration for all MCP servers and their capabilities
- **Dynamic Tool Catalog**: Real-time discovery of available tools across registered servers
- **MCP Server Version Routing**: Run multiple versions of the same server behind a single gateway endpoint with instant rollback, version pinning, and deprecation lifecycle
- **Custom Metadata**: Add rich custom metadata to servers and agents for organization, compliance, and integration tracking, fully searchable via semantic search
- **Custom Entity Types**: Admins define schema-driven catalog types at runtime (e.g. LLM prompts, model cards, workflows) with typed fields; users create records via API or a dynamically-rendered UI tab, with lexical + semantic search, ratings, visibility controls, and pagination
- **Server & Agent Rating System**: 5-star rating widget with aggregate scoring, one rating per user, and rotating buffer
- **Health Monitoring**: Built-in health checks and status monitoring for all registered services
- **Scalable Architecture**: Docker-based deployment with horizontal scaling support

## Agent Registry & A2A Communication
- **A2A Protocol Support**: Agent registration, discovery, and direct agent-to-agent communication
- **Agent Security Scanning**: Integrated scanning using Cisco AI Defense A2A Scanner with YARA pattern matching and heuristic threat detection
- **Agent Discovery API**: Semantic search API for dynamic agent composition at runtime
- **Agent Cards & Metadata**: Rich metadata for agent capabilities, skills, and authentication schemes

## Authentication & Security
- **Multi-Provider OAuth 2.0/OIDC Support**: Keycloak, Microsoft Entra ID, AWS Cognito integration
- **Multi-Provider IAM**: Harmonized API for user and group management across identity providers
- **Static Token Auth**: IdP-independent API access for Registry endpoints using static API keys, designed for CI/CD pipelines and trusted network environments
- **Enterprise SSO Ready**: Seamless integration with existing identity providers including Microsoft Entra ID
- **Service Principal Support**: M2M service accounts with OAuth2 Client Credentials flow for AI agent identity
- **Fine-Grained Access Control**: Scopes define which MCP servers, methods, tools, and agents each user can access
- **Application-Level Rate Limiting**: Identity/group/target-aware request limits at the auth-server `/validate` hop (complementary to per-IP nginx edge limiting). Cap a caller (user or agent, by rate-limit group membership), each caller independently per target (`caller_target`), and/or a target (a single MCP server / A2A agent, or a **server group** — a named set of servers each getting its own independent per-member bucket) per time window, with config-time lockout-safeguard floors, admin bypass, a fail-open availability guardrail (fail-closed per-limit), and 429 responses carrying `X-RateLimit-*` / `Retry-After`. Includes **quarantine** (an absolute kill switch for a caller or target, managed admin-only from the Users / M2M rows and the Rate Limits panel; admin-group callers cannot be quarantined, enforced server-side and fail-closed). Off by default; definitions managed via the admin API / CLI / UI. See [Rate Limiting Design](design/rate-limiting.md)
- **Self-Signed JWT Tokens**: Human users can generate tokens for CLI tools and AI coding assistants
- **Secure Token Management**: OAuth token refresh and validation with centralized session management
- **MCP Server Security Scanning**: Integrated vulnerability scanning with Cisco AI Defense MCP Scanner
- **Compliance Audit Logging**: Comprehensive audit logs for all API and MCP access events with TTL-based retention, credential masking, and admin UI for compliance monitoring

## Intelligent Tool Discovery
- **Hybrid Search**: Combined vector similarity with tokenized keyword matching for servers, tools, and agents
- **Semantic Search**: HNSW vector search using sentence transformers or LiteLLM-supported providers
- **Unified Search**: Single endpoint searches across MCP servers, tools, and A2A agents
- **Tag-Based Filtering**: Multi-tag filtering with AND logic for precise tool selection
- **Flexible Embeddings**: Local sentence-transformers, OpenAI, Amazon Bedrock Titan, or any LiteLLM-supported provider
- **Performance Optimized**: Configurable result limits and caching for fast response times

## Developer Experience
- **MCP Registry CLI**: Claude Code-like conversational interface for registry management with real-time token status and cost tracking
- **Registry Management API**: Programmatic API for managing servers, groups, and users with Python client
- **Multiple Client Libraries**: Python agent with extensible authentication
- **Comprehensive Documentation**: Setup guides, API documentation, and integration examples
- **Testing Framework**: 850+ pytest tests (unit, integration, E2E) with GitHub Actions CI
- **Development Tools**: Docker Compose for local development and testing

## Federation & External Registries
- **Peer-to-Peer Registry Federation**: Connect MCP Gateway Registry instances for bidirectional server and agent sync with static token or OAuth2 authentication
- **Federation UI**: VS Code-style Settings page for managing peer registries, sync modes (all, whitelist, tag filter), and monitoring sync status
- **Federated Registry**: Import servers and agents from external registries
- **Anthropic MCP Registry**: Import curated MCP servers with API compatibility
- **Workday ASOR**: Import AI agents from Agent System of Record
- **Automatic Sync**: Scheduled synchronization with external registries and peer registries
- **Amazon Bedrock AgentCore**: Gateway support with dual authentication

## Enterprise Integration
- **Container-Ready Deployment**: Docker Hub images with pre-built containers
- **AWS ECS Production Deployment**: Multi-AZ Fargate deployment with ALB, auto-scaling, CloudWatch, and Terraform
- **Flexible Deployment Modes**: CloudFront Only, Custom Domain with Route53/ACM, or CloudFront + Custom Domain
- **Reverse Proxy Architecture**: Nginx-based ingress with SSL termination
- **DocumentDB & MongoDB CE Storage**: Distributed storage with HNSW vector search
- **Real-Time Metrics & Observability**: Grafana dashboards with SQLite and OpenTelemetry integration
- **Configuration Management**: Environment-based configuration with validation

## Technical Specifications
- **Protocol Compliance**: Full MCP (Model Context Protocol) specification support
- **A2A Protocol**: Agent-to-Agent protocol support for autonomous agent ecosystems
- **High Performance**: Async/await architecture with concurrent request handling
- **Extensible Design**: Plugin architecture for custom authentication providers
- **Cross-Platform**: Linux, macOS, Windows support with consistent APIs

## Deployment Options
- **Pre-built Images**: Deploy instantly with Docker Hub images
- **Quick Start**: Docker Compose setup in minutes
- **AWS ECS Fargate**: Production deployment with Terraform
- **Cloud Native**: Kubernetes manifests and cloud deployment guides
- **Local Development**: MongoDB CE with full-featured local development
- **Podman Support**: Rootless container deployment for macOS and Linux

## Use Cases Supported
- **AI Agent Orchestration**: Centralized tool access for autonomous agents
- **Agent-to-Agent Communication**: Direct peer-to-peer agent communication through unified registry
- **CI/CD Integration**: Static token auth for automated pipelines without IdP dependency
- **Enterprise Tool Consolidation**: Single gateway for diverse internal tools
- **Development Team Productivity**: Unified interface for developer tools and services
- **Research & Analytics**: Streamlined access to data processing and analysis tools
- **Customer Support**: Integrated access to support tools and knowledge bases

## Competitive Advantages
- **Zero Vendor Lock-in**: Open architecture supporting any MCP-compliant server
- **Unified Agent & Server Registry**: Single control plane for both MCP servers and AI agents
- **Minimal Configuration**: Automatic tool discovery reduces setup complexity
- **Enterprise Security**: Authentication and authorization with multiple IdP support
- **Developer Friendly**: Clear APIs, CLI tools, and comprehensive documentation
- **Cost Effective**: Reduces integration overhead and maintenance complexity
