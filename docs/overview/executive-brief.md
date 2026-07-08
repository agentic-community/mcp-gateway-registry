# Executive Brief

*A one-page overview for decision-makers evaluating adoption. Every quantitative claim is
verifiable against the repository; figures are stated with an "as of" date.*

## The problem

AI assets are proliferating faster than organizations can govern them. Teams stand up MCP
servers, agents, and skills independently — each with its own credentials, its own endpoint, and
its own discovery story. The result is credentials scattered across developer dotfiles, no
central inventory of what exists, no audit trail of who accessed what, and no clean way to
off-board a user or revoke access when someone leaves.

## The solution

The MCP Gateway & Registry provides **one governed control plane** for every AI asset. A single
registry gives humans, agents, and AI coding assistants one place to discover MCP servers,
agents, and skills. An optional gateway puts all traffic through **one authenticated entry point**,
enforcing access with your existing identity provider before any request reaches a backend. The
registry decides what is allowed; the gateway enforces it; every access is audited.

## Governance and security posture

Each capability below exists in the codebase today; follow the link for detail.

- **Identity via your existing IdP.** Six providers are supported in code — Keycloak, Amazon
  Cognito, Microsoft Entra ID, Okta, Auth0, and PingFederate — selected by configuration, with no
  coupling to any one. See [Authentication](../registry-auth-architecture.md).
- **Fine-grained authorization.** Scopes govern which servers, tools, and agents a user or service
  account can reach; skills carry per-skill tool allowlists. See [Scopes](../scopes.md).
- **Audit logging.** A dedicated audit subsystem records access events with credential masking.
  See [Audit Logging](../audit-logging.md).
- **Security scanning of registered assets.** Servers, skills, and agents are scanned; unsafe items
  are held for review. See [Security Scanner](../security-scanner.md) and
  [Security Posture](../security-posture.md).
- **Admission control that fails closed.** An optional external registration gate can approve or
  deny assets before they are persisted; if it is unreachable, registration is blocked, not waved
  through. See [Registration Webhooks & Gate](../registration-webhooks.md).
- **Agent trust verification (optional).** Read-only "bring your own ANS ID" integration verifies
  agent identity metadata without the registry managing any certificates. See
  [ANS Integration](../design/ans-integration.md).

## Deployment and operations

Three deployment surfaces are supported, and configuration is kept at parity across all three via
a [single cross-surface parameter reference](../unified-parameter-reference.md):

- **Docker Compose** — for evaluation and local development.
- **Amazon ECS (Terraform)** — a Fargate stack under [`terraform/aws-ecs/`](../../terraform/aws-ecs).
- **Amazon EKS (Helm)** — Helm charts under [`charts/`](../../charts).

Operational signal is first-class: services emit metrics via OpenTelemetry with an always-on
Prometheus `/metrics` endpoint and optional OTLP push to a backend of your choice, plus health
endpoints for monitoring. See [Observability](../OBSERVABILITY.md).

## Maturity signals

*As of 2026-07-08:*

- **Tests:** 4,940 test functions across 252 test files (`grep -rc "def test_" tests/`).
- **Releases:** 30 versioned release notes in [`docs/release-notes/`](../release-notes/1.26.0.md),
  the latest being **1.26.0** (July 2026).
- **Community:** 775 GitHub stars and 205 forks
  ([repository](https://github.com/agentic-community/mcp-gateway-registry)).
- **Hands-on training:** an
  [AWS Workshop Studio lab](https://catalog.us-east-1.prod.workshops.aws/workshops/0c3265a6-1a4a-467b-ae56-e4d019184b0e/en-US)
  and an [AWS Show & Tell talk](https://www.youtube.com/watch?v=dk0qVukHLGU).
- **Demos:** a catalog of walkthroughs in [Demo Videos](../demo-videos.md).

## How to evaluate it in 30 minutes

1. Follow the [Quick Start](../quickstart.md) to get a running instance.
2. Watch the end-to-end demo in [Demo Videos](../demo-videos.md).
3. Read the [Theory of the System](../design/theory-of-the-system.md) to understand the design and
   its invariants.

## License and community

Licensed under **Apache 2.0**. Developed in the open under the
[agentic-community](https://github.com/agentic-community/mcp-gateway-registry) organization, with
feature work shipped as public pull requests.
