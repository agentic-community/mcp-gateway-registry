# Getting Started

## Prerequisite Reading
**READ THIS FIRST:** [CONTRIBUTING.md](CONTRIBUTING.md)

Before you start contributing, please review the project's contribution guidelines.

**Then read the [Theory of the System](docs/design/theory-of-the-system.md)**. It is the causal design
narrative and the core invariants (control-plane/data-plane split, generic reverse-proxy gateway,
A2A peer-to-peer, config parity, fail-closed admission, and more). A change that breaks an
invariant without arguing for it will be flagged in review, so understand them before you design.

## Setup Instructions for Contributors

### Step 1: Choose Your Development Environment
We recommend the fastest option to get started:

#### Option A: macOS Setup (Fastest ⚡)
Complete this setup guide first:

- [macOS Setup Guide](docs/macos-setup-guide.md)
- Time to first run: ~30 minutes

#### Option B: EC2 Complete Configuration (Preferred for Server Setup)
If working on EC2 or a Linux server, complete this guide first:

- [Complete Setup Guide](docs/complete-setup-guide.md)
- Time to first run: ~60 minutes

## Before You Start Coding

### 1. Ask Your Coding Assistant to Read Documentation
Before making any code changes, ask your AI coding assistant to read:

**LLM/AI Documentation (Critical for understanding the project):**
- [docs/llms.txt](docs/llms.txt)
- [docs/design/theory-of-the-system.md](docs/design/theory-of-the-system.md) - the system's core invariants and the reasoning behind them

**Coding Standards and Guidelines:**
- [AGENTS.md](AGENTS.md) - Project-specific coding standards (`CLAUDE.md` imports it)

### 2. Review the AGENTS.md File
This project uses [AGENTS.md](AGENTS.md) for coding standards, imported by `CLAUDE.md` so every agent reads one file. The file is already included in the repository root - make sure to review it before contributing.

## Testing Your Changes

Before submitting a pull request, you must run and pass the test suite:

### Quick Start Testing
```bash
# One-time: install the dev extras. A plain `uv sync` omits test dependencies
# such as freezegun, and the check below fails without them.
uv sync --extra dev

# Run the suite locally. These are the same commands CI runs on your PR.
# Drop -n for serial if 8 workers exhaust memory on a small instance.
uv run python scripts/test.py check
uv run python scripts/test.py coverage -n 8

# Faster iteration while you work:
uv run python scripts/test.py fast

# Other targets (unit, integration, e2e, auth, servers, search, health, core):
uv run python scripts/test.py --help
```

The pytest suite mocks its external dependencies, so it needs no running stack and no credentials.

### For PR Merge (REQUIRED)

The same `check` and `coverage` commands as above. All tests must pass (0 failures) before merging, and CI runs them again on the PR.

### Understanding the Tests
See the testing documentation:

- **[tests/README.md](tests/README.md)** - Start here. Layout of the test tree, fixtures and auto-mocking
- **[docs/testing/README.md](docs/testing/README.md)** - Testing guide: markers, coverage, CI and troubleshooting
- **[docs/testing/QUICK-START.md](docs/testing/QUICK-START.md)** - Quick reference for the `scripts/test.py` targets
- **[docs/scopes.md](docs/scopes.md)** - Permission model. Scopes live in DocumentDB; see [docs/scopes-mgmt.md](docs/scopes-mgmt.md) to manage them

### Testing Against a Running Stack

Manual and end-to-end checks against a deployed gateway need fresh tokens. Keycloak access tokens expire after 5 minutes by default.

```bash
# Tokens for all agents and services
./credentials-provider/generate_creds.sh

# Tokens for the access-control test accounts (admin, LOB1, LOB2)
./keycloak/setup/generate-agent-token.sh admin-bot
./keycloak/setup/generate-agent-token.sh lob1-bot
./keycloak/setup/generate-agent-token.sh lob2-bot
```

Then follow [docs/testing.md](docs/testing.md) for the CLI and agent test workflows.

## Fork and Contribute

### Repository Access
**Important:** There is no direct access to this repository. To contribute:

1. **Fork the repository on GitHub**
   ```
   https://github.com/agentic-community/mcp-gateway-registry
   ```

2. **Clone your fork locally**
   ```bash
   git clone https://github.com/YOUR-USERNAME/mcp-gateway-registry.git
   cd mcp-gateway-registry
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```

4. **Make your changes** following the coding standards in AGENTS.md

5. **Commit and push to your fork**
   ```bash
   git push origin feat/your-feature-name
   ```

6. **Create a Pull Request** to the main repository
   - Use a clear, descriptive PR title
   - Reference any related issues
   - Include test results and screenshots if applicable

## Development Checklist
Before submitting a pull request:

- [ ] Completed one of the setup guides (macOS or EC2)
- [ ] Read docs/llms.txt
- [ ] Read AGENTS.md (coding standards)
- [ ] Code follows project conventions (use ruff, mypy, pytest)
- [ ] Tests pass locally: `uv run python scripts/test.py check` and `uv run python scripts/test.py coverage -n 8` (0 failures)
- [ ] Reviewed test documentation: [tests/README.md](tests/README.md)
- [ ] Changes are pushed to a fork, not directly to this repo
- [ ] Pull request is created with clear description

## Questions?
- Check the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details
- Review existing PRs to see contribution patterns
- Ask your coding assistant to review the documentation with you

Happy coding! 🚀
