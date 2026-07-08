# Documentation

This directory contains the project documentation as plain Markdown. Browse it directly on
GitHub, or start from the top-level [README](../README.md).

The published landing page at
[agentic-community.github.io/mcp-gateway-registry](https://agentic-community.github.io/mcp-gateway-registry/)
is a single page generated from the top-level `README.md` by
[`scripts/build_landing_page.py`](../scripts/build_landing_page.py) and deployed by the
`Build and Deploy Landing Page` GitHub Actions workflow. The detailed guides below are read as
Markdown here in the repository.

## Where to start

- [Quick Start](quickstart.md) and [Installation Guide](installation.md)
- [Complete Setup Guide](complete-setup-guide.md)
- [Authentication](auth.md) and [Access Control & Scopes](scopes.md)
- [Theory of the System](design/theory-of-the-system.md) - the design and its invariants
- [Executive Brief](overview/executive-brief.md)
- [FAQ / Troubleshooting](faq/index.md)
- [Release Notes](release-notes/)

## Contributing docs

Add or edit Markdown files anywhere under `docs/`. Use repository-relative links so they resolve
both on GitHub and in forks. No build step or navigation config is required.
