# Claude Coding Rules

## Overview
This document contains coding standards and best practices that must be followed for all code development. These rules prioritize maintainability, simplicity, and modern Python development practices.

## Core Principles
- Write code with minimal complexity for maximum maintainability and clarity
- Choose simple, readable solutions over clever or complex implementations
- Prioritize code that any team member can confidently understand, modify, and debug

## Pull Request Evaluation

When evaluating pull requests for merge, adopt the **Merge Specialist** persona defined in [.claude/skills/pr-review/personas/merge-specialist.md](.claude/skills/pr-review/personas/merge-specialist.md). This persona provides comprehensive guidelines for:

- Running and verifying tests
- Assessing code quality against these standards
- Reviewing architecture and design decisions
- Checking for breaking changes
- Evaluating performance impact
- Ensuring documentation is complete

**IMPORTANT**: Before approving any PR for merge, the Merge Specialist must verify that all tests pass and no existing functionality is broken. A PR with failing tests should NEVER be approved for merge.

## Python Version

`pyproject.toml` is the source of truth: `requires-python = ">=3.14"`. Do not restate a
version anywhere else; read it from there.

One known inconsistency to be aware of rather than copy: `[tool.ruff] target-version`
is still `py311`, so ruff will not apply modernizations newer than 3.11 even though the
runtime is 3.14. Raising it is a small separate change; until then, expect ruff to be
more conservative than the runtime allows.

## Technology Stack

### Package Management
- Always use `uv` and `pyproject.toml` for package management
- Use `uv` for every dependency operation. Never invoke `pip` on its own

### Modern Python Libraries
- **Data Processing**: Use `polars` instead of `pandas`
- **Web APIs**: Use `fastapi` instead of `flask`
- **Code Formatting/Linting**: Use `ruff` for both linting and formatting
- **Type Checking**: Use `mypy` - type checks have become actually useful and should be part of CI/CD
- **Performance**: Leverage modern CPython improvements - CPython is now much faster

## Code Style Guidelines

### Function Structure
- All internal/private functions must start with an underscore (`_`)
- Private functions should be placed at the top of the file, followed by public functions
- Functions should be modular, containing no more than 30-50 lines
- Use two blank lines between function definitions
- One function parameter per line for better readability

### Type Annotations
- Use clear type annotations for all function parameters
- One function parameter per line for better readability
- Use modern type hint syntax (PEP 604/585). See Python version below
- Example:
  ```python
  def process_data(
      input_file: str,
      output_format: str,
      validate: bool = True
  ) -> dict[str, Any]:
      pass
  ```

### Modern Type Hint Standards

Use built-in generics and `|` unions, never `typing.List` or `Optional[X]`:

```python
def process(data: list[dict[str, Any]], limit: int | None = None) -> dict | None: ...
```

`ruff` enforces this (`UP006`, `UP007`, `UP037`), so a violation is a lint failure
rather than a review comment.

### Class Definitions with Pydantic

Prefer a Pydantic `BaseModel` over a bare class so validation, coercion and
serialization come for free. Put constraints in `Field(...)` and use a validator for
anything cross-field.

### Main Function Pattern
- The main function should act as a control flow orchestrator
- Parse command line arguments and delegate to other functions
- Avoid implementing business logic directly in main()

### Command-Line Interface Design

Use `argparse` with a `RawDescriptionHelpFormatter` and an `epilog` holding real
example invocations. Two project conventions:

- Accept a value from either a CLI flag or an environment variable, flag winning.
- Where a count means "process everything", use `0` for all, and say so in `--help`.

### Imports
- Write imports as multi-line imports for better readability
- Example:
  ```python
  from .services.output_formatter import (
      _display_evaluation_results,
      _print_results_summary,
      _check_mcp_generation_criteria
  )
  ```

### Constants
- Don't hard code constants within functions
- For trivial constants, declare them at the top of the file:
  ```python
  STARTUP_DELAY: int = 10
  MAX_RETRIES: int = 3
  ```
- For many constants, create a separate `constants.py` file with a class structure

### Logging Configuration
- Always use the following logging configuration:
  ```python
  import logging

  # Configure logging with basicConfig
  logging.basicConfig(
      level=logging.INFO,  # Set the log level to INFO
      # Define log message format
      format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
  )
  ```

### Logging Best Practices
- Add sufficient log messages throughout the code to aid in debugging and monitoring
- Don't shy away from adding debug logs using `logging.debug()` for detailed tracing
- When printing a dictionary as part of a trace message, always pretty print it:
  ```python
  logger.info(f"Processing data:\n{json.dumps(data_dict, indent=2, default=str)}")
  ```
- Consider adding a `--debug` flag to the application that sets the logging level to DEBUG:
  ```python
  if args.debug:
      logging.getLogger().setLevel(logging.DEBUG)
  ```

### Performance Feedback

For anything that can run longer than a few seconds: log the elapsed time when it
finishes, and log a `WARNING` before starting the full-dataset path. Log the resolved
configuration at startup (`config.model_dump()`) so a run is reproducible from its own
logs.

### Performance Optimization
- Use `@lru_cache` decorator where appropriate for expensive computations

### External Resource Management

When reading from an API, dataset or database outside this repo:

- Pin the version, revision or schema date in a constant, with the docs URL beside it.
- Log each filter's effect on row count (`'key=value': 120 -> 8 items`), so an empty
  result is traceable to the filter that caused it.
- Raise with a message naming the source, the credential it needs and the docs URL.
  Never return an empty result the caller cannot distinguish from a failure.

### Decorators and Functional Patterns

Use a decorator when it is standard and single-purpose (`@property`, `@dataclass`,
`@lru_cache`). Use a comprehension or `map` for a simple transformation. Do not chain
`reduce`/`filter`/`map` into one expression: write the loop.

Keep nesting to two or three levels. Prefer an early return, and extract nested logic
into a named `_helper` rather than indenting further.

### Code Validation
- Always run `uv run python -m py_compile <filename>` after making changes to Python files
- Always run `bash -n <filename>` after making changes to bash/shell scripts to check syntax

## Error Handling and Exceptions

- Catch specific exception types. Never a bare `except:`.
- Log with context, then re-raise or wrap. `logger.exception` inside an `except` block
  keeps the traceback.
- Fail fast and loudly. Do not swallow an error into a falsy return the caller cannot
  distinguish from a legitimate empty result.
- Wrap a third-party error in a domain exception with `raise ... from e`.
- Error messages say what was attempted, what was wrong, and what to do about it.

## Testing Standards

### Testing Framework
- Use `pytest` as the primary testing framework
- The suite enforces a 35% floor repo-wide (`--cov-fail-under=35`); aim for 80% on code you add,
  which is the bar the per-feature testing plans use
- Use `pytest-cov` for coverage reporting

### Test Structure

Follow Arrange, Act, Assert. Name the test for the behavior it pins
(`test_rejects_expired_token`, not `test_token_2`). Mock at the boundary, put shared
data in a fixture, and cover the error paths as well as the happy one.

### Testing Best Practices
- Follow AAA pattern: Arrange, Act, Assert
- One assertion per test when possible
- Use descriptive test names that explain what is being tested
- Mock external dependencies
- Use fixtures for common test data
- Test both happy paths and error cases

### Running Tests Before Pull Requests

**CRITICAL**: Always run the full test suite before submitting a pull request or after completing a major feature.

#### When to Run Tests
1. **Before submitting a pull request**: All tests must pass before creating a PR
2. **After completing a major feature**: Verify no regressions were introduced
3. **After making significant refactoring changes**: Ensure existing functionality still works
4. **After updating dependencies**: Verify compatibility with new versions

#### How to Run Tests

`scripts/test.py` is the canonical runner, and it is what CI invokes. Prefer it over a
bare `pytest` so local and CI behavior cannot diverge:

```bash
uv run python scripts/test.py check          # verify the test dependencies resolve
uv run python scripts/test.py coverage -n 8  # what CI runs on every PR
```

Other targets: `unit`, `integration`, `e2e`, `fast`, `full`, plus the focused
`auth`, `servers`, `search`, `health` and `core` suites. `-n` takes a worker count or
`auto`; omit it for serial.

A bare `uv run pytest tests/ -n 8` still works and is fine for iterating on one file,
but it is not the gate. When a result has to match CI, run the `coverage` target.

Do not record an expected test count here. It rots within a release, and the real
figure is whatever CI reports on the PR.

#### Test Prerequisites
Before running tests, ensure:

1. **MongoDB is running** (for integration tests):
   ```bash
   docker ps | grep mongo
   # Should show: mcp-mongodb running on 0.0.0.0:27017
   ```

2. **Test environment is configured**:
   - Tests automatically set `DOCUMENTDB_HOST=localhost`
   - Tests use `mongodb-ce` storage backend
   - Tests use `directConnection=true` for single-node MongoDB

#### Continuous Integration
Tests run automatically via GitHub Actions when:
- Pull requests are created targeting `main` or `develop` branches
- Code is pushed to `main` or `develop` branches

See [.github/workflows/registry-test.yml](.github/workflows/registry-test.yml:7-8) for CI configuration.

#### Acceptable Test Results
- **All unit tests must pass** (no failures allowed in unit tests)
- **Integration tests**: Some tests may be skipped due to known issues
- **Coverage**: Minimum 35% coverage required (`--cov-fail-under=35` in pyproject.toml:114)
- **Warnings**: Minor warnings are acceptable, but investigate new warnings

#### What to Do If Tests Fail
1. Review the test failure output carefully
2. Fix the failing test(s) before submitting PR
3. Re-run tests to verify the fix
4. Never submit a PR with failing tests
5. If a test failure is unrelated to your changes, investigate and fix it or document why it should be skipped

## Async/Await Best Practices

- Use `async with` for async context managers so the resource is released on error.
- Use `asyncio.gather(..., return_exceptions=True)` for concurrent work, and handle the
  returned exceptions rather than letting one failure hide the rest.
- Never call a blocking function from a coroutine. Push it to a thread, or use the
  async client.
- Enter async code from sync code through `asyncio.run()`, once, at the edge.

## Documentation Standards

Use Google-style docstrings: a one-line summary, then `Args:`, `Returns:`, `Raises:`,
and an `Example:` where the call shape is not obvious. Every public function needs one.
Document the exceptions a caller has to handle, and update the docstring in the same
commit as the behavior.

## Security Guidelines

### Security invariants (always apply)

These are hard rules distilled from prior security-review findings. Keep them in
mind on every change touching auth, authorization, outbound requests, credential
handling, config generation, or logging. Full detail, rationale, and examples
are in [docs/SECURITY_GUIDELINES.md](docs/SECURITY_GUIDELINES.md). Read it before
working in those areas.

- **Use the canonical helper, never reinvent/copy-paste** (see the "Canonical
  helpers" section in the guidelines doc): outbound HTTP from user/registry URLs →
  `registry/utils/url_guard.py` `guarded_client`/`guarded_async_client` (never a
  bare `httpx` client), IP categories/literal and tunnel unwrapping →
  `registry/utils/url_guard.py`, URL identity → `normalize_url_identity`, URL log
  redaction → `registry/common/log_redaction.py` `redact_url`; CSRF →
  `registry/auth/csrf.py`; internal service auth →
  `registry/auth/internal.py`; signing-secret validation →
  `registry/common/secret_key.py`; read redaction → `registry/services/visibility.py`;
  frontend hrefs → `frontend/src/utils/safeUrl.ts`; log redaction (never log raw
  headers/body/user_context/claims) → `registry/common/log_redaction.py`; writing a
  credential to disk → `os.open(..., 0o600)` atomically, never print it. A copied
  snippet is how these findings get reopened.
- **Fail closed.** On error, missing config, or ambiguity, DENY. A check that can
  be silently skipped (optional param, truthiness on an emptyable value, a
  sanitizer that isn't called) is equivalent to no check.
- **Signing secrets:** validate for missing AND weak (unset/empty/whitespace,
  `< 32` stripped chars, known-weak literals, weak-check before length) at every
  signing entrypoint. Never ship a default/vendor credential fallback in code;
  use one chokepoint that raises and denylists the weak value. Example creds must
  be unmistakably fake (`YOUR_*`).
- **SSRF:** one shared hardened URL guard for every outbound fetch. Block
  RFC-1918/loopback/link-local/reserved/multicast and exact cloud/workload
  metadata endpoints (AWS plus Alibaba `100.100.100.200`), including scoped/
  mapped/NAT64/6to4/Teredo forms. Validate at registration AND pin at fetch time.
  Credentialed OAuth token endpoints MUST use the HTTPS-only empty-allowlist
  `CREDENTIALED_OAUTH_PROFILE` at both points, never `PROXY_PROFILE`. Credential
  builders self-guard the exact derived destination before adding/decrypting any
  configured header; missing/invalid/mismatched destinations return protocol-
  only headers. Actual ARD fetches use guarded transport; its domain wrapper
  delegates canonical URL/IP logic.
- **Injection:** sanitize at EVERY interpolation site AND validate at the source
  (a sanitizer that exists but isn't called is worthless); `re.escape` user input
  in regex/`$regex` queries.
- **Authorization:** deny by default; never treat a broad/execute scope as admin;
  enforce ownership server-side before every mutation across the whole endpoint
  family; `dict.get("k")` not `getattr(dict, "k")`; no substring matching for
  privilege decisions; verify externally-supplied JWTs (sig/iss/aud/exp) before
  trusting claims; attach shared/global credentials only on explicit admin opt-in.
- **Virtual backend grants:** internal subrequests that rewrite the backing URL
  must preserve any resource-bound token's virtual URI in a trusted nginx-only
  header (clear client copies on normal routes), require the nginx marker and
  resolved upstream, and check the rewritten backing method/tool grant separately.
  Never grant direct backing access from virtual binding alone.
- **MCP backend identity:** preserve the exact registered path (including a
  legitimate trailing `/mcp`) across auth, signed proxy token, and vend; never
  strip a transport segment twice. Bind the exact outbound URL at credential
  write (snapshot at consent START) and check it on every vend read; select the
  version inside the registration's own nginx location (see Route-derived authz).
- **Never log** secrets, tokens, PII, or full credential/claim payloads. Redact
  (including setup/debug scripts in verbose mode).
- **OAuth/OIDC:** bind the code flow to the login with a per-login `nonce` (checked
  after signature verification) + PKCE (`S256`), fail closed if the verifier is
  missing. Authorize the EXACT bytes you forward, never a separately-captured
  copy; fail closed when the body isn't inspectable.
- **Response projection:** redaction + access checks must be uniform across the
  whole entity family (versions, bulk, discovery projections, search, admin-config
  reads) via one shared helper, not just the reported endpoint. Mutation success
  responses, webhooks, lifecycle events, and exports must project a fresh recursive
  token-free copy; never serialize the encrypted storage object directly.
- **Tokens/JWT:** never derive `verify_aud`/issuer/alg from an unverified claim:
  enforce audience against a config allowlist, fail closed. Never auto-grant
  groups/admin from a code-shipped mapping (config-driven, fail closed).
- **Admin ops:** refuse self-delete + last-admin removal/demotion (fail closed if
  the admin population can't be counted); audit admin-tier grants. A config/secrets
  export must deny sensitive values by default. Gate them behind a SEPARATE
  explicit acknowledgement (not just `include_sensitive`) and fail closed.
- **Never put a secret on subprocess argv** (world-readable via `ps`). Pass via
  `env=`/stdin. **Trust forwarded metadata only from the proxy hop:** rightmost/
  trusted XFF (not leftmost), allowlist `Host` before building a redirect_uri.
  Scope the ASGI server's `--forwarded-allow-ips` to the real peer (loopback when
  nginx is co-located), NEVER `*`, because `*` lets uvicorn set `request.client` from the
  spoofable leftmost XFF. An nginx `auth_request` subrequest does NOT inherit the
  parent location's `proxy_set_header`, so re-set trusted headers inside the
  subrequest block. A `1.2.3.4:0` ASGI *access-log* line ≠ a spoofed *audit* value
  (different resolvers). Check the durable record before concluding a breach.
- **One entity type's access grant must never gate a different entity type** (a
  skill filter keyed on agent access = bypass); filter each resource by its own
  access check, admin-only universal bypass.
- **Honor the disabled/inactive flag** on every request where access is derived
  (group→scope enrichment / validate), not just login; filter it in the query AND
  re-check the doc; fail closed.
- **IAM least privilege:** no wildcard `Action`/`Resource`. Scope to the exact
  operations the code calls + specific ARNs; cross-account AssumeRole behind an
  explicit-list default-empty (fail closed); keep Terraform+CDK in parity.
- **Internal tokens:** short TTL isn't enough: add `jti` + single-use via a
  shared store (unless the token is legitimately verified twice per flow).
- **DoS:** rate-limit at the inbound edge (nginx `limit_req`), never the shared
  `/validate` subrequest; cover all deploy modes. **Audit:** durable-by-default
  (fail closed), attributable to a specific actor.
- **OAuth CSRF `state`:** validate at the token-exchange point, not a post-hoc
  check that may be dead code; OAuth discovery metadata → `Cache-Control: no-store`.
- **Frontend:** one shared URL-scheme guard on every dynamic href/`window.open`/
  markdown link (allowlist http/https/mailto, render unsafe as text); enforce
  `react/jsx-no-script-url`.
- **Deployment/config:** sensitive ports on loopback not `0.0.0.0`; no working
  default for a required secret (`${VAR:?}` + reject weak literals); no secrets in
  Docker build ARGs; pin image tags; TLS verify on by default (private certs via
  a CA bundle, never `verify=False`); dangerous toggles require an explicit flag +
  localhost guard.
- **Approval TOCTOU:** snapshot what a user approves (destinations, token
  endpoint) into the signed consent state when consent BEGINS; the callback binds
  to the snapshot, never to a re-read server record. Apply a credential's binding
  checks to EVERY vault read that is used/refreshed, not just the first.
- **Route-derived authz:** bind to what the selecting nginx location asserted
  (location `set` vars shared with `auth_request`) with the location's own match
  semantics; no global `$uri`-keyed maps for per-registration selection; internal
  names from registry values must be injective encodings. A header only a NEW
  template sets/clears is forgeable behind an OLD one: honor it only with the
  marker under a new header name (`X-Validate-Binding-Secret`).
- **After fixing a finding, grep the pattern repo-wide**, because findings usually have
  siblings the report didn't list.

**Standing rule:** when you find or fix a NEW category of security issue, persist
the generalizable lesson in [docs/SECURITY_GUIDELINES.md](docs/SECURITY_GUIDELINES.md)
(one entry per pattern) and, if it's a new category, add a matching line to the
invariants checklist above so it loads every session.

### Input Validation
- Always validate and sanitize user inputs
- Use Pydantic models for request/response validation
- Never trust external data

### Secrets Management
```python
import os
from typing import Optional

def get_secret(key: str, default: Optional[str] = None) -> str:
    """Retrieve secret from environment variable.

    Never hardcode secrets in source code.
    """
    value = os.environ.get(key, default)
    if value is None:
        raise ValueError(f"Required secret '{key}' not found in environment")
    return value
```

### Security Best Practices
- Never log sensitive information (passwords, tokens, PII)
- Use environment variables for configuration
- Validate all inputs, especially from external sources
- Use parameterized queries for database operations
- Keep dependencies updated for security patches

### Security Scanning with Bandit
- Run Bandit regularly as part of the development workflow
- Handle false positives with `# nosec` comments and clear justification
- Common patterns to handle:
  ```python
  # When using random for ML reproducibility (not cryptography)
  # This is not for security/cryptographic purposes - nosec B311
  random.seed(random_seed)
  samples = random.sample(dataset, size)  # nosec B311

  # When loading from trusted sources with version pinning
  # This is acceptable for evaluation tools using well-known datasets - nosec B615
  ds = load_dataset(DATASET_NAME, revision="main")  # nosec B615
  ```
- Run security scans with: `uv run bandit -c pyproject.toml -r registry/ auth_server/`

### Server Binding Security
- When starting a server, never bind it to `0.0.0.0` unless absolutely necessary
- Prefer binding to `127.0.0.1` for local-only access
- If external access is needed, bind to the specific private IP address:
  ```python
  # Bad - exposes to all interfaces
  app.run(host="0.0.0.0", port=8000)

  # Good - local only
  app.run(host="127.0.0.1", port=8000)

  # Good - specific private IP
  import socket
  private_ip = socket.gethostbyname(socket.gethostname())
  app.run(host=private_ip, port=8000)
  ```

### LLM Agent Tool-Execution Safety

When an LLM autonomously emits tool calls (a tool loop, an agent, an A2A server),
the model's output is untrusted: it can be steered by prompt injection in registry
data, upstream tool results, documentation, or inbound messages. System-prompt
`<security>` guidance is NOT an enforcement control: it can be ignored or
overridden. Enforce at the execution boundary instead:

- **Gate every mutating/destructive tool call behind a mandatory human confirmation.**
  Classify each tool invocation read vs. mutate at the point of execution;
  anything not provably read-only is treated as mutating and requires explicit
  approval. Fail closed: if no confirmation channel is available (e.g.
  non-interactive mode), deny the action rather than run it.
- **Deny-by-default executable allowlist for any shell/exec tool.** Permit only a
  fixed set of read-only diagnostic binaries; reject unknown executables,
  path-qualified executables, and shell metacharacters (`;`, `|`, `&`, backticks,
  redirection) before running. Run with a scrubbed environment so read-only
  commands cannot read credential-bearing variables and echo them back.
- **Do not authenticate an agent/tool-loop endpoint by network isolation alone.**
  An A2A / agent HTTP server that drives an LLM tool loop must validate an inbound
  bearer JWT (signature/issuer/expiry/audience against the IdP JWKS) on every
  request before the message reaches the model. Bind to loopback by default;
  require an explicit opt-in to expose on all interfaces. Auth stays enforced
  regardless of bind address, and an unconfigured auth layer denies (never falls
  open). Provide the guard as the shipped default so downstream copies of sample
  agents inherit it.

### Subprocess Security Guidelines

When using the `subprocess` module, follow these security patterns to prevent Bandit B603/B607 findings and avoid shell injection vulnerabilities.

#### ✅ ALWAYS Use List Form (Not String Commands)

```python
# Good - list form prevents shell injection
result = subprocess.run(
    ["nginx", "-s", "reload"],
    capture_output=True,
    text=True,
    timeout=5,
)

# Bad - string form with shell=True is vulnerable to injection
result = subprocess.run("nginx -s reload", shell=True)  # NEVER DO THIS
```

#### ✅ ALWAYS Add Timeout

```python
# Good - prevents DoS from hanging processes
result = subprocess.run(cmd, timeout=30, capture_output=True)

# Bad - no timeout can cause infinite hangs
result = subprocess.run(cmd, capture_output=True)  # Missing timeout!
```

#### ✅ ALWAYS Handle Errors

```python
# Good - proper error handling
try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,  # Raises CalledProcessError on non-zero exit
        timeout=30,
    )
except subprocess.TimeoutExpired:
    logger.error("Command timed out")
    return False
except subprocess.CalledProcessError as e:
    logger.error(f"Command failed: {e.stderr}")
    return False
```

#### ✅ Approved Subprocess Patterns

**Pattern 1: System Utilities (hardcoded commands)**
```python
# System commands with hardcoded paths and flags
result = subprocess.run(
    ["nginx", "-t"],  # nosec B603 B607 - hardcoded command
    capture_output=True,
    text=True,
    timeout=5,
)

result = subprocess.run(
    ["hostname", "-I"],  # nosec B603 B607 - hardcoded command
    capture_output=True,
    text=True,
    timeout=2,
)
```

**Pattern 2: Internal Scripts (controlled paths)**
```python
# Internal scripts with validated arguments
script_path = os.path.join(project_root, "scripts/generate_token.sh")
result = subprocess.run(
    [script_path, validated_arg],  # nosec B603 - hardcoded internal script path
    capture_output=True,
    text=True,
    timeout=30,
    cwd=working_directory,
)
```

**Pattern 3: External Tools (hardcoded flags, data as arguments)**
```python
# External tools with hardcoded flags - user data passed as arguments, not commands
cmd = ["mcp-scanner", "--format", "json", "--url", user_provided_url]
result = subprocess.run(  # nosec B603 - args are hardcoded flags passed to mcp-scanner tool
    cmd,
    capture_output=True,
    text=True,
    check=True,
    timeout=60,
)
```

#### ✅ Security Comment Standards for Subprocess

When suppressing Bandit warnings for subprocess calls, **always include a clear justification**:

```python
# Good - explains why it's safe
subprocess.run(
    ["nginx", "-s", "reload"],
    ...
)  # nosec B603 B607 - hardcoded command

# Good - explains the security model
subprocess.run(
    [script_path, arg],
    ...
)  # nosec B603 - hardcoded internal script path

# Good - explains what's hardcoded
subprocess.run(
    cmd,
    ...
)  # nosec B603 - args are hardcoded flags passed to tool

# Bad - no justification
subprocess.run(cmd, ...)  # nosec B603
```

**Valid Justification Templates:**
- `# nosec B603 B607 - hardcoded command` - for system utilities (nginx, hostname, etc.)
- `# nosec B603 - hardcoded internal script path` - for internal project scripts
- `# nosec B603 - hardcoded internal script path and flags` - when both path and flags are hardcoded
- `# nosec B603 - args are hardcoded flags passed to [tool-name]` - for external tools

#### ❌ NEVER Do These With Subprocess

```python
# NEVER use shell=True with any user input
user_cmd = f"tool --arg {user_input}"
subprocess.run(user_cmd, shell=True)  # VULNERABLE TO INJECTION

# NEVER construct commands from user input
cmd = f"grep {user_search_term} file.txt"  # VULNERABLE
subprocess.run(cmd, shell=True)

# NEVER skip timeout - can hang forever
subprocess.run(["long-running-command"])  # NO TIMEOUT

# NEVER ignore errors without logging
result = subprocess.run(cmd, capture_output=True)
# No error handling - failures go unnoticed
```

### SQL Security Guidelines

When working with databases, follow these patterns to prevent SQL injection vulnerabilities (Bandit B608).

#### ✅ ALWAYS Use Parameterized Queries

```python
# Good - parameterized query with placeholders
cutoff = datetime.now().isoformat()
query = "DELETE FROM table_name WHERE created_at < ?"
cursor.execute(query, (cutoff,))

# Bad - string formatting is vulnerable to SQL injection
cutoff_str = f"'{datetime.now().isoformat()}'"
query = f"DELETE FROM table_name WHERE created_at < {cutoff_str}"  # VULNERABLE
cursor.execute(query)
```

#### ✅ Validate Identifiers Against Allowlists

For table names and column names that cannot be parameterized, use allowlist validation:

```python
# Define allowlists for table and column names
ALLOWED_TABLES = {"users", "metrics", "auth_logs"}
ALLOWED_COLUMNS = {"created_at", "updated_at", "timestamp"}

def validate_table_name(table: str) -> str:
    """Validate table name against allowlist."""
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table: {table}")
    return table

def validate_column_name(column: str) -> str:
    """Validate column name against allowlist."""
    if column not in ALLOWED_COLUMNS:
        raise ValueError(f"Invalid column: {column}")
    return column

# Use validated identifiers with nosec comment
table = validate_table_name(user_provided_table)
column = validate_column_name(user_provided_column)
query = f"SELECT * FROM {table} WHERE {column} = ?"  # nosec B608 - table and column validated against allowlists
cursor.execute(query, (value,))
```

#### ✅ Return Query and Parameters as Tuple

For query-building methods, return both query string and parameters:

```python
def get_cleanup_query(
    table_name: str,
    days: int
) -> tuple[str, tuple]:
    """Get cleanup query and parameters.

    Returns:
        Tuple of (query_string, parameters)
    """
    # Validate table name against allowlist
    table_name = validate_table_name(table_name)

    # Calculate cutoff date
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    # Build parameterized query
    query = f"DELETE FROM {table_name} WHERE created_at < ?"  # nosec B608 - table_name validated against allowlist

    return query, (cutoff,)

# Use the query and parameters
query, params = get_cleanup_query("metrics", 90)
cursor.execute(query, params)
```

#### ✅ Security Comment Standards for SQL

When suppressing B608 warnings, **always document the validation**:

```python
# Good - documents allowlist validation
query = f"SELECT * FROM {table}"  # nosec B608 - table name validated against allowlist
cursor.execute(query, params)

# Good - references validation function
query = f"DELETE FROM {table}"  # nosec B608 - table validated by validate_table_name()
cursor.execute(query, params)

# Good - explains multiple validations
query = f"SELECT {column} FROM {table}"  # nosec B608 - table and column validated against allowlists
cursor.execute(query, params)

# Bad - no justification
query = f"SELECT * FROM {table}"  # nosec B608
cursor.execute(query)
```

**Valid Justification Templates:**
- `# nosec B608 - table name validated against allowlist`
- `# nosec B608 - column name validated against allowlist`
- `# nosec B608 - table and column validated against allowlists`
- `# nosec B608 - identifier validated by _validate_identifier()`

#### ❌ NEVER Do These With SQL

```python
# NEVER use string formatting for values
value = user_input
query = f"SELECT * FROM users WHERE name = '{value}'"  # VULNERABLE TO SQL INJECTION
cursor.execute(query)

# NEVER concatenate user input into queries
query = "SELECT * FROM " + user_table + " WHERE id = " + user_id  # VULNERABLE
cursor.execute(query)

# NEVER skip validation for identifiers
table = request.args.get('table')  # No validation!
query = f"SELECT * FROM {table}"  # VULNERABLE
cursor.execute(query)

# NEVER use datetime() SQL functions with interpolated values
days = user_input
query = f"DELETE FROM t WHERE created_at < datetime('now', '-{days} days')"  # VULNERABLE
cursor.execute(query)
```

### Security Checklist for Code Review

When reviewing code with subprocess or SQL operations, verify:

**Subprocess Checklist:**
- [ ] Using list form (not string commands)
- [ ] No `shell=True` anywhere
- [ ] Timeout specified
- [ ] Error handling includes `TimeoutExpired` and `CalledProcessError`
- [ ] Commands are hardcoded (no dynamic construction from user input)
- [ ] `# nosec` comments include clear justifications
- [ ] Arguments passed as list elements (not interpolated into commands)

**SQL Checklist:**
- [ ] Using parameterized queries for all values
- [ ] Table and column names validated against allowlists
- [ ] No string formatting or concatenation for SQL values
- [ ] Query methods return `tuple[str, tuple]`
- [ ] `# nosec` comments document validation method
- [ ] No datetime() SQL functions with interpolated parameters

## Working With the User

Each rule below is followed by the number of times it had to be repeated across the last
30 days of session transcripts. They are ordered by that count, so the ordering is
evidence rather than preference.

### Run prose you are writing to a file or sending to someone through the writing skill (40)

The most frequent correction in this repo by a wide margin. Invoke the `writing` skill
and run its revision pass before producing: docs, READMEs, release notes, PR and issue
text, commit bodies, design documents, explainers, emails, and customer-facing
bulletins. Then run the linter:

```bash
uv run python scripts/prose-scan.py --strict <file> [<file> ...]
```

It catches the phrase-level tells. The shape-level ones it cannot see still need a read:
parallel sentence structure, paragraph pinning, summary beats, setup/payoff pairs,
teaser openers that announce a count, appended reassurance clauses, decorative bolding.

Scope: this is a gate on artifacts, not on conversation. A chat answer should be plain
and free of the obvious tells, but it does not need the full pass.

Never use an em-dash, or a double hyphen standing in for one, in any file.

### Answer or draft first; do not execute yet (31)

A question is a request for an answer, not a trigger to act. When the user asks "can we",
"should we", "what are my options", or "why is", answer and stop. When they ask for the
text of an issue, comment or email, produce the text and wait. "Propose", "draft" and
"just answer" are explicit, but the default holds without them.

### Update the docs in the same change (13)

A behavior change is not done when the code works. Update the markdown that describes
the old behavior in the same commit: the feature's own docs page,
`docs/unified-parameter-reference.md` for any parameter, `api/openapi.json` for any
route. Never announce a feature in `README.md`: it has a CI-enforced line budget and is
rotated only by the `release-notes` skill.

### Frame issues and customer communication softly (10)

For work that improves something already working, say what is being optimized, not that
the current implementation is defective. Reserve defect language for defects. In
customer-facing security text keep the register clinical and match the existing
bulletins. Be precise about scope: "org admin" is not "admin".

### Do not start a server, and do not leave one running (6)

Generated HTML here is self-contained and needs no server. Offer the command and let the
user run it, which is also what makes VS Code forward the port. If you do start one,
report the PID. Two servers from earlier sessions were found still listening days later.

### Look it up in the code before asking (3)

If the answer is in the repository, read it. Do not ask the user what a function returns
or where a value is configured. The same applies outside the team: ask a customer only
for what we cannot determine ourselves, and never ask them to choose a mechanism that is
ours to define.

### Explain at the level asked (5)

Asked to explain simply, or at "100 level": plain language, no line numbers, no code,
and why it matters before how it works. If the user says they still do not understand,
the explanation was wrong. Rewrite it rather than restating it at greater length.

### Do not poll a background task

`sleep 300; tail ...` hits the foreground timeout and fails. The harness re-invokes on
completion.

## Finding a Deployment to Debug Against

Before guessing at a symptom, find the running deployment and read from it. Neither
source below is in git, so both describe the machine you are on rather than the repo.

### Amazon ECS

`terraform/aws-ecs/terraform.tfstate` is the inventory of the deployed stack. It is
gitignored (`terraform/aws-ecs/.gitignore` matches `*.tfstate*`), so it exists only on a
machine that has run `terraform apply`, and it is the fastest way to learn cluster names,
service names, log groups, load balancer and CloudFront hostnames, and which secrets
exist.

**It also contains plaintext secrets**: 14 `aws_secretsmanager_secret_version` resources,
6 `aws_ssm_parameter`, and 5 `random_password`. Query it for the keys you need. Never
`cat` it, never paste its contents into a file, a PR, an issue, or a chat message, and
never serve the directory over HTTP.

Pull just the identifiers:

```bash
python3 - <<'EOF'
import json
state = json.load(open("terraform/aws-ecs/terraform.tfstate"))
def names(kind):
    return [
        inst["attributes"].get("name")
        for res in state["resources"] if res.get("type") == kind
        for inst in res.get("instances", [])
    ]
print("clusters:   ", names("aws_ecs_cluster"))
print("services:   ", names("aws_ecs_service"))
print("log groups: ", names("aws_cloudwatch_log_group"))
EOF
```

On the current deployment that yields the `mcp-gateway-ecs-cluster` and `keycloak`
clusters, services named `mcp-gateway-v2-registry`, `-auth`, `-mcpgw`,
`-metrics-service` and `-grafana`, and log groups under `/ecs/mcp-gateway-v2-*`. Treat
those as examples: read the state rather than hardcoding them, because a differently
named deployment is normal.

Then go to the logs and the task state:

```bash
aws logs tail /ecs/mcp-gateway-v2-registry --since 30m --follow
aws ecs describe-services --cluster mcp-gateway-ecs-cluster --services mcp-gateway-v2-registry \
  --query 'services[0].{desired:desiredCount,running:runningCount,events:events[:5]}'
aws ecs describe-tasks --cluster mcp-gateway-ecs-cluster \
  --tasks $(aws ecs list-tasks --cluster mcp-gateway-ecs-cluster \
            --service-name mcp-gateway-v2-registry --query 'taskArns[0]' --output text) \
  --query 'tasks[0].containers[].{name:name,status:lastStatus,reason:reason}'
```

A task that will not start usually explains itself in `stoppedReason` or in the last few
`events` entries, before any application log is written.

### Local Docker Compose

The local stack serves the gateway on `http://localhost` (port 80 maps to the registry's
8080) and everything else on loopback-only ports. Start from what is actually running:

```bash
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'
docker compose logs -f --tail 100 registry
docker compose exec registry sh -c 'grep -n -A3 "<server-path>" /etc/nginx/conf.d/*.conf'
```

Container names are prefixed `mcp-gateway-registry-`, so the registry is
`mcp-gateway-registry-registry-1`. A `(healthy)` marker comes from the container's own
health check, which calls the backend directly on 127.0.0.1 and therefore says nothing
about whether the nginx proxy path works. Test that through `http://localhost`.

Quick reachability:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost/health
```

### Tokens for a local request

`.token` at the repo root holds a bearer token for the local stack. It is nested JSON, so
the token is at `.tokens.access_token`, not at the top level:

```bash
export ACCESS_TOKEN=$(jq -r '.tokens.access_token' .token)
curl -sS http://localhost/api/servers -H "Authorization: Bearer $ACCESS_TOKEN" | jq
```

A token is signed with the deployment's own `SECRET_KEY`, so a token minted locally
returns 401 against ECS and vice versa. When a request 401s, check which deployment the
token came from before debugging the auth code.

## Component Map: What to Run After Changing What

Canonical commands live in `.github/workflows/`, `pyproject.toml` and
`.pre-commit-config.yaml`. Where an example here disagrees with those, they win.

| Changed | Run |
|---------|-----|
| `registry/`, `auth_server/` | `uv run python scripts/test.py coverage -n 8`; re-read the security invariants above |
| `frontend/` | `npm run lint && npm run build` in `frontend/` |
| `charts/` | `helm unittest charts/mcpgw charts/auth-server charts/registry charts/mcp-gateway-registry-stack`, then `helm dep update` on the stack chart if a subchart changed |
| `terraform/` | `terraform fmt -check && terraform validate`, then `terraform plan` |
| `api/` routes | Refresh `api/openapi.json` (see below). Update `api/registry_client.py` and `api/registry_management.py` |
| `scripts/`, any shell | `bash -n <file>`; `uv run python -m py_compile <file>` for Python |
| Any config parameter | All three deployment surfaces plus `docs/unified-parameter-reference.md` |
| Any markdown | `uv run python scripts/prose-scan.py --strict <file>` |

### Generated artifacts

Never hand-edit these, and never stage one you did not deliberately regenerate:

| Artifact | Regenerate with |
|----------|-----------------|
| `uv.lock` | `uv lock`. `uv run` rewrites it as a side effect; discard that churn |
| `frontend/package-lock.json` | `npm install`. Same churn caveat |
| `charts/*/charts/*.tgz` | `helm dep update` |
| `api/openapi.json` | The procedure below. Nothing generates it automatically and no CI job checks it |
| `frontend/dist/` | `npm run build` |

### Rendering untrusted content

Anything that renders content fetched from GitHub, a registrant, or a customer into HTML
treats it as hostile: no raw HTML passthrough, escape before interpolating, allowlist URL
schemes on every href, and refuse an SVG carrying a script, an event handler or an
external reference. `scripts/render-doc-html.py` is the worked example. The same applies
to terminal output: never echo fetched text containing escape sequences straight to a
terminal.

## Development Workflow

### Recommended Development Tools
- **Ruff**: For linting and formatting (replaces multiple tools like isort and many flake8 plugins)
- **Bandit**: For security vulnerability scanning
- **MyPy**: For type checking
- **Pytest**: For testing

### Pre-commit Workflow

#### Option 1: Automated Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically run checks before each commit:

```bash
# Install pre-commit (one-time setup)
uv pip install pre-commit

# Install the git hooks (one-time per repo clone)
pre-commit install

# Now all checks run automatically on git commit
git add file.py
git commit -m "Your message"  # Hooks run automatically

# Run hooks manually on all files
pre-commit run --all-files
```

**What runs automatically:**
- ✅ Ruff linter with auto-fixes
- ✅ Ruff formatter (PEP 604/585 modernization)
- ✅ Trailing whitespace removal
- ✅ End-of-file fixes
- ✅ YAML/JSON validation
- ✅ Bandit security scan
- ✅ MyPy type checking
- ✅ Fast unit tests
- ✅ Python/shell syntax checks

#### Option 2: Manual Workflow

Before committing code, run these checks in order:

```bash
# 1. Format and lint with auto-fixes
uv run ruff check --fix . && uv run ruff format .

# 2. Security scanning
uv run bandit -c pyproject.toml -r registry/ auth_server/

# 3. Type checking
uv run mypy registry/ auth_server/ metrics-service/

# 4. Run tests
uv run pytest

# Or run all checks in one command:
uv run ruff check --fix . && uv run ruff format . && uv run bandit -c pyproject.toml -r registry/ auth_server/ && uv run mypy registry/ auth_server/ metrics-service/ && uv run pytest
```

### Code Formatting Standards

**Ruff Configuration**: This project uses ruff for formatting with the following key settings (see `pyproject.toml`):

- **Target Python**: see `pyproject.toml`, which is the source of truth
- **Line Length**: 100 characters
- **Type Hint Modernization**: Automatic via ruff rules:
  - `UP006`: Use PEP 585 built-in generics (`list`, `dict`, `tuple`)
  - `UP007`: Use PEP 604 union syntax (`X | Y` instead of `Union[X, Y]`)
  - `UP037`: Remove quotes from type annotations
  - `I001`: Auto-sort imports (isort compatible)

**Formatting automatically handles:**
- Type hint modernization (PEP 604/585)
- Import organization (stdlib, third-party, local)
- Trailing whitespace removal
- Consistent indentation (4 spaces)
- Line length enforcement
- Docstring formatting

**Example ruff modernizations:**
```python
# Before ruff format
from typing import Optional, List, Dict
def func(x: Optional[List[Dict]]) -> Optional[str]: pass

# After ruff format (automatic)
def func(x: list[dict] | None) -> str | None: pass
```

### Adding Development Dependencies
```bash
# Add development dependencies
uv add --dev ruff mypy bandit pytest pytest-cov pre-commit
```

## Dependency Management

### Project Configuration
Always specify Python version in `pyproject.toml` to avoid warnings:
```toml
[project]
name = "project-name"
version = "0.1.0"
description = "Project description"
requires-python = ">=3.14"  # Always specify this!
dependencies = [
    # ... dependencies
]
```

### Version Pinning
In `pyproject.toml`:
```toml
[project]
dependencies = [
    "fastapi>=0.100.0,<0.200.0",  # Minor version flexibility
    "pydantic==2.5.0",  # Exact version for critical dependencies
    "polars>=0.19.0",  # Minimum version only
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "bandit>=1.7.0",
]
```

### Dependency Guidelines
- Pin exact versions for critical dependencies
- Use version ranges for stable libraries
- Separate dev dependencies from runtime dependencies
- Regularly update dependencies for security patches
- Document why specific versions are pinned

## Project Structure

### Standard Layout

This repo keeps its Python packages at the root (`registry/`, `auth_server/`, `api/`,
`cli/`, `metrics-service/`) with no `src/` directory. Match the surrounding tree when
adding to an existing package. For a new sub-project, use `src/<name>/` with `models/`,
`services/`, `api/`, `utils/`, and a sibling `tests/` split into `unit/` and
`integration/`.

### Module Organization
- Keep related functionality together
- Use clear, descriptive module names
- Avoid circular imports
- Keep modules focused on a single responsibility

### Comprehensive .gitignore

The repo's `.gitignore` already covers the usual Python, virtualenv, tooling-cache,
IDE and OS entries, plus `.scratchpad/`, `logs/` and `.aws/`. Add to it rather than
rewriting it.

**Never stage a lockfile you did not deliberately change.** `uv run` rewrites
`uv.lock` as a side effect (dropping `exclude-newer`, adding platform markers), and
`frontend/package-lock.json` churns similarly. Both show as modified after ordinary
work. Stage files by name; never `git add -A` or `git add .`.

## Scratchpad for Planning & Design

`.scratchpad/` at the repo root holds the intermediate documents of active work:
design sketches, analyses, task plans, draft issues and posts, session notes. It is in
`.gitignore`, so nothing written there can be committed, and nothing there is suitable
as long-term documentation.

Default to writing plans and analyses as files here rather than into chat. Name them so
the subject is obvious: `plan-<feature>.md`, `analysis-<topic>.md`,
`design-<feature>.md`, `draft-github-issue.md`, `session-notes-YYYY-MM-DD.md`. Add a
date when a second version will follow. Per-issue and per-PR work goes in
`.scratchpad/issue-NNNN/` and `.scratchpad/pr-NNNN/`. Each file should stand alone: a
heading, a created date, and enough context to still read a month later.

**`.scratchpad/` also holds credential files** (`.hftoken`, `.oai`, `.bedrock`,
`.gh-client-id-secret`). Never serve the directory over HTTP and never point a preview
server at it; point at the single document's folder instead.

## Environment Configuration

Configuration is a Pydantic `Settings` class in `registry/core/config.py`, one typed
field per variable, with a description explaining why the field exists and what the
default means. Ship every new variable in `.env.example`, never commit `.env`, and give
a default wherever one is safe. A required secret has no default: it raises.

## Data Validation with Pydantic

Every API request and response body is a Pydantic model. Put constraints in
`Field(...)` rather than in handler code, use a validator for cross-field rules, and
add a `json_schema_extra` example so the generated OpenAPI docs are usable. Validation
errors must name the offending field.

## Platform Naming
- Always refer to the service as "Amazon Bedrock" (never "AWS Bedrock")

## GitHub Commit and Pull Request Guidelines
- Never include auto-generated messages like "🤖 Generated with [Claude Code]"
- Never include "Co-Authored-By: Claude <noreply@anthropic.com>"
- Keep commit messages clean and professional
- When creating pull requests, do not include Claude Code attribution or generation messages
- Pull request descriptions should be professional and focus on the technical changes

## Documentation Guidelines
- Never add emojis to README.md files in repositories
- Keep README files professional and emoji-free
- Never hard word-wrap Markdown files at 76 characters (or any fixed column). Write one sentence or paragraph per line and let the editor soft-wrap. Hard wrapping creates noisy diffs and breaks tables, lists, and links.

### Emoji Usage Guidelines
- **Code**: Absolutely no emojis in source code, comments, or docstrings
- **Documentation**: Avoid emojis in all documentation files (.md, .rst, etc.)
- **Log Messages**: Use plain text only for log messages - no emojis
- **Shell Scripts**: Avoid emojis in shell scripts - prefer plain text status messages
- **Comments**: Use clear, descriptive text instead of emojis in code comments

**Rationale**: Emojis can cause encoding issues, reduce accessibility, appear unprofessional in enterprise environments, and may not render consistently across different systems and terminals.

### README Best Practices
A well-structured README should include:

1. **Prerequisites Section**: List external dependencies and setup requirements
   ```markdown
   ## Prerequisites
   - Python 3.14+
   - AWS credentials configured
   - Amazon Bedrock Guardrail with sensitive information filters
   ```

2. **Links to External Resources**: Provide links to datasets, documentation, and services
   ```markdown
   - Evaluate performance on the [dataset-name](https://link-to-dataset)
   - See [AWS documentation](https://docs.aws.amazon.com/...) for setup
   ```

3. **Clear Command Examples**: Show all command-line options with examples
   ```markdown
   ## Usage
   # Basic usage
   uv run python -m module_name --required-param value

   # With all options
   uv run python -m module_name --param1 value1 --param2 value2

   # Using environment variables
   export CONFIG_VAR=value
   uv run python -m module_name
   ```

4. **Development Workflow**: Include a section on development practices
   ```markdown
   ## Development Workflow
   # Run all checks before committing
   uv run ruff check --fix . && uv run ruff format . && uv run bandit -c pyproject.toml -r registry/ auth_server/
   ```

5. **Performance Warnings**: Alert users about time-intensive operations
   ```markdown
   # Evaluate full dataset (warning: this may take a long time)
   uv run python -m module_name --sample-size 0
   ```

## Helm Chart Development

The Helm charts under `charts/` have `helm-unittest` suites in each chart's `tests/` directory. The plugin is installed locally (`helm plugin list` shows `unittest`).

### Running the suites

```bash
# All four charts (subcharts + parent stack)
helm unittest charts/mcpgw charts/auth-server charts/registry charts/mcp-gateway-registry-stack
```

For stack-level tests that render via subchart dependencies, run `helm dep update charts/mcp-gateway-registry-stack` after editing any subchart so the packaged `.tgz` files in `charts/mcp-gateway-registry-stack/charts/` pick up your changes.

### Tests are part of the contract

Any change to a Helm chart must keep the unittest suites passing, and new chart functionality must come with new unittest coverage. Things worth testing:
- New values.yaml fields that affect rendered output (happy path + edge cases).
- New `fail` guards or validation helpers (assert the failure mode with `failedTemplate: errorPattern: ...`).
- Changes to `env:` / `envFrom:` ordering, resource refs, or conditional blocks.

Prefer `equal:` over `contains:` when asserting something must appear in a specific position, since `contains:` only checks membership and will not catch ordering regressions.

### Keeping index-based and reserved-name assertions in sync

Two places encode assumptions about the current set of chart-managed env vars. Both need updating whenever `env:` entries, `envFrom:` sources, or secret/configmap keys are added, removed, or reordered in a deployment template:

1. **Reserved-name lists in `charts/<subchart>/reserved-env-names.txt`.** These are the union of every name the chart's deployment can render into `env:` (across all conditional branches) plus every key consumed via `envFrom`. Over-rejection is preferred to under-rejection. If you add a new env var, a new secret key, or wire up a new configmap/secret via `envFrom`, extend the corresponding chart's `.txt` file and run `helm dep update` on any parent chart that depends on the subchart. Edit the `.txt`, not `templates/_helpers.tpl`: the helper there (`{{ define "<chart>.reservedEnvNames" }}`) only loads the file via `.Files.Get`. Its block comment lists the sources the list is cross-referenced against, numbered to match the file; update that comment when the set of sources changes.

2. **Index-based order assertions in the `tests/extra_env_test.yaml` suites** (e.g. `spec.template.spec.containers[0].env[6].name: KEYCLOAK_ADMIN_PASSWORD`). These hardcode the position where chart-managed entries end and user-supplied `extraEnv` entries begin. They come with `NOTE:` comments describing the conditional values they assume (e.g. `global.authProvider.type=keycloak`, `awsRegistry.federationEnabled=false`). If you insert a new conditional block before the `extraEnv` append, update the expected indices, and update the `NOTE:` comment to describe the new assumption.

If you only update one of the two, the unittest suite catches it: a drifting reserved list lets a rejection test fail (the catastrophic key isn't rejected anymore), and drifting order assertions fail the positional tests. Both run in the `unittest` job of `.github/workflows/helm-test.yml` on any PR that touches `charts/**`.

Note: only the unittest suite enforces these invariants. `helm lint`, `helm template`, and `kubeconform` do not. If you add a new env var to a deployment but forget to update the reserved list AND nobody writes an extraEnv test for that name, nothing will detect the gap. Treat the reserved list and the deployment template as a matched pair.

## Regenerating `api/openapi.json`

`api/openapi.json` is a **hand-refreshed artifact**. No script generates it and no CI job checks it, so it silently goes stale: when PR #1711 shipped a new endpoint, the committed spec was also still missing audit-endpoint changes from three earlier merged PRs. Refresh it whenever a release adds or changes a route.

Regenerate from a running registry and re-apply the release version:

```bash
# The container MUST be built from the commit you are documenting (see below).
curl -s http://localhost/openapi.json > /tmp/live.json

python3 - <<'PY'
import json, pathlib
spec = json.load(open("/tmp/live.json"))
spec["info"]["version"] = "1.31.0"          # the release this lands in
pathlib.Path("api/openapi.json").write_text(json.dumps(spec, indent=2) + "\n")
PY
```

Four things get this wrong, in rough order of how easy they are to miss:

1. **Do not pass `ensure_ascii=False`.** The committed file escapes non-ASCII as `\uXXXX`. Writing literal characters instead rewrites every em-dash in every docstring, turning a 9-line change into 56 insertions and 28 deletions and burying the real diff. `json.dumps` defaults to `ensure_ascii=True`, so just leave it alone. Use `indent=2` and a trailing newline to match.

2. **Set `info.version` by hand.** The live app reports a git-describe development string (`1.30.0-46-gcb189fa7-feat/some-branch`), never a release version. The artifact carries the clean semver of the release it ships in, as every prior refresh did (1.24.7, 1.24.8, 1.31.0).

3. **Build the container from the commit you are documenting.** A spec generated from a stale build documents code that is not on `main`. If the running build is behind, either rebuild or prove the gap cannot matter: confirm no changed file altered a route decorator, handler signature, `response_model` or `status_code`, since those are what FastAPI derives the schema from. A changed handler *body* cannot affect the spec.

4. **Diff semantically, not by eye.** An 824 KB JSON file hides accidental removals. Enumerate what moved before committing:

```bash
python3 -c "
import json, subprocess
old = json.loads(subprocess.run(['git','show','HEAD:api/openapi.json'],capture_output=True,text=True).stdout)
new = json.load(open('api/openapi.json'))
op, np = set(old['paths']), set(new['paths'])
print('paths added:  ', sorted(np-op))
print('paths removed:', sorted(op-np))
print('paths changed:', [p for p in sorted(op&np) if old['paths'][p] != new['paths'][p]])
osc, nsc = old.get('components',{}).get('schemas',{}), new.get('components',{}).get('schemas',{})
print('schemas added/removed:', sorted(set(nsc)-set(osc)), sorted(set(osc)-set(nsc)))
"
```

Expect only the paths your release touched. Anything removed is a regression: it usually means the container was built with a feature flag off or in the wrong `DEPLOYMENT_MODE`, so a whole router did not register.

## Docker Build and Deployment

Build scripts live in `scripts/`. Follow the existing ones rather than inventing a
new shape:

- `set -e` at the top, so a failed step stops the script.
- Resolve paths from `${BASH_SOURCE[0]}`, never from the caller's working directory.
- Configuration from environment variables with defaults
  (`AWS_REGION="${AWS_REGION:-us-east-1}"`).
- Derive the account id (`aws sts get-caller-identity`) rather than hardcoding it.
- Log in to ECR, and create the repository if `aws ecr describe-repositories` fails.
- Plain-text progress messages, no emojis.
- Write the pushed image URI to a file so later scripts can read it.
- **Tag with an immutable version, not `:latest`.** The security invariants require
  pinned image tags, and a deployment that reads `:latest` cannot be rolled back to a
  known build. Push the version tag; add `:latest` only as an extra alias if something
  genuinely needs it.

For an ARM64 build, register the QEMU handlers first
(`docker run --rm --privileged multiarch/qemu-user-static --reset -p yes`).

Run `bash -n <script>` after editing any shell script.

## GitHub Issue Management

### Label Management Best Practices
When creating GitHub issues:

1. **Check Available Labels First**: Always get a list of available labels for the repository before creating issues
   ```bash
   gh label list
   ```

2. **Use Only Existing Labels**: Only apply labels that already exist in the repository to avoid errors during issue creation

3. **Suggest New Labels**: If you believe a new label would be beneficial, make a suggestion in the issue description or as a separate comment, but don't attempt to add non-existent labels during issue creation

4. **Label Application**: Apply labels that are available and relevant to the issue type and scope

**Example Workflow**:
```bash
# First check available labels
gh label list

# Create issue with only existing labels
gh issue create --title "..." --body-file "..." --label "enhancement,bug"

# If new labels are needed, suggest them in issue comments
gh issue comment 123 --body "Suggest adding 'agentcore' label for AgentCore-related issues"
```

## Deployment Surface Customization

### Helm Charts

Helm charts support `extraEnv` for injecting custom environment variables. The reserved names are read from `charts/*/reserved-env-names.txt` files using `.Files.Get` and `splitList` functions.

**Example:**
```yaml
extraEnv:
  - name: MY_CUSTOM_VAR
    value: "my-value"
  - name: MY_FLAG
    value: "true"
```

**Reserved Names:** The following variables are managed by the deployment and cannot be overridden:
- `DEPLOYMENT_MODE`, `REGISTRY_MODE`, `SECRET_KEY`, etc.

See `charts/*/reserved-env-names.txt` for the complete list per service.

### Docker Compose

For Docker Compose deployments, you can inject custom environment variables by creating files in `extra_env/` at the repo root (next to `.env`):

```bash
# Create the extra_env directory at the repo root
mkdir -p extra_env

# Create your environment file
cat > extra_env/registry.env << EOF
# Custom environment variables for the registry
MY_FEATURE_FLAG=true
CUSTOM_TIMEOUT=30
EOF

# Start the stack
./build_and_run.sh
```

The `env_file:` entries are configured for `registry.env`, `auth-server.env`, and `mcpgw.env` with `required: false`, so missing files are allowed.

**Overriding the path:** Both the preflight validator and the compose files resolve the extra_env directory from `$MCP_EXTRA_ENV_DIR`, falling back to `./extra_env` (relative to the repo root) when unset. Export `MCP_EXTRA_ENV_DIR` before running `build_and_run.sh` (or `docker compose` directly) to point both surfaces at an alternative location, for example a tmpfs for local testing or `/etc/mcp-gateway/extra_env` for a shared host deployment.

**Compose version requirement:** The `env_file: [{ path, required }]` object form requires **Docker Compose v2.24 or newer** (Docker Desktop 4.27+) or **Podman Compose v1.0.7+**. Older versions silently ignore the `required` flag and will error out if the referenced file does not exist. Check with `docker compose version` or `podman-compose --version` before deploying.

**Preflight Validation:** Before starting containers, `build_and_run.sh` validates that no environment variable in your extra_env files conflicts with chart-managed reserved names. If a collision is found, deployment fails with a clear error message pointing to the exact line number and file. The validator also warns on malformed lines (missing `=` or empty keys), normalizes names to upper-case before comparing (so `secret_key=foo` is still caught), and logs the per-service custom-variable count at startup. The same validator lives at `scripts/validate-extra-env.sh` and can be run standalone for CI or pre-commit use.

### Terraform / ECS

For Terraform/ECS deployments, add `*_extra_env` variables to your `terraform.tfvars`:

```hcl
registry_extra_env = [
  { name = "MY_FEATURE_FLAG", value = "true" },
  { name = "CUSTOM_TIMEOUT", value = "30" },
]

auth_server_extra_env = [
  { name = "MY_AUTH_VAR", value = "value" },
]

mcpgw_extra_env = [
  { name = "MY_MCPGW_VAR", value = "test" },
]
```

**Validation:** Terraform validates that none of the names conflict with reserved variables at `terraform plan` time using `file()` and `contains()` functions.

## Federated Registry Implementation Workflow

These gates apply to any multi-step feature. Before calling a sub-feature done:

- Every acceptance criterion is verified by a test, not by inspection.
- No `TODO` or `FIXME` left behind.
- The code compiles and lints clean, and the existing suite still passes.
- The design document is updated if implementation revealed new scope.

For the design, review and testing structure itself, use the `new-feature-design`
skill; for review, use `pr-review`.
