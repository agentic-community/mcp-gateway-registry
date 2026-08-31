---
name: debug
description: Debug issues in the MCP Gateway Registry using first-principles thinking. Invoke when something is broken, timing out, returning errors, or behaving unexpectedly. Forces structured root-cause analysis before any code change is proposed.
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.0"
---

# Debug Skill

Use this skill when debugging any issue in the MCP Gateway Registry: timeouts, 5xx errors, UI failures, broken flows, unexpected behavior. Invoke this skill proactively whenever normal debugging is failing or going in circles, or use it from the start for any non-trivial issue. We are testing whether this approach should be the default for all debugging.

## Expert Personas

Before starting, announce that you are consulting the following expert panel. Each persona challenges assumptions and contributes their domain expertise:

- **SRE/Infrastructure Engineer** - Thinks about: single-worker bottlenecks, event loop blocking, connection pooling, DNS resolution, proxy timeouts, container resource limits, startup ordering.
- **Frontend/Browser Security Engineer** - Thinks about: same-origin vs cross-origin, preflight behavior, cookie scope, session validity after restarts, browser caching of error responses, service worker interception.
- **Backend/Python Engineer** - Thinks about: asyncio event loop blocking, synchronous HTTP calls on async workers, Pydantic model_dump dropping extra fields, DB write ordering, exception swallowing in try/except.
- **Senior Staff Engineer (Skeptic)** - Questions EVERY theory before implementation. Asks: "If that were true, why does X work?" and "Was this working before our change?" Forces the team to prove hypotheses before writing code.

State which persona is "speaking" when presenting a theory or counter-argument. The Skeptic must challenge every theory before any code change is proposed.

## First Principles Debugging

### Step 1: Reproduce and observe (do NOT theorize yet)

- What exact error does the user see? (HTTP status, error message, timing)
- What was the user doing? (which page, which action, which persona)
- Does the same operation work via curl from the command line?
- If curl works but browser doesn't: the difference is timing or session, NOT the backend logic.

### Step 2: Check what the worker is DOING right now

This is the single most important step. The registry runs a **single uvicorn worker**. If a request times out, the worker is busy doing something else.

```bash
docker compose logs registry --since=30s 2>&1 | tail -30
```

Read what the logs say at the EXACT timestamp of the failure. Do NOT filter by keywords related to your bug. Look at EVERYTHING the worker is doing. Common blockers:
- GoDaddy ANS API pagination (hundreds of sequential HTTP GETs)
- Federation sync (AgentCore, peer registries)
- Health checks pinging 30+ MCP servers
- Vector index processing (embedding generation, index updates)
- Security scans (yara, spec analyzers take 3-8 seconds each)

If the worker is busy with a background task, your request is queued. The fix is to make that task non-blocking, NOT to change the endpoint code.

### Step 3: Is this new or pre-existing?

Before touching ANY code, answer:
- Was this working before on the current branch?
- Was this working on `main` / the last release tag?
- Did a recent restart, rebuild, or config change cause it?

If the issue exists on the baseline tag with zero changes, it is pre-existing and unrelated to whatever PR you are working on. Do NOT fix pre-existing issues in a PR scoped to something else unless the user explicitly asks.

### Step 4: If you see something suspicious in the logs, ask HOW it connects

When you see an error or warning in the logs, do NOT immediately assume it is the root cause. Ask:
- Is this error on the code path of the failing request, or is it a different background operation?
- Does this error BLOCK the worker, or is it caught and handled?
- Would fixing this error actually change the user-visible symptom?

Example: seeing `FAISS index not initialized` in logs does NOT mean FAISS is the problem. It might be a non-fatal warning from a background task that runs in parallel. Trace the actual call stack of the failing request.

### Step 5: Before proposing a code change, answer these questions

1. **Is a code change actually needed?** Could this be a configuration issue (.env, timeout, feature flag)?
2. **Present options to the user before implementing.** Do NOT just pick a fix and implement it. Present 2-3 well-thought-out options with trade-offs (complexity, risk, scope, reversibility) and let the user choose. One-line fixes are preferred over architectural changes, but the user decides.
3. **Does this fix have side effects?** Does it change behavior for OTHER callers, OTHER deployments, OTHER code paths?
4. **Will hot-patching work?** If you docker-cp a file into a running container:
   - The restart will invalidate sessions (auth-server restart = all browser cookies invalid)
   - nginx_service.py will re-render the nginx config on startup (may break routing)
   - Startup tasks will re-run (may block the worker for 30+ seconds)
   - Consider whether a simple rebuild is faster than iterative hot-patching.
5. **Does this need a test?** If you are changing backend logic, YES. Always.

### Step 6: Cross-question your own theory

Before implementing, argue against your own hypothesis:
- "If CORS is the problem, why do all the GETs work from the same browser?"
- "If the scan timeout is the problem, why does curl finish in 5 seconds?"
- "If nginx is blocking the request, why is there no error in the nginx error log?"
- "If the session is invalid, why did the dashboard load successfully moments ago?"

If you cannot answer the counter-question, your theory is wrong. Go back to Step 2.

## Anti-patterns to avoid

- **Do NOT chase CORS on same-origin requests.** If the browser URL and the API URL have the same scheme + host + port, CORS does not apply. Period.
- **Do NOT hot-patch containers repeatedly.** Each restart creates new state. After 2 failed hot-patches, do a clean `build_and_run.sh` and test once.
- **Do NOT make async/background changes to "fix" a timeout** unless you have proven the timeout is caused by the specific function you are making async. The real cause is usually a DIFFERENT function blocking the worker.
- **Do NOT add CORS, nginx, or auth-server changes** without first confirming the request actually reaches (or fails to reach) those layers. Check logs at each layer in order: nginx access log → auth-server validate → registry endpoint.
- **Do NOT assume "no logs = request didn't reach the backend."** It might mean the worker is busy processing something else and your request is queued behind it.

## Debugging checklist (copy into your response)

When debugging, paste this and fill it in:

```
[ ] Exact error: ___
[ ] Same operation via curl: works / fails / different error
[ ] Registry logs at failure timestamp show: ___
[ ] Worker was busy doing: ___ (or idle)
[ ] Pre-existing on baseline? yes / no / untested
[ ] Theory: ___
[ ] Counter-argument against theory: ___
[ ] Theory survives counter-argument? yes / no
```

Only propose a fix after all boxes are checked and the theory survives.
