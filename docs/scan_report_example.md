# MCP Server Security Scan Report

**Scan Date:** 2025-10-21 23:50:03 UTC
**Analyzers Used:** yara

## Executive Summary

- **Total Servers Scanned:** 4
- **Passed:** 3 (75%)
- **Failed:** 1 (25%)

### Aggregate Vulnerability Statistics

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 1 |
| Medium | 0 |
| Low | 0 |

## Per-Server Scan Results

### io.mcpgateway/currenttime

- **URL:** `https://mcpgateway.ddns.net/currenttime/mcp`
- **Status:** ✅ SAFE

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 0 |

### io.mcpgateway/mcpgw

- **URL:** `https://mcpgateway.ddns.net/mcpgw/mcp`
- **Status:** ❌ UNSAFE

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 1 |
| Medium | 0 |
| Low | 0 |

#### Detailed Findings

**Tool: `healthcheck`**

- **Analyzer:** yara_analyzer
- **Severity:** HIGH
- **Threats:** INJECTION ATTACK
- **Summary:** Detected 1 threat: sql injection

**Taxonomy:**
```json
{
  "scanner_category": "INJECTION ATTACK",
  "aitech": "AITech-9.1",
  "aitech_name": "Model or Agentic System Manipulation",
  "aisubtech": "AISubtech-9.1.4",
  "aisubtech_name": "Injection Attacks (SQL, Command Execution, XSS)",
  "description": "Injecting malicious payloads such as SQL queries, command sequences, or scripts into MCP servers or tools that process model or user input, leading to data exposure, remote code execution, or compromise of the underlying system environment."
}
```

<details>
<summary>Tool Description</summary>

```
Retrieves health status information from all registered MCP servers via the registry's internal API.

Returns:
    Dict[str, Any]: Health status information for all registered servers, including:
        - status: 'healthy' or 'disabled'
        - last_checked_iso: ISO timestamp of when the server was last checked
        - num_tools: Number of tools provided by the server

Raises:
    Exception: If the API call fails or data cannot be retrieved
```
</details>

**Error:** Scanner exit code: 1

### io.mcpgateway/realserverfaketools

- **URL:** `https://mcpgateway.ddns.net/realserverfaketools/mcp`
- **Status:** ✅ SAFE

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 0 |

### io.mcpgateway/sre-gateway

- **URL:** `https://mcpgateway.ddns.net/sre-gateway/mcp`
- **Status:** ✅ SAFE

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 0 |

**Error:** Scanner exit code: 1

---

*Report generated on 2025-10-21 23:50:03 UTC*
