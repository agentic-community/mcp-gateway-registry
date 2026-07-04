import type {AgentToolInvocation} from "./tools.js";

/**
 * Tool-execution safety policy for the agent loop.
 *
 * The LLM emits tool_use blocks autonomously across many rounds, driven in part
 * by untrusted context (registry data, upstream tool results, documentation).
 * A system-prompt instruction is NOT enforcement: it can be ignored, hallucinated
 * around, or overridden by prompt injection. Enforcement therefore lives here, at
 * the execution boundary, and holds regardless of what the model emits.
 *
 * Two independent controls:
 *   1. A deny-by-default executable allowlist for `shell_command` — only
 *      read-only diagnostic binaries may run, and only with safe arguments.
 *   2. A mandatory human-confirmation gate for every mutating action
 *      (all shell commands, and any registry_task / mcp_command classified as
 *      state-changing).
 *
 * Both fail closed: an unknown executable is rejected, and a gated action with no
 * confirmation handler (or a declined prompt) is denied.
 */

/**
 * Read-only diagnostic executables permitted for `shell_command`.
 *
 * Deny-by-default: anything not listed here is rejected outright, before any
 * confirmation prompt. Keep this to inspection/diagnostic binaries only — no
 * package managers, no interpreters, no editors, no privilege tools, nothing that
 * writes, deletes, or reaches the network.
 */
export const ALLOWED_SHELL_EXECUTABLES: ReadonlySet<string> = new Set([
  "cat",
  "ls",
  "pwd",
  "echo",
  "head",
  "tail",
  "wc",
  "grep",
  "find",
  "stat",
  "file",
  "date",
  "whoami",
  "hostname",
  "jq",
  "cut",
  "sort",
  "uniq",
  "base64",
  "df",
  "du",
  "ps",
  "uptime",
]);

/**
 * Shell metacharacters that would let an argument re-introduce shell semantics or
 * chain a second command even though execFileSync does not spawn a shell. Any
 * argument (or the command as a whole) containing one of these is rejected so the
 * allowlist cannot be bypassed by smuggling a second executable into an argument.
 */
const SHELL_METACHARACTERS: readonly string[] = [
  ";",
  "&",
  "|",
  "`",
  "$(",
  ">",
  "<",
  "\n",
  "\r",
];

/**
 * Slash-command category+subcommand pairs that only read state. Everything not on
 * this list is treated as mutating and gated. `mcp_command` connectivity verbs
 * (ping/list/init) are handled separately in `_isMutatingMcp`.
 */
const READ_ONLY_SLASH_COMMANDS: ReadonlySet<string> = new Set([
  "service monitor",
  "service list",
  "service list-groups",
  "user list",
  "user list-users",
  "user list-groups",
  "diagnostic run-suite",
  "diagnostic run-test",
  "import dry",
  "agents list",
  "agents get",
  "agents search",
  "agents test",
  "agents test-all",
]);

/**
 * Environment variables that a read-only diagnostic command legitimately needs.
 * Everything else (tokens, client secrets, API keys) is withheld so an approved
 * command such as `env`-adjacent inspection or a shelled tool cannot exfiltrate
 * credentials that live in the parent process environment.
 */
const SHELL_ENV_ALLOWLIST: readonly string[] = ["PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR"];


/**
 * Build a minimal environment for shell tool execution, exposing only the
 * variables a diagnostic binary needs and withholding all credential-bearing
 * variables from the parent process.
 */
export function buildShellEnv(parentEnv: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const key of SHELL_ENV_ALLOWLIST) {
    const value = parentEnv[key];
    if (value !== undefined) {
      env[key] = value;
    }
  }
  return env;
}


export interface ShellCheckResult {
  allowed: boolean;
  executable: string;
  reason?: string;
}

export interface ToolPolicyDecision {
  /** Whether the invocation is permitted to run at all (allowlist/parse check). */
  allowed: boolean;
  /** Whether a human confirmation is required before running. */
  requiresConfirmation: boolean;
  /** Human-readable summary of the action, shown in the confirmation prompt. */
  summary: string;
  /** Populated when `allowed` is false. */
  denyReason?: string;
}


function _containsMetacharacter(value: string): boolean {
  return SHELL_METACHARACTERS.some((meta) => value.includes(meta));
}


function _normalizeSlashCommand(commandText: string): string {
  const withoutSlash = commandText.trim().replace(/^\/+/, "").trim();
  const parts = withoutSlash.split(/\s+/).filter((part) => part.length > 0);
  // category + subcommand identify the action; remaining tokens are arguments.
  return parts.slice(0, 2).join(" ").toLowerCase();
}


function _isMutatingSlashCommand(commandText: string): boolean {
  const normalized = _normalizeSlashCommand(commandText);
  if (!normalized) {
    // Empty / unparseable command: fail closed, treat as mutating.
    return true;
  }
  return !READ_ONLY_SLASH_COMMANDS.has(normalized);
}


function _isMutatingMcp(input: Record<string, unknown>): boolean {
  const command = String(input.command || "").toLowerCase();
  // ping/list/init are connectivity/discovery only. `call` invokes an arbitrary
  // upstream MCP tool, which may mutate state, so it is always gated.
  return command === "call" || command === "";
}


/**
 * Validate a `shell_command` string against the deny-by-default executable
 * allowlist and reject shell metacharacters in the executable or its arguments.
 *
 * Fails closed: an empty command, an executable not on the allowlist, or any
 * metacharacter results in `allowed: false`.
 */
export function checkShellCommand(command: string): ShellCheckResult {
  const trimmed = command.trim();
  if (!trimmed) {
    return {allowed: false, executable: "", reason: "Empty shell command."};
  }

  const parts = trimmed.split(/\s+/);
  const executable = parts[0] ?? "";

  // Reject any absolute/relative path form (e.g. ./x, /bin/x, ../x). The allowlist
  // matches bare command names resolved on PATH; a path lets the model point at an
  // arbitrary binary that shares a name with an allowed one.
  if (executable.includes("/")) {
    return {
      allowed: false,
      executable,
      reason: `Path-qualified executables are not permitted: '${executable}'.`,
    };
  }

  if (!ALLOWED_SHELL_EXECUTABLES.has(executable)) {
    return {
      allowed: false,
      executable,
      reason: `Executable '${executable}' is not on the diagnostic allowlist.`,
    };
  }

  if (_containsMetacharacter(trimmed)) {
    return {
      allowed: false,
      executable,
      reason: "Shell metacharacters are not permitted (command chaining/redirection blocked).",
    };
  }

  return {allowed: true, executable};
}


/**
 * Classify a mapped tool invocation into a policy decision: whether it may run at
 * all, and whether it needs human confirmation first.
 *
 * Fails closed: unknown tool types are denied; anything not provably read-only is
 * treated as mutating and gated.
 */
export function evaluateToolPolicy(invocation: AgentToolInvocation): ToolPolicyDecision {
  if (invocation.type === "shell") {
    const command = String(invocation.input.command || "").trim();
    const check = checkShellCommand(command);
    if (!check.allowed) {
      return {
        allowed: false,
        requiresConfirmation: false,
        summary: `shell_command: ${command}`,
        denyReason: check.reason,
      };
    }
    // Even an allowlisted, read-only diagnostic command requires confirmation:
    // untrusted context can still steer the model to exfiltrate file contents.
    return {
      allowed: true,
      requiresConfirmation: true,
      summary: `shell_command: ${command}`,
    };
  }

  if (invocation.type === "task") {
    const command = String(invocation.input.command || "");
    const mutating = _isMutatingSlashCommand(command);
    return {
      allowed: true,
      requiresConfirmation: mutating,
      summary: `registry_task: ${command.trim() || "(empty)"}`,
    };
  }

  if (invocation.type === "mcp") {
    const mutating = _isMutatingMcp(invocation.input);
    const command = String(invocation.input.command || "");
    const tool = invocation.input.tool ? String(invocation.input.tool) : "";
    const summary = tool ? `mcp_command: ${command} ${tool}` : `mcp_command: ${command}`;
    return {
      allowed: true,
      requiresConfirmation: mutating,
      summary,
    };
  }

  if (invocation.type === "docs") {
    // Read-only documentation lookup; never mutating.
    return {allowed: true, requiresConfirmation: false, summary: "read_docs"};
  }

  // Unknown tool type: deny.
  return {
    allowed: false,
    requiresConfirmation: false,
    summary: `unknown: ${invocation.name}`,
    denyReason: `Unknown tool '${invocation.name}' is not permitted.`,
  };
}
