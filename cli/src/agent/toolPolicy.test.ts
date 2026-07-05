import assert from "node:assert/strict";
import {test} from "node:test";

import {buildShellEnv, checkShellCommand, evaluateToolPolicy} from "./toolPolicy.js";
import {executeMappedTool, mapToolCall} from "./tools.js";
import type {TaskContext} from "../tasks/types.js";

const CONTEXT: TaskContext = {
  gatewayUrl: "http://localhost/mcpgw/mcp",
  gatewayBaseUrl: "http://localhost",
};


test("checkShellCommand allows read-only diagnostic binaries", () => {
  const result = checkShellCommand("cat /tmp/example.json");
  assert.equal(result.allowed, true);
  assert.equal(result.executable, "cat");
});


test("checkShellCommand rejects a destructive executable", () => {
  const result = checkShellCommand("rm -rf /");
  assert.equal(result.allowed, false);
  assert.match(result.reason ?? "", /not on the diagnostic allowlist/);
});


test("checkShellCommand rejects path-qualified executables", () => {
  const result = checkShellCommand("/bin/cat /etc/passwd");
  assert.equal(result.allowed, false);
  assert.match(result.reason ?? "", /Path-qualified/);
});


test("checkShellCommand rejects shell metacharacters (command chaining)", () => {
  const result = checkShellCommand("cat file.json; rm -rf /");
  assert.equal(result.allowed, false);
  assert.match(result.reason ?? "", /metacharacters/);
});


test("checkShellCommand rejects an empty command", () => {
  assert.equal(checkShellCommand("   ").allowed, false);
});


test("evaluateToolPolicy gates every shell command even when allowlisted", () => {
  const decision = evaluateToolPolicy(mapToolCall({name: "shell_command", input: {command: "cat x.json"}}));
  assert.equal(decision.allowed, true);
  assert.equal(decision.requiresConfirmation, true);
});


test("checkShellCommand rejects the env dumper (removed from allowlist)", () => {
  // `env` would print all environment variables, including tokens/secrets.
  assert.equal(checkShellCommand("env").allowed, false);
});


test("buildShellEnv withholds credential-bearing variables", () => {
  const parent = {
    PATH: "/usr/bin",
    HOME: "/home/op",
    GATEWAY_TOKEN: "FAKE_TOKEN_PLACEHOLDER", // pragma: allowlist secret
    M2M_CLIENT_SECRET: "FAKE_SECRET_PLACEHOLDER", // pragma: allowlist secret
    AWS_SECRET_ACCESS_KEY: "FAKE_AWS_PLACEHOLDER", // pragma: allowlist secret
  } as unknown as NodeJS.ProcessEnv;
  const scrubbed = buildShellEnv(parent);
  assert.equal(scrubbed.PATH, "/usr/bin");
  assert.equal(scrubbed.HOME, "/home/op");
  assert.equal(scrubbed.GATEWAY_TOKEN, undefined);
  assert.equal(scrubbed.M2M_CLIENT_SECRET, undefined);
  assert.equal(scrubbed.AWS_SECRET_ACCESS_KEY, undefined);
});


test("evaluateToolPolicy denies a disallowed shell executable outright", () => {
  const decision = evaluateToolPolicy(mapToolCall({name: "shell_command", input: {command: "curl http://evil"}}));
  assert.equal(decision.allowed, false);
  assert.equal(decision.requiresConfirmation, false);
});


test("evaluateToolPolicy treats a mutating registry_task as requiring confirmation", () => {
  const decision = evaluateToolPolicy(
    mapToolCall({name: "registry_task", input: {command: "/user delete --username victim"}})
  );
  assert.equal(decision.requiresConfirmation, true);
});


test("evaluateToolPolicy treats a read-only registry_task as no-confirmation", () => {
  const decision = evaluateToolPolicy(mapToolCall({name: "registry_task", input: {command: "/user list"}}));
  assert.equal(decision.requiresConfirmation, false);
});


test("evaluateToolPolicy fails closed on an empty/unparseable registry_task", () => {
  const decision = evaluateToolPolicy(mapToolCall({name: "registry_task", input: {command: ""}}));
  assert.equal(decision.requiresConfirmation, true);
});


test("evaluateToolPolicy gates a mutating mcp call but not connectivity verbs", () => {
  const callDecision = evaluateToolPolicy(mapToolCall({name: "mcp_command", input: {command: "call", tool: "x"}}));
  assert.equal(callDecision.requiresConfirmation, true);
  const listDecision = evaluateToolPolicy(mapToolCall({name: "mcp_command", input: {command: "list"}}));
  assert.equal(listDecision.requiresConfirmation, false);
});


test("evaluateToolPolicy denies an unknown tool", () => {
  const decision = evaluateToolPolicy(mapToolCall({name: "definitely_not_a_tool", input: {}}));
  assert.equal(decision.allowed, false);
});


test("executeMappedTool fails closed when a mutating call has no confirmation handler", async () => {
  const invocation = mapToolCall({name: "registry_task", input: {command: "/user delete --username victim"}});
  const result = await executeMappedTool(invocation, CONTEXT.gatewayUrl, CONTEXT);
  assert.equal(result.isError, true);
  assert.match(result.output, /requires human confirmation/);
});


test("executeMappedTool does not execute a declined mutating call", async () => {
  const invocation = mapToolCall({name: "registry_task", input: {command: "/user delete --username victim"}});
  let asked = false;
  const result = await executeMappedTool(invocation, CONTEXT.gatewayUrl, CONTEXT, async () => {
    asked = true;
    return false;
  });
  assert.equal(asked, true);
  assert.equal(result.isError, true);
  assert.match(result.output, /declined by operator/);
});


test("executeMappedTool blocks a disallowed shell executable before any prompt", async () => {
  const invocation = mapToolCall({name: "shell_command", input: {command: "rm -rf /"}});
  let asked = false;
  const result = await executeMappedTool(invocation, CONTEXT.gatewayUrl, CONTEXT, async () => {
    asked = true;
    return true;
  });
  assert.equal(asked, false, "disallowed executable must be rejected without prompting");
  assert.equal(result.isError, true);
  assert.match(result.output, /not on the diagnostic allowlist/);
});


test("executeMappedTool runs an approved allowlisted shell command", async () => {
  const invocation = mapToolCall({name: "shell_command", input: {command: "echo hello-safe"}});
  const result = await executeMappedTool(invocation, CONTEXT.gatewayUrl, CONTEXT, async () => true);
  assert.equal(result.isError, undefined);
  assert.match(result.output, /hello-safe/);
});
