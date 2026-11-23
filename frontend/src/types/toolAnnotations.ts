/**
 * MCP tool annotations a server declared for one tool. Advisory: the registry
 * stores what the server sent and does not verify it. Absent when none were sent.
 *
 * MCP spec: https://modelcontextprotocol.io/specification/2025-11-25/server/tools
 */
export interface ToolAnnotations {
  title?: string;
  readOnlyHint?: boolean;
  destructiveHint?: boolean;
  idempotentHint?: boolean;
  openWorldHint?: boolean;
}
