import axios from 'axios';
import { useIAMResource } from './useIAMResource';

// ─── Types ──────────────────────────────────────────────────────

// The rate-limit axis. `caller_target` is a per-caller-per-target group (each
// member gets an independent quota per target); `quarantine` is a sentinel
// kill-switch group carrying no rate. Mirrors registry/rate_limiting/models.py.
export type RateLimitAxis = 'caller' | 'target' | 'caller_target' | 'quarantine';

// A friendly label per axis for the UI (never render the raw enum).
export const AXIS_LABELS: Record<RateLimitAxis, string> = {
  caller: 'Per caller',
  target: 'Per target',
  caller_target: 'Per caller, per target',
  quarantine: 'Quarantine (kill switch)',
};

// A rate-limit definition. Caller / caller_target (group) definitions carry
// per-caller-type limits (user_max_requests / agent_max_requests); target
// definitions carry a single max_requests; quarantine sentinels carry a scope
// and no rate. Mirrors registry/rate_limiting/models.py.
export interface RateLimitDefinition {
  axis: RateLimitAxis;
  entity_type: string;
  name: string;
  max_requests?: number | null;
  user_max_requests?: number | null;
  agent_max_requests?: number | null;
  window_seconds: number;
  scope?: 'caller' | 'target' | null;
  fail_closed: boolean;
  enabled: boolean;
  // server_group target entity only: the set of server paths the definition
  // covers. Each member is limited to max_requests/window INDIVIDUALLY (its own
  // bucket), not a shared pool. Absent for every other definition.
  members?: string[] | null;
}

// A rate-limit membership: maps a caller (user/client) OR a target (server/agent)
// to rate-limit group name(s). Target subjects exist only to be quarantined.
export interface RateLimitMembership {
  subject_type: 'user' | 'client' | 'server' | 'agent';
  subject: string;
  groups: string[];
}

// Everything currently quarantined, as returned by GET /api/rate-limit-quarantine.
export interface QuarantineList {
  callers: RateLimitMembership[];
  targets: RateLimitMembership[];
}

export const CALLER_ENTITY_TYPE = 'group';
export const TARGET_ENTITY_TYPES = ['mcp_server', 'a2a_agent', 'server_group'];
export const SERVER_GROUP_ENTITY_TYPE = 'server_group';

// Friendly labels for target entity types (never render the raw enum).
export const TARGET_ENTITY_TYPE_LABELS: Record<string, string> = {
  mcp_server: 'MCP server',
  a2a_agent: 'A2A agent',
  server_group: 'Server group (per-member uniform)',
};
export const CALLER_SUBJECT_TYPES = ['user', 'client'];
export const TARGET_SUBJECT_TYPES = ['server', 'agent'];
export const QUARANTINE_CALLER_GROUP = 'quarantine-callers';
export const QUARANTINE_TARGET_GROUP = 'quarantine-targets';

// Build the server-derived definition id: '<axis>:<entity_type>:<name>:<window_seconds>'.
export function definitionId(d: RateLimitDefinition): string {
  return `${d.axis}:${d.entity_type}:${d.name}:${d.window_seconds}`;
}

export function membershipId(m: RateLimitMembership): string {
  return `${m.subject_type}:${m.subject}`;
}

// ─── Definitions ────────────────────────────────────────────────

export function useRateLimitDefinitions() {
  const { data, isLoading, error, refetch } = useIAMResource<RateLimitDefinition>(
    async () => {
      const res = await axios.get('/api/rate-limits');
      return res.data.definitions || [];
    },
    'Failed to load rate-limit definitions',
  );
  return { definitions: data, isLoading, error, refetch };
}

export async function setRateLimitDefinition(d: RateLimitDefinition): Promise<RateLimitDefinition> {
  const res = await axios.put(
    `/api/rate-limits/${encodeURIComponent(definitionId(d))}`,
    d,
  );
  return res.data;
}

export async function deleteRateLimitDefinition(id: string): Promise<void> {
  await axios.delete(`/api/rate-limits/${encodeURIComponent(id)}`);
}

export async function setRateLimitEnabled(id: string, enabled: boolean): Promise<RateLimitDefinition> {
  const res = await axios.post(
    `/api/rate-limits-enabled/${encodeURIComponent(id)}`,
    null,
    { params: { enabled } },
  );
  return res.data;
}

// ─── Memberships ────────────────────────────────────────────────

export function useRateLimitMemberships() {
  const { data, isLoading, error, refetch } = useIAMResource<RateLimitMembership>(
    async () => {
      const res = await axios.get('/api/rate-limit-memberships');
      return res.data.memberships || [];
    },
    'Failed to load rate-limit memberships',
  );
  return { memberships: data, isLoading, error, refetch };
}

// Set (create/replace) a caller's rate-limit group membership. An empty groups
// list is stored as a membership with no groups; callers should delete instead
// when they want to remove the record entirely.
export async function setRateLimitMembership(
  subjectType: 'user' | 'client',
  subject: string,
  groups: string[],
): Promise<RateLimitMembership> {
  const id = `${subjectType}:${subject}`;
  const res = await axios.put(
    `/api/rate-limit-memberships/${encodeURIComponent(id)}`,
    { subject_type: subjectType, subject, groups },
  );
  return res.data;
}

export async function deleteRateLimitMembership(id: string): Promise<void> {
  await axios.delete(`/api/rate-limit-memberships/${encodeURIComponent(id)}`);
}

// ─── Quarantine (kill switch) ───────────────────────────────────

export type QuarantineSubjectType = 'user' | 'client' | 'server' | 'agent';

// List everything currently quarantined (callers + targets).
export function useQuarantineList() {
  const { data, isLoading, error, refetch } = useIAMResource<RateLimitMembership>(
    async () => {
      const res = await axios.get('/api/rate-limit-quarantine');
      // Flatten for the shared IAM resource shape; callers/targets are also
      // distinguishable by subject_type.
      const body = res.data as QuarantineList;
      return [...(body.callers || []), ...(body.targets || [])];
    },
    'Failed to load quarantine list',
  );
  return { quarantined: data, isLoading, error, refetch };
}

// Quarantine a subject (drops ALL its data-plane traffic). The server picks the
// reserved group from the subject type.
export async function quarantineAdd(
  subjectType: QuarantineSubjectType,
  subject: string,
): Promise<RateLimitMembership> {
  const id = `${subjectType}:${subject}`;
  const res = await axios.post(`/api/rate-limit-quarantine/${encodeURIComponent(id)}`);
  return res.data;
}

// Remove a subject from quarantine.
export async function quarantineRemove(
  subjectType: QuarantineSubjectType,
  subject: string,
): Promise<{ removed: boolean }> {
  const id = `${subjectType}:${subject}`;
  const res = await axios.delete(`/api/rate-limit-quarantine/${encodeURIComponent(id)}`);
  return res.data;
}
