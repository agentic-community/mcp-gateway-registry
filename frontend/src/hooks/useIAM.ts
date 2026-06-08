import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

// ─── Types ──────────────────────────────────────────────────────

export interface IAMGroup {
  name: string;
  description?: string;
  path?: string;
  members_count?: number;
  // True if the group is managed in the upstream IdP; false for local-only.
  // Null/undefined for legacy records that predate the flag. See issue #946.
  is_idp_managed?: boolean | null;
}

export interface IAMUser {
  username: string;
  email?: string;
  first_name?: string;
  last_name?: string;
  groups?: string[];
  enabled?: boolean;
  is_admin?: boolean;
  account_type?: string;
  serviceAccountsEnabled?: boolean;
}

export interface M2MCredentials {
  client_id: string;
  client_secret: string;
  name: string;
}

export interface CreateHumanUserPayload {
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  password?: string;
  groups?: string[];
}

export interface CreateM2MPayload {
  name: string;
  description?: string;
  groups?: string[];
}

export interface CreateGroupPayload {
  name: string;
  description?: string;
  // scope_config is included for future backend support.
  // Currently the backend accepts but does not process it.
  scope_config?: Record<string, unknown>;
}

// ─── Hook: useIAMGroups ─────────────────────────────────────────

export function useIAMGroups() {
  const [groups, setGroups] = useState<IAMGroup[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchGroups = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await axios.get('/api/management/iam/groups');
      setGroups(res.data.groups || res.data || []);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load groups');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => { fetchGroups(); }, [fetchGroups]);

  return { groups, isLoading, error, refetch: fetchGroups };
}

export async function createGroup(payload: CreateGroupPayload): Promise<any> {
  const res = await axios.post('/api/management/iam/groups', payload);
  return res.data;
}

export async function deleteGroup(name: string): Promise<void> {
  await axios.delete(`/api/management/iam/groups/${encodeURIComponent(name)}`);
}

// ─── Group Detail Types & Functions ────────────────────────────

export interface GroupDetail {
  id: string;
  name: string;
  path?: string;
  description?: string;
  server_access?: Array<{server: string; methods: string[]; tools?: string[]}>;
  group_mappings?: string[];
  ui_permissions?: Record<string, string[]>;
  agent_access?: string[];
  // See issue #946. True=IdP-managed, false=local-only, null=legacy/unknown.
  is_idp_managed?: boolean | null;
}

export interface UpdateGroupPayload {
  description?: string;
  scope_config?: {
    server_access?: Array<{server: string; methods: string[]; tools?: string[]}>;
    ui_permissions?: Record<string, string[]>;
    agent_access?: string[];
  };
}

export async function getGroup(groupName: string): Promise<GroupDetail> {
  const res = await axios.get(`/api/management/iam/groups/${encodeURIComponent(groupName)}`);
  return res.data;
}

export async function updateGroup(
  groupName: string,
  payload: UpdateGroupPayload
): Promise<GroupDetail> {
  const res = await axios.patch(
    `/api/management/iam/groups/${encodeURIComponent(groupName)}`,
    payload
  );
  return res.data;
}

// ─── Hook: useIAMUsers ──────────────────────────────────────────

export function useIAMUsers(search?: string) {
  const [users, setUsers] = useState<IAMUser[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchUsers = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const params: Record<string, string | number> = { limit: 500 };
      if (search) params.search = search;
      const res = await axios.get('/api/management/iam/users', { params });
      setUsers(res.data.users || res.data || []);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load users');
    } finally {
      setIsLoading(false);
    }
  }, [search]);

  useEffect(() => { fetchUsers(); }, [fetchUsers]);

  return { users, isLoading, error, refetch: fetchUsers };
}

export async function createHumanUser(payload: CreateHumanUserPayload): Promise<any> {
  const res = await axios.post('/api/management/iam/users/human', payload);
  return res.data;
}

export async function createM2MAccount(payload: CreateM2MPayload): Promise<M2MCredentials> {
  const res = await axios.post('/api/management/iam/users/m2m', payload);
  return res.data;
}

export async function deleteUser(username: string): Promise<void> {
  await axios.delete(`/api/management/iam/users/${encodeURIComponent(username)}`);
}

export interface UpdateUserGroupsResponse {
  username: string;
  groups: string[];
  added: string[];
  removed: string[];
}

export async function updateUserGroups(
  username: string,
  groups: string[]
): Promise<UpdateUserGroupsResponse> {
  const res = await axios.patch(
    `/api/management/iam/users/${encodeURIComponent(username)}/groups`,
    { groups }
  );
  return res.data;
}

// ─── M2M Client (idp_m2m_clients) Types ────────────────────────
// Mirrors registry/schemas/idp_m2m_client.py IdPM2MClient / Create / Patch / ListResponse.

export interface M2MClient {
  client_id: string;
  name: string;
  description?: string | null;
  groups: string[];
  enabled: boolean;
  provider: 'manual' | 'okta' | 'auth0' | 'keycloak' | 'entra' | string;
  created_at: string;
  updated_at: string;
  idp_app_id?: string | null;
  created_by?: string | null;
}

export interface RegisterM2MClientPayload {
  client_id: string;
  client_name: string;
  groups: string[];
  description?: string;
}

export interface PatchM2MClientPayload {
  client_name?: string;
  groups?: string[];
  description?: string;
  enabled?: boolean;
}

export interface M2MClientListResponse {
  total: number;
  limit: number;
  skip: number;
  items: M2MClient[];
}

// ─── Hook: useM2MClients ────────────────────────────────────────

export function useM2MClients() {
  const [clients, setClients] = useState<M2MClient[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchClients = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await axios.get<M2MClientListResponse>(
        '/api/iam/m2m-clients',
        { params: { limit: 1000 } }
      );
      setClients(res.data.items || []);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load M2M clients');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => { fetchClients(); }, [fetchClients]);

  return { clients, isLoading, error, refetch: fetchClients };
}

export async function registerM2MClient(
  payload: RegisterM2MClientPayload
): Promise<M2MClient> {
  const res = await axios.post<M2MClient>('/api/iam/m2m-clients', payload);
  return res.data;
}

export async function patchM2MClient(
  clientId: string,
  payload: PatchM2MClientPayload
): Promise<M2MClient> {
  const res = await axios.patch<M2MClient>(
    `/api/iam/m2m-clients/${encodeURIComponent(clientId)}`,
    payload
  );
  return res.data;
}

export async function deleteM2MClient(clientId: string): Promise<void> {
  await axios.delete(
    `/api/iam/m2m-clients/${encodeURIComponent(clientId)}`
  );
}

// ─── IdP User-Group (idp_user_groups) Types ────────────────────
// Mirrors registry/schemas/idp_user_group.py IdPUserGroup / Create / Patch / ListResponse.
// Used by the auth server as a fallback authorization database for IdPs that
// do NOT carry group memberships in JWTs (e.g. PingFederate). See issue #1127.

export interface IdPUserGroup {
  username: string;
  groups: string[];
  enabled: boolean;
  provider: string;
  email?: string | null;
  created_by?: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateUserGroupPayload {
  username: string;
  groups: string[];
  email?: string;
}

export interface PatchUserGroupPayload {
  groups?: string[];
  email?: string | null;
  enabled?: boolean;
}

export interface UserGroupListResponse {
  total: number;
  limit: number;
  skip: number;
  items: IdPUserGroup[];
}

export interface ListUserGroupsParams {
  skip?: number;
  limit?: number;
  provider?: string;
  q?: string;
}

export async function listUserGroups(
  params: ListUserGroupsParams = {}
): Promise<UserGroupListResponse> {
  // Strip undefined/empty values so we don't send empty query params.
  const cleaned: Record<string, string | number> = {};
  if (params.skip !== undefined) cleaned.skip = params.skip;
  if (params.limit !== undefined) cleaned.limit = params.limit;
  if (params.provider) cleaned.provider = params.provider;
  if (params.q) cleaned.q = params.q;

  const res = await axios.get<UserGroupListResponse>(
    '/api/iam/user-groups',
    { params: cleaned }
  );
  return res.data;
}

export async function createUserGroup(
  payload: CreateUserGroupPayload
): Promise<IdPUserGroup> {
  // Backend defaults provider to "manual" for hand-registered records, so we
  // do not send it from the client (the schema does not accept it).
  const res = await axios.post<IdPUserGroup>('/api/iam/user-groups', payload);
  return res.data;
}

export async function getUserGroup(username: string): Promise<IdPUserGroup> {
  const res = await axios.get<IdPUserGroup>(
    `/api/iam/user-groups/${encodeURIComponent(username)}`
  );
  return res.data;
}

export async function updateUserGroup(
  username: string,
  patch: PatchUserGroupPayload
): Promise<IdPUserGroup> {
  // Mirrors backend's model_dump(exclude_unset=True): callers should only pass
  // fields that actually changed so unset fields are not written.
  const res = await axios.patch<IdPUserGroup>(
    `/api/iam/user-groups/${encodeURIComponent(username)}`,
    patch
  );
  return res.data;
}

export async function deleteUserGroup(username: string): Promise<void> {
  await axios.delete(
    `/api/iam/user-groups/${encodeURIComponent(username)}`
  );
}

// Issue #1127: companion endpoint that creates the user inside PingFederate's
// Simple Password Credential Validator. The registry does not store this
// password; it is forwarded once to the auth server which calls the
// PingFederate admin API. Caller invokes this AFTER the registry-side
// idp_user_groups record was created successfully.
export async function createPingFederateUser(
  username: string,
  password: string
): Promise<void> {
  await axios.post(
    `/api/iam/user-groups/${encodeURIComponent(username)}/pingfederate-user`,
    { password }
  );
}

// ─── Hook: useUserGroups ────────────────────────────────────────

export interface UseUserGroupsParams extends ListUserGroupsParams {}

export function useUserGroups(params: UseUserGroupsParams = {}) {
  const [data, setData] = useState<UserGroupListResponse>({
    total: 0,
    limit: params.limit ?? 25,
    skip: params.skip ?? 0,
    items: [],
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Stable dependency string -- avoids re-running fetch on every render when
  // callers pass a fresh object literal.
  const paramsKey = JSON.stringify({
    skip: params.skip,
    limit: params.limit,
    provider: params.provider,
    q: params.q,
  });

  const fetchUserGroups = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await listUserGroups(params);
      setData(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load user groups');
    } finally {
      setIsLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paramsKey]);

  useEffect(() => {
    fetchUserGroups();
  }, [fetchUserGroups]);

  return {
    data,
    items: data.items,
    total: data.total,
    isLoading,
    error,
    refetch: fetchUserGroups,
  };
}
