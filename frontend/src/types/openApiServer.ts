/**
 * Types for OpenAPI MCP servers (issue #2022).
 *
 * These mirror the response models and request bodies of the
 * /api/openapi-servers routes. A target's credential is write-only: responses
 * carry only `credential_set`, never the value. Non-admin callers get no
 * `spec_url` or `base_url`, so both are optional here.
 */


/** The auth schemes a REST target can use. */
export type OpenApiAuthScheme =
  | 'none'
  | 'api_key_header'
  | 'api_key_query'
  | 'bearer'
  | 'basic';


/**
 * Who supplies a target's credential. `shared`: one admin key for every
 * caller (stored with the target). `per_user`: each user adds their own key,
 * stored in the egress vault, and the target stores none.
 */
export type OpenApiCredentialMode = 'shared' | 'per_user';


/** The schemes a per-user target may use. */
export const PER_USER_SCHEMES: ReadonlyArray<OpenApiAuthScheme> = ['bearer', 'api_key_header'];


/** Whether the container has confirmed the latest generation of a server. */
export type OpenApiSyncStatus = 'applied' | 'pending';


/** The container's status for one target. */
export type OpenApiTargetStatus = 'ok' | 'error' | 'pending';


/** A target's auth as the API returns it. The credential appears only as a flag. */
export interface OpenApiTargetAuthResponse {
  scheme: OpenApiAuthScheme;
  header_name?: string | null;
  query_param?: string | null;
  username?: string | null;
  credential_set: boolean;
  /** Missing on targets written before per-user credentials; treat as `shared`. */
  credential_mode?: OpenApiCredentialMode;
}


/** One REST target as the API returns it. */
export interface OpenApiTarget {
  name: string;
  description?: string;
  spec_url?: string;
  /** `upload` when an admin sent the spec itself; such a target has no spec_url. */
  spec_source?: OpenApiSpecSource;
  spec_sha256?: string;
  spec_title?: string;
  spec_version?: string;
  openapi_version?: string;
  base_url?: string;
  auth: OpenApiTargetAuthResponse;
  tool_name_overrides?: Record<string, string>;
  description_overrides?: Record<string, string>;
  include_tags?: string[];
  include_operations?: string[];
  keep_output_schemas?: boolean;
  tool_names?: string[];
  created_by?: string;
  created_at?: string;
  updated_at?: string;
  /** The container's current status for this target (detail response only). */
  status?: OpenApiTargetStatus;
  /** The container's error message when `status` is `error`. */
  error?: string | null;
}


/** One OpenAPI MCP server as the API returns it. */
export interface OpenApiServer {
  slug: string;
  name: string;
  description?: string;
  path: string;
  mcp_url?: string;
  generation?: number;
  sync_status?: OpenApiSyncStatus;
  tags?: string[];
  visibility?: OpenApiVisibility;
  allowed_groups?: string[];
  targets: OpenApiTarget[];
  created_by?: string;
  created_at?: string;
  updated_at?: string;
}


/** Who can see a server. */
export type OpenApiVisibility = 'public' | 'private' | 'group-restricted';


/** Body of POST /api/openapi-servers. */
export interface CreateOpenApiServerRequest {
  name: string;
  description: string;
  tags: string[];
  visibility: OpenApiVisibility;
  allowed_groups: string[];
}


/** Body of PATCH /api/openapi-servers/{slug}. The name cannot change. */
export interface UpdateOpenApiServerRequest {
  description?: string;
  tags?: string[];
  visibility?: OpenApiVisibility;
  allowed_groups?: string[];
}


/**
 * A target's auth as the admin sends it. The credential is write-only. A
 * `per_user` target sends only `scheme` and, for `api_key_header`,
 * `header_name`: never `credential`, `query_param` or `username`.
 */
export interface OpenApiTargetAuthInput {
  scheme: OpenApiAuthScheme;
  credential_mode?: OpenApiCredentialMode;
  header_name?: string;
  query_param?: string;
  username?: string;
  credential?: string;
}


/** Where a target's spec came from. Missing on older targets; treat as `url`. */
export type OpenApiSpecSource = 'url' | 'upload';


/** Body of POST /api/openapi-servers/{slug}/targets. Send spec_url or spec_content. */
export interface CreateOpenApiTargetRequest {
  name: string;
  description: string;
  spec_url?: string;
  /** The spec's JSON or YAML text, uploaded instead of a URL. */
  spec_content?: string;
  base_url: string;
  auth: OpenApiTargetAuthInput;
  tool_name_overrides: Record<string, string>;
  description_overrides: Record<string, string>;
  include_tags: string[];
  include_operations: string[];
  keep_output_schemas: boolean;
}


/**
 * Body of PATCH /api/openapi-servers/{slug}/targets/{name}.
 *
 * Every field is optional. A missing `auth` leaves the stored credential as it
 * is. `refresh_spec` refetches the spec URL and runs the checks again;
 * `spec_content` replaces the spec with uploaded text. Send at most one.
 */
export interface UpdateOpenApiTargetRequest {
  description?: string;
  base_url?: string;
  auth?: OpenApiTargetAuthInput;
  tool_name_overrides?: Record<string, string>;
  description_overrides?: Record<string, string>;
  include_tags?: string[];
  include_operations?: string[];
  keep_output_schemas?: boolean;
  refresh_spec?: boolean;
  spec_content?: string;
}


/** Response of POST (and PATCH) on a target. */
export interface OpenApiTargetWriteResponse {
  name: string;
  spec_title?: string;
  openapi_version?: string;
  auth?: OpenApiTargetAuthResponse;
  tool_names?: string[];
  warnings?: string[];
  findings?: unknown[];
  generation?: number;
  sync_status?: OpenApiSyncStatus;
}


/** The error codes the routes return in the `error` field of a 4xx/5xx body. */
export type OpenApiErrorCode =
  | 'duplicate_name'
  | 'limit_reached'
  | 'container_unavailable'
  | 'url_blocked'
  | 'spec_too_large'
  | 'spec_invalid'
  | 'conversion_failed'
  | 'too_many_tools'
  | 'tool_name_invalid'
  | 'spec_fetch_failed'
  | 'egress_auth_disabled'
  | 'not_per_user';


/** The lifetime units a per-user key accepts. */
export type OpenApiKeyTtlUnit = 'minutes' | 'hours' | 'days';


/** Body of PUT .../targets/{name}/my-credential. The secret is write-only. */
export interface SetMyCredentialRequest {
  secret: string;
  ttl_value: number;
  ttl_unit: OpenApiKeyTtlUnit;
}


/** The caller's key status for one per-user target. Never carries the key. */
export interface MyCredentialStatus {
  target: string;
  configured: boolean;
  expires_at: string | null;
}


/** Response of GET /api/openapi-servers/{slug}/my-credentials. */
export interface MyCredentialsResponse {
  slug: string;
  targets: MyCredentialStatus[];
}


/**
 * A parsed API error. `code` is null for errors without a code (for example a
 * body validation failure), in which case `message` holds the text to show.
 */
export interface OpenApiApiError {
  status: number | null;
  code: string | null;
  message: string;
  /** too_many_tools: tool count after the filters. */
  count?: number;
  /** too_many_tools and limit_reached: the limit that was hit. */
  limit?: number;
  /** too_many_tools: every tag in the spec with its operation count. */
  tags?: Record<string, number>;
  /** tool_name_invalid: the offending tool names. */
  names?: string[];
  /** url_blocked: which URL the guard rejected, when the body says. */
  field?: string;
}
