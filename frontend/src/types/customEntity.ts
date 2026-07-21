/**
 * Types mirroring registry/schemas/custom_entity_models.py.
 *
 * A custom entity type is described by a CustomTypeDescriptor (a name plus a
 * list of typed field descriptors). Records of that type carry a uniform
 * envelope (name/description/visibility/owner/tags/...) plus a per-type
 * `attributes` bag whose shape is validated against the descriptor server-side.
 */

/** Allowed datatypes for a custom-type field (v1: scalars + scalar arrays). */
export type CustomFieldType =
  | 'string'
  | 'text'
  | 'number'
  | 'bool'
  | 'enum'
  | 'date'
  | 'array<string>';

/** One field in a custom type's schema. */
export interface CustomFieldDescriptor {
  name: string;
  label?: string | null;
  datatype: CustomFieldType;
  enum_values?: string[] | null;
  required: boolean;
  semantic: boolean;
  show_in_list: boolean;
}

/** Admin-authored schema for a custom entity type. */
export interface CustomTypeDescriptor {
  name: string;
  display_name?: string | null;
  description?: string | null;
  fields: CustomFieldDescriptor[];
  schema_version: number;
  created_by?: string | null;
  created_at: string;
}

/** A record of a custom type (envelope + attributes). */
export interface CustomEntityRecord {
  path: string;
  entity_type: string;
  name: string;
  description?: string | null;
  visibility: string;
  allowed_groups: string[];
  owner?: string | null;
  tags: string[];
  is_enabled: boolean;
  num_stars: number;
  rating_details: Array<{ user: string; rating: number }>;
  created_at: string;
  updated_at: string;
  attributes: Record<string, unknown>;
  // Gateway-proxy opt-in. Top-level (mixin) fields on the wire, NOT in attributes.
  // proxy_client_url is the read-only, server-derived client path.
  is_proxied?: boolean;
  proxy_target_url?: string | null;
  proxy_client_url?: string | null;
  // Upstream custom-header NAMES (values encrypted, never returned).
  // custom_header_overridable_names is the caller-overridable subset.
  custom_header_names?: string[];
  custom_header_overridable_names?: string[];
}

/** One upstream custom header on the create payload / rotation body. */
export interface CustomEntityHeader {
  name: string;
  value?: string;
  overridable?: boolean;
}

/** Client payload for POST /api/custom/{type}. */
export interface CustomEntityCreate {
  name: string;
  description?: string | null;
  visibility: string;
  allowed_groups: string[];
  tags: string[];
  attributes: Record<string, unknown>;
  is_proxied?: boolean;
  proxy_target_url?: string | null;
  custom_headers?: CustomEntityHeader[];
}

/** Client payload for PUT /api/custom/{type}/{uuid} (all optional). */
export interface CustomEntityUpdate {
  name?: string;
  description?: string | null;
  visibility?: string;
  allowed_groups?: string[];
  tags?: string[];
  attributes?: Record<string, unknown> | null;
  is_proxied?: boolean;
  proxy_target_url?: string | null;
}

/** Shape of the 400 validation-error body: { detail: [{ field, message }, ...] }. */
export interface CustomEntityFieldError {
  field: string;
  message: string;
}

// Value bounds mirrored from the backend (custom_entity_models.py).
export const MAX_STRING_LEN = 1_000;
export const MAX_TEXT_LEN = 50_000;
export const MAX_ARRAY_ITEMS = 200;
