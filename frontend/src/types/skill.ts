/**
 * Shared Skill type definitions for the MCP Gateway Registry frontend.
 */

/**
 * Represents a tool allowed by a skill.
 */
export interface AllowedTool {
  tool_name: string;
  server_path?: string;
  capabilities?: string[];
}


/**
 * Represents a requirement for a skill.
 */
export interface SkillRequirement {
  type: string;
  target: string;
  min_version?: string;
  required?: boolean;
}


/**
 * A single classified file inside a skill's resource manifest.
 * Mirrors registry/schemas/skill_models.py::SkillResource.
 */
export interface SkillResource {
  path: string;
  type: 'script' | 'reference' | 'asset' | 'agent';
  size_bytes: number;
  description?: string | null;
  language?: string | null;
}


/**
 * Per-skill manifest of companion resources discovered alongside SKILL.md.
 * Mirrors registry/schemas/skill_models.py::SkillResourceManifest.
 */
export interface SkillResourceManifest {
  scripts: SkillResource[];
  references: SkillResource[];
  assets: SkillResource[];
  agents: SkillResource[];
}


/**
 * Skill interface representing an Agent Skill.
 */
/**
 * Skill metadata with author, version, and custom key-value pairs.
 */
export interface SkillMetadata {
  author?: string | null;
  version?: string | null;
  extra?: Record<string, any>;
}


export interface Skill {
  name: string;
  path: string;
  description?: string;
  skill_md_url: string;
  skill_md_raw_url?: string;
  version?: string;
  author?: string;
  visibility: 'public' | 'private' | 'group';
  is_enabled: boolean;
  tags?: string[];
  owner?: string;
  registry_name?: string;
  target_agents?: string[];
  allowed_tools?: AllowedTool[];
  requirements?: SkillRequirement[];
  metadata?: SkillMetadata | null;
  repository_url?: string;
  auth_scheme?: 'none' | 'global_credentials' | 'bearer' | 'api_key';
  auth_header_name?: string;
  num_stars?: number;
  rating_details?: Array<{ user: string; rating: number }>;
  // Lightweight scan summary from the list payload, used to colour the shield
  // icon without a per-card /security-scan fetch. Undefined if not yet scanned.
  security_scan?: {
    scan_failed?: boolean;
    critical_issues?: number;
    high_severity?: number;
    medium_severity?: number;
    low_severity?: number;
  } | null;
  status?: 'active' | 'draft' | 'deprecated' | 'beta';
  health_status?: string;
  last_checked_time?: string;
  created_at?: string;
  updated_at?: string;
  resource_manifest?: SkillResourceManifest | null;
  // ARD discovery imports: read-only marker from the federation sync layer and
  // the URL to the source registry's descriptor for the "View at source" link.
  is_read_only?: boolean;
  ard_source_url?: string;
}
