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
  status?: 'active' | 'draft' | 'deprecated' | 'beta';
  health_status?: 'healthy' | 'unhealthy' | 'unknown';
  last_checked_time?: string;
  created_at?: string;
  updated_at?: string;
}
