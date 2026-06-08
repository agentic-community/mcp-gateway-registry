/**
 * Type definitions for the /api/{entity}/check-duplicates response
 * shape. Mirrors the Pydantic models in
 * registry/schemas/duplicate_check_models.py — keep field names in
 * sync.
 */

export type EntityType = 'mcp_server' | 'a2a_agent' | 'skill';


export interface ExistingEntity {
  entity_type: EntityType;
  /**
   * Path of the existing entity. Blanked when the caller cannot view
   * the entity per visibility rules (the existence of the collision
   * is still exposed via list-membership).
   */
  path: string;
  /** Display name of the existing entity. Blanked when redacted. */
  name: string;
  owner: string | null;
  registered_at: string | null;
  /**
   * Score from the semantic search backend. Populated only for
   * entries surfaced by the similarity check; null for exact-URL
   * collisions.
   */
  relevance_score: number | null;
  /** Human-readable reason the entry was surfaced. */
  match_reason: string | null;
}


export interface DuplicateCheckResult {
  /**
   * Existing entities (across all entity types) whose identity URL
   * matches the proposed registration. Empty when no URL collision
   * is found.
   */
  collision_with: ExistingEntity[];
  /**
   * Best-effort list of semantically similar existing entities,
   * filtered by visibility and capped by `dedup_max_suggestions`.
   * Independent of `collision_with` — both can be populated
   * simultaneously.
   */
  advisory_matches: ExistingEntity[];
  /** Minimum semantic-search score for a similarity match (echoes settings). */
  threshold: number;
  /**
   * True when the semantic search backend was reachable. False
   * indicates the similarity check was skipped due to embedder
   * unavailability; the exact-match check still ran.
   */
  similarity_search_available: boolean;
  /**
   * Convenience flag: true iff `collision_with` is non-empty.
   * Computed server-side.
   */
  has_collision: boolean;
}
