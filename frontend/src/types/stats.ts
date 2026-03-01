/**
 * Shared system statistics type definitions for the MCP Gateway Registry frontend.
 *
 * These interfaces match the backend API response from /api/stats endpoint.
 */


/**
 * Database health status.
 */
export interface DatabaseStatus {
  backend: string;    // "file" | "documentdb" | "mongodb-ce"
  status: string;     // "Healthy" | "Unhealthy" | "N/A"
  host: string;       // Database host (e.g., "localhost:27017")
}


/**
 * Authentication server health status.
 */
export interface AuthStatus {
  provider: string;   // "cognito" | "keycloak" | "entra" | "github"
  status: string;     // "Healthy" | "Unhealthy"
  url: string;        // Auth server URL
}


/**
 * Registry resource counts.
 */
export interface RegistryStatsData {
  servers: number;
  agents: number;
  skills: number;
}


/**
 * Complete system statistics response from /api/stats.
 */
export interface SystemStats {
  uptime_seconds: number;
  started_at: string;           // ISO 8601 timestamp
  version: string;
  deployment_type: string;      // "Kubernetes" | "ECS" | "EC2" | "Local"
  deployment_mode: string;      // "with-gateway" | "registry-only"
  registry_stats: RegistryStatsData;
  database_status: DatabaseStatus;
  auth_status: AuthStatus;
}
