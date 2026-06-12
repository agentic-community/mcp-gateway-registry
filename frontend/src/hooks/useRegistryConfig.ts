import { useState, useEffect } from 'react';
import axios from 'axios';

/** Lightweight tab descriptor from /api/config (name + display label only). */
export interface CustomTypeTab {
  name: string;
  display_name: string;
}

interface RegistryConfig {
  deployment_mode: 'with-gateway' | 'registry-only';
  registry_mode: 'full' | 'skills-only' | 'mcp-servers-only' | 'agents-only';
  auth_provider?: string;
  nginx_updates_enabled: boolean;
  coding_assistants: string[];
  dedup_registration_hint_enabled: boolean;
  // Issue #1127: backend-driven gating for the IAM > User Groups tab.
  // True only when the active auth provider is in the IDP fallback allowlist
  // (e.g. PingFederate). For Keycloak/Okta/Entra/etc. the JWT carries groups
  // and the tab serves no purpose.
  user_group_management_enabled?: boolean;
  // Issue #1127: true only when the active auth provider is exactly
  // "pingfederate" and the auth server is configured to manage PingFederate
  // Simple PCV users via its admin API.
  pingfederate_user_management_enabled?: boolean;
  features: {
    mcp_servers: boolean;
    agents: boolean;
    skills: boolean;
    virtual_servers: boolean;
    federation: boolean;
    gateway_proxy: boolean;
    custom_types: boolean;
  };
  custom_types: CustomTypeTab[];
}

const DEFAULT_CONFIG: RegistryConfig = {
  deployment_mode: 'with-gateway',
  registry_mode: 'full',
  auth_provider: 'cognito',
  nginx_updates_enabled: true,
  coding_assistants: [],
  dedup_registration_hint_enabled: false,
  user_group_management_enabled: false,
  pingfederate_user_management_enabled: false,
  features: {
    mcp_servers: true,
    agents: true,
    skills: true,
    virtual_servers: true,
    federation: true,
    gateway_proxy: true,
    custom_types: false,
  },
  custom_types: [],
};

let cachedConfig: RegistryConfig | null = null;

export function useRegistryConfig(): {
  config: RegistryConfig | null;
  loading: boolean;
  error: Error | null;
} {
  const [config, setConfig] = useState<RegistryConfig | null>(cachedConfig);
  const [loading, setLoading] = useState(!cachedConfig);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (cachedConfig) return;

    setLoading(true);
    axios
      .get<RegistryConfig>('/api/config')
      .then((res) => {
        cachedConfig = res.data;
        setConfig(res.data);
        setError(null);
      })
      .catch((err) => {
        console.error('Failed to load registry config:', err);
        setError(err);
        setConfig(DEFAULT_CONFIG);
      })
      .finally(() => setLoading(false));
  }, []);

  return { config, loading, error };
}
