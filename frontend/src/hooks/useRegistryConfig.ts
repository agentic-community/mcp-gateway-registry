import { useState, useEffect } from 'react';
import axios from 'axios';

interface RegistryConfig {
  deployment_mode: 'with-gateway' | 'registry-only';
  registry_mode: 'full' | 'skills-only' | 'mcp-servers-only' | 'agents-only';
  auth_provider: string;
  nginx_updates_enabled: boolean;
  coding_assistants: string[];
  dedup_registration_hint_enabled: boolean;
  features: {
    mcp_servers: boolean;
    agents: boolean;
    skills: boolean;
    virtual_servers: boolean;
    federation: boolean;
    gateway_proxy: boolean;
  };
}

const DEFAULT_CONFIG: RegistryConfig = {
  deployment_mode: 'with-gateway',
  registry_mode: 'full',
  auth_provider: 'cognito',
  nginx_updates_enabled: true,
  coding_assistants: [],
  dedup_registration_hint_enabled: false,
  features: {
    mcp_servers: true,
    agents: true,
    skills: true,
    virtual_servers: true,
    federation: true,
    gateway_proxy: true,
  },
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
