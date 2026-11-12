import React from 'react';
import { useRegistryConfig } from '../hooks/useRegistryConfig';

export const DeploymentModeIndicator: React.FC = () => {
  const { config } = useRegistryConfig();

  if (!config || config.deployment_mode === 'with-gateway') {
    return null;
  }

  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-info-100 text-info-800 dark:bg-info-900 dark:text-info-200"
      title="Registry is running without gateway integration. Nginx reverse proxy features are disabled."
    >
      Registry Only
    </span>
  );
};
