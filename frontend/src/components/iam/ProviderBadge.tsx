import React from 'react';

/**
 * Color treatment per IdP provider. `manual` (hand-registered records) is
 * highlighted; all upstream IdPs share a neutral gray. Shared by the IAM M2M
 * and user-group tables, which previously each carried an identical copy.
 */
const PROVIDER_STYLES: Record<string, string> = {
  manual: 'bg-info-100 dark:bg-info-900/30 text-info-700 dark:text-info-300',
  pingfederate: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  okta: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  auth0: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  keycloak: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  entra: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
};

const ProviderBadge: React.FC<{ provider: string }> = ({ provider }) => {
  const style =
    PROVIDER_STYLES[provider] ??
    'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300';
  return (
    <span className={`inline-block px-2 py-0.5 text-xs rounded-full font-mono ${style}`}>
      {provider}
    </span>
  );
};

export default ProviderBadge;
