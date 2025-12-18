import React from 'react';
import { Badge } from './ui/Badge';
import type {
  UpstreamAuthType,
  UpstreamCredentialStatus,
} from '../api/types';

interface UpstreamAuthBadgeProps {
  authType: UpstreamAuthType;
  credentialStatus?: UpstreamCredentialStatus;
  provider?: string;
  compact?: boolean;
}

/**
 * Display upstream authentication type and credential status
 */
export const UpstreamAuthBadge: React.FC<UpstreamAuthBadgeProps> = ({
  authType,
  credentialStatus,
  provider,
  compact = false,
}) => {
  // Map auth type to display text
  const getAuthTypeLabel = (): string => {
    switch (authType) {
      case 'none':
        return 'No Auth';
      case 'api-key':
        return 'API Key';
      case 'oauth2':
        return provider ? `OAuth2 (${provider})` : 'OAuth2';
      case 'oidc':
        return 'OIDC';
      case 'provider-oauth':
        return provider ? `OAuth (${provider})` : 'OAuth';
      case 'jwt':
        return 'JWT';
      case 'mtls':
        return 'mTLS';
      case 'header-trust':
        return 'Header Trust';
      default:
        return 'Unknown';
    }
  };

  // Map credential status to badge variant
  const getStatusVariant = ():
    | 'success'
    | 'warning'
    | 'error'
    | 'neutral'
    | 'info' => {
    if (!credentialStatus) return 'neutral';
    switch (credentialStatus) {
      case 'configured':
        return 'success';
      case 'missing':
        return 'warning';
      case 'expired':
        return 'error';
      case 'revoked':
        return 'error';
      default:
        return 'neutral';
    }
  };

  // Map credential status to display text
  const getStatusLabel = (): string | null => {
    if (!credentialStatus) return null;
    switch (credentialStatus) {
      case 'configured':
        return 'Configured';
      case 'missing':
        return 'Missing';
      case 'expired':
        return 'Expired';
      case 'revoked':
        return 'Revoked';
      default:
        return null;
    }
  };

  const authLabel = getAuthTypeLabel();
  const statusLabel = getStatusLabel();
  const statusVariant = getStatusVariant();

  if (compact) {
    // Compact mode: single badge with status color
    return (
      <Badge variant={statusVariant} size="sm">
        {authLabel}
        {statusLabel && ` • ${statusLabel}`}
      </Badge>
    );
  }

  // Full mode: separate badges for auth type and status
  return (
    <div className="flex items-center gap-1.5">
      <Badge variant="info" size="sm">
        Upstream: {authLabel}
      </Badge>
      {statusLabel && (
        <Badge variant={statusVariant} size="sm">
          {statusLabel}
        </Badge>
      )}
    </div>
  );
};

export default UpstreamAuthBadge;
