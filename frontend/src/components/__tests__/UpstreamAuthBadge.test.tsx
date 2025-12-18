import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import UpstreamAuthBadge from '../UpstreamAuthBadge';
import type { UpstreamAuthType, UpstreamCredentialStatus } from '@/api/types';

describe('UpstreamAuthBadge', () => {
  describe('auth type labels', () => {
    it('should display "No Auth" for none type', () => {
      render(<UpstreamAuthBadge authType="none" />);
      expect(screen.getByText(/No Auth/i)).toBeInTheDocument();
    });

    it('should display "API Key" for api-key type', () => {
      render(<UpstreamAuthBadge authType="api-key" />);
      expect(screen.getByText(/API Key/i)).toBeInTheDocument();
    });

    it('should display "OAuth2" for oauth2 type', () => {
      render(<UpstreamAuthBadge authType="oauth2" />);
      expect(screen.getByText(/OAuth2/i)).toBeInTheDocument();
    });

    it('should display "OAuth2 (provider)" when provider is specified', () => {
      render(<UpstreamAuthBadge authType="oauth2" provider="github" />);
      expect(screen.getByText(/OAuth2 \(github\)/i)).toBeInTheDocument();
    });

    it('should display "OIDC" for oidc type', () => {
      render(<UpstreamAuthBadge authType="oidc" />);
      expect(screen.getByText(/OIDC/i)).toBeInTheDocument();
    });

    it('should display "JWT" for jwt type', () => {
      render(<UpstreamAuthBadge authType="jwt" />);
      expect(screen.getByText(/JWT/i)).toBeInTheDocument();
    });

    it('should display "mTLS" for mtls type', () => {
      render(<UpstreamAuthBadge authType="mtls" />);
      expect(screen.getByText(/mTLS/i)).toBeInTheDocument();
    });

    it('should display "Header Trust" for header-trust type', () => {
      render(<UpstreamAuthBadge authType="header-trust" />);
      expect(screen.getByText(/Header Trust/i)).toBeInTheDocument();
    });
  });

  describe('credential status', () => {
    it('should display "Configured" status', () => {
      render(
        <UpstreamAuthBadge
          authType="api-key"
          credentialStatus="configured"
        />
      );
      expect(screen.getByText('Configured')).toBeInTheDocument();
    });

    it('should display "Missing" status', () => {
      render(
        <UpstreamAuthBadge
          authType="oauth2"
          credentialStatus="missing"
        />
      );
      expect(screen.getByText('Missing')).toBeInTheDocument();
    });

    it('should display "Expired" status', () => {
      render(
        <UpstreamAuthBadge
          authType="jwt"
          credentialStatus="expired"
        />
      );
      expect(screen.getByText('Expired')).toBeInTheDocument();
    });

    it('should display "Revoked" status', () => {
      render(
        <UpstreamAuthBadge
          authType="api-key"
          credentialStatus="revoked"
        />
      );
      expect(screen.getByText('Revoked')).toBeInTheDocument();
    });

    it('should not display status when not provided', () => {
      render(<UpstreamAuthBadge authType="api-key" />);
      expect(screen.queryByText('Configured')).not.toBeInTheDocument();
      expect(screen.queryByText('Missing')).not.toBeInTheDocument();
      expect(screen.queryByText('Expired')).not.toBeInTheDocument();
      expect(screen.queryByText('Revoked')).not.toBeInTheDocument();
    });
  });

  describe('compact mode', () => {
    it('should display combined badge in compact mode', () => {
      render(
        <UpstreamAuthBadge
          authType="api-key"
          credentialStatus="configured"
          compact={true}
        />
      );
      expect(screen.getByText(/API Key • Configured/i)).toBeInTheDocument();
    });

    it('should display only auth type in compact mode without status', () => {
      render(
        <UpstreamAuthBadge
          authType="oauth2"
          compact={true}
        />
      );
      expect(screen.getByText(/OAuth2/i)).toBeInTheDocument();
      expect(screen.queryByText(/•/)).not.toBeInTheDocument();
    });
  });

  describe('full mode', () => {
    it('should display separate badges in full mode', () => {
      render(
        <UpstreamAuthBadge
          authType="oauth2"
          credentialStatus="missing"
          compact={false}
        />
      );
      expect(screen.getByText(/Upstream: OAuth2/i)).toBeInTheDocument();
      expect(screen.getByText('Missing')).toBeInTheDocument();
    });
  });

  describe('all auth types with all statuses', () => {
    const authTypes: UpstreamAuthType[] = [
      'none',
      'api-key',
      'oauth2',
      'oidc',
      'provider-oauth',
      'jwt',
      'mtls',
      'header-trust',
    ];
    const statuses: UpstreamCredentialStatus[] = [
      'configured',
      'missing',
      'expired',
      'revoked',
    ];

    authTypes.forEach((authType) => {
      statuses.forEach((status) => {
        it(`should render ${authType} with ${status} status`, () => {
          const { container } = render(
            <UpstreamAuthBadge
              authType={authType}
              credentialStatus={status}
            />
          );
          expect(container.firstChild).toBeInTheDocument();
        });
      });
    });
  });
});
