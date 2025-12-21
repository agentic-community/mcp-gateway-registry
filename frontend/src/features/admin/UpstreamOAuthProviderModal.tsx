/**
 * UpstreamOAuthProviderModal - Modal for creating/editing upstream OAuth providers (admin)
 */

import { useEffect, useMemo, useState } from 'react';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Textarea } from '@/components/ui/Textarea';
import { Button } from '@/components/ui/Button';
import { useToast } from '@/components/ui/Toast';
import {
  createUpstreamOAuthProvider,
  updateUpstreamOAuthProvider,
} from '@/api/admin';
import type {
  UpstreamOAuthProviderPublic,
  UpstreamOAuthProviderCreateRequest,
  UpstreamOAuthProviderUpdateRequest,
} from '@/api/types';

function parseScopes(
  raw: string
): string[] {
  return raw
    .split(/[\s,]+/g)
    .map((item) => item.trim())
    .filter(Boolean);
}

function stringifyScopes(
  scopes: string[]
): string {
  return scopes.join(' ');
}

function parseExtraParamsJson(
  raw: string
): Record<string, string> {
  const trimmed = raw.trim();
  if (!trimmed) {
    return {};
  }

  const parsed: unknown = JSON.parse(trimmed);
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('extra_authorize_params must be a JSON object');
  }

  const normalized: Record<string, string> = {};
  for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
    const normalizedKey = String(key).trim();
    const normalizedValue = String(value).trim();
    if (!normalizedKey || !normalizedValue) {
      continue;
    }
    normalized[normalizedKey] = normalizedValue;
  }

  return normalized;
}

function stringifyExtraParamsJson(
  value: Record<string, string>
): string {
  if (!value || Object.keys(value).length === 0) {
    return '';
  }
  return JSON.stringify(value, null, 2);
}

interface UpstreamOAuthProviderModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
  provider?: UpstreamOAuthProviderPublic | null;
}

export function UpstreamOAuthProviderModal({
  open,
  onClose,
  onSuccess,
  provider,
}: UpstreamOAuthProviderModalProps) {
  const { addToast } = useToast();
  const isEditMode = Boolean(provider);

  const [providerId, setProviderId] = useState('');
  const [authorizationEndpoint, setAuthorizationEndpoint] = useState('');
  const [tokenEndpoint, setTokenEndpoint] = useState('');
  const [clientId, setClientId] = useState('');
  const [clientSecret, setClientSecret] = useState('');
  const [defaultScopes, setDefaultScopes] = useState('');
  const [extraAuthorizeParamsJson, setExtraAuthorizeParamsJson] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!open) {
      return;
    }

    if (provider) {
      setProviderId(provider.provider.provider_id);
      setAuthorizationEndpoint(provider.provider.authorization_endpoint);
      setTokenEndpoint(provider.provider.token_endpoint);
      setClientId(provider.provider.client_id);
      setClientSecret('');
      setDefaultScopes(stringifyScopes(provider.provider.default_scopes));
      setExtraAuthorizeParamsJson(
        stringifyExtraParamsJson(provider.provider.extra_authorize_params)
      );
    } else {
      setProviderId('');
      setAuthorizationEndpoint('');
      setTokenEndpoint('');
      setClientId('');
      setClientSecret('');
      setDefaultScopes('');
      setExtraAuthorizeParamsJson('');
    }

    setErrors({});
  }, [open, provider]);

  const secretHelperText = useMemo(() => {
    if (!isEditMode) {
      return 'Client secret is required and will never be shown again.';
    }
    return 'Leave blank to keep the existing secret. Enter a new value to rotate.';
  }, [isEditMode]);

  const validate = (): boolean => {
    const nextErrors: Record<string, string> = {};

    if (!providerId.trim()) {
      nextErrors.provider_id = 'Provider ID is required';
    }
    if (!authorizationEndpoint.trim()) {
      nextErrors.authorization_endpoint = 'Authorization endpoint is required';
    }
    if (!tokenEndpoint.trim()) {
      nextErrors.token_endpoint = 'Token endpoint is required';
    }
    if (!clientId.trim()) {
      nextErrors.client_id = 'Client ID is required';
    }
    if (!isEditMode && !clientSecret.trim()) {
      nextErrors.client_secret = 'Client secret is required';
    }

    try {
      parseExtraParamsJson(extraAuthorizeParamsJson);
    } catch (err: any) {
      nextErrors.extra_authorize_params = err?.message || 'Invalid JSON';
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validate()) {
      return;
    }

    setIsSubmitting(true);
    try {
      const scopes = parseScopes(defaultScopes);
      const extraParams = parseExtraParamsJson(extraAuthorizeParamsJson);

      if (isEditMode && provider) {
        const update: UpstreamOAuthProviderUpdateRequest = {
          authorization_endpoint: authorizationEndpoint.trim(),
          token_endpoint: tokenEndpoint.trim(),
          client_id: clientId.trim(),
          default_scopes: scopes,
          extra_authorize_params: extraParams,
        };

        if (clientSecret.trim()) {
          update.client_secret = clientSecret.trim();
        }

        await updateUpstreamOAuthProvider(provider.provider.provider_id, update);
        addToast({
          type: 'success',
          title: 'Provider updated',
          message: `Provider "${provider.provider.provider_id}" has been updated`,
        });
      } else {
        const create: UpstreamOAuthProviderCreateRequest = {
          provider_id: providerId.trim(),
          authorization_endpoint: authorizationEndpoint.trim(),
          token_endpoint: tokenEndpoint.trim(),
          client_id: clientId.trim(),
          client_secret: clientSecret.trim(),
          default_scopes: scopes,
          extra_authorize_params: extraParams,
        };

        await createUpstreamOAuthProvider(create);
        addToast({
          type: 'success',
          title: 'Provider created',
          message: `Provider "${providerId.trim()}" has been created`,
        });
      }

      onSuccess();
      onClose();
    } catch (err: any) {
      addToast({
        type: 'error',
        title: isEditMode ? 'Failed to update provider' : 'Failed to create provider',
        message: err?.message || 'An error occurred',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={isEditMode ? 'Edit Upstream OAuth Provider' : 'Create Upstream OAuth Provider'}
      size="lg"
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <p className="text-sm text-blue-700 dark:text-blue-400">
            Provider client secrets are write-only and encrypted at rest. The UI never displays
            stored secrets after creation.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Input
            label="Provider ID"
            value={providerId}
            onChange={(e) => setProviderId(e.target.value)}
            error={errors.provider_id}
            placeholder="github"
            disabled={isEditMode}
            required={!isEditMode}
          />

          <Input
            label="Client ID"
            value={clientId}
            onChange={(e) => setClientId(e.target.value)}
            error={errors.client_id}
            placeholder="client-id"
            required
          />
        </div>

        <div className="grid grid-cols-1 gap-4">
          <Input
            label="Authorization Endpoint"
            value={authorizationEndpoint}
            onChange={(e) => setAuthorizationEndpoint(e.target.value)}
            error={errors.authorization_endpoint}
            placeholder="https://provider.example/oauth/authorize"
            required
          />

          <Input
            label="Token Endpoint"
            value={tokenEndpoint}
            onChange={(e) => setTokenEndpoint(e.target.value)}
            error={errors.token_endpoint}
            placeholder="https://provider.example/oauth/token"
            required
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Input
            label="Client Secret"
            value={clientSecret}
            onChange={(e) => setClientSecret(e.target.value)}
            error={errors.client_secret}
            placeholder={isEditMode ? '(unchanged)' : 'client-secret'}
            type="password"
            helperText={secretHelperText}
            required={!isEditMode}
          />

          <Input
            label="Default Scopes"
            value={defaultScopes}
            onChange={(e) => setDefaultScopes(e.target.value)}
            placeholder="repo user:email"
            helperText="Space- or comma-separated."
          />
        </div>

        <Textarea
          label="Extra Authorize Params (JSON)"
          value={extraAuthorizeParamsJson}
          onChange={(e) => setExtraAuthorizeParamsJson(e.target.value)}
          error={errors.extra_authorize_params}
          placeholder='{"prompt":"consent"}'
          rows={6}
        />

        <ModalFooter>
          <Button variant="secondary" type="button" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button variant="primary" type="submit" loading={isSubmitting}>
            {isEditMode ? 'Save Changes' : 'Create Provider'}
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}

