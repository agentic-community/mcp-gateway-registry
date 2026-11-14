import React, { useState } from 'react';
import {
  MyCredentialStatus,
  OpenApiKeyTtlUnit,
} from '../types/openApiServer';
import {
  setMyCredential,
  deleteMyCredential,
  parseOpenApiError,
} from '../hooks/useOpenApiServers';


type ToastFn = (message: string, type: 'success' | 'error' | 'info') => void;


/** The default lifetime of a new key: 30 days, the maximum the backend allows. */
const DEFAULT_TTL_VALUE: number = 30;
const DEFAULT_TTL_UNIT: OpenApiKeyTtlUnit = 'days';


/** The largest lifetime per unit. Each equals 30 days. */
const MAX_TTL: Record<OpenApiKeyTtlUnit, number> = {
  minutes: 30 * 24 * 60,
  hours: 30 * 24,
  days: 30,
};


/** One sentence per error code of the my-credential routes. */
const KEY_ERROR_SENTENCES: Record<string, string> = {
  egress_auth_disabled: 'Per-user keys are turned off on this deployment (egress auth is disabled). Ask an administrator.',
  not_per_user: 'This API uses a shared credential, so it does not take your own key.',
};


/** The text to show for a failed key write. Never includes the key. */
export function keyErrorText(
  err: unknown,
  fallback: string,
): string {
  const parsed = parseOpenApiError(err, fallback);
  const sentence = parsed.code ? KEY_ERROR_SENTENCES[parsed.code] : undefined;
  return sentence || parsed.message;
}


/** "Set, expires ..." or "Expired ..." or "Not set". */
function _statusText(status: MyCredentialStatus | null): string {
  if (!status || !status.configured) {
    return 'Not set';
  }
  if (!status.expires_at) {
    return 'Set';
  }
  const expires = new Date(status.expires_at);
  if (Number.isNaN(expires.getTime())) {
    return 'Set';
  }
  if (expires.getTime() < Date.now()) {
    return `Expired ${expires.toLocaleString()}`;
  }
  return `Set, expires ${expires.toLocaleString()}`;
}


interface KeyFormProps {
  targetName: string;
  saving: boolean;
  onSubmit: (secret: string, ttlValue: number, ttlUnit: OpenApiKeyTtlUnit) => void;
  onCancel: () => void;
}


/** The key field and lifetime. The key always starts empty. */
const _KeyForm: React.FC<KeyFormProps> = ({ targetName, saving, onSubmit, onCancel }) => {
  const [secret, setSecret] = useState('');
  const [ttlValue, setTtlValue] = useState(DEFAULT_TTL_VALUE);
  const [ttlUnit, setTtlUnit] = useState<OpenApiKeyTtlUnit>(DEFAULT_TTL_UNIT);
  const [error, setError] = useState<string | null>(null);
  const keyId = `openapi-my-key-${targetName}`;
  const ttlId = `openapi-my-key-ttl-${targetName}`;
  const unitId = `openapi-my-key-unit-${targetName}`;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!secret.trim()) {
      setError('Enter your key.');
      return;
    }
    if (!Number.isInteger(ttlValue) || ttlValue < 1 || ttlValue > MAX_TTL[ttlUnit]) {
      setError(`Enter a lifetime from 1 to ${MAX_TTL[ttlUnit]} ${ttlUnit} (30 days at most).`);
      return;
    }
    setError(null);
    onSubmit(secret.trim(), ttlValue, ttlUnit);
    // The key is write-only: drop it from state as soon as it is sent.
    setSecret('');
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-2" data-testid={`openapi-my-key-form-${targetName}`}>
      <div>
        <label htmlFor={keyId} className="block text-xs font-medium text-fg-muted mb-1">
          Your key for {targetName}
        </label>
        <input id={keyId} type="password" value={secret} autoComplete="new-password"
          onChange={(e) => setSecret(e.target.value)} placeholder="Paste the key from the API provider"
          className="input text-sm" />
      </div>
      <div className="flex items-end gap-2">
        <div>
          <label htmlFor={ttlId} className="block text-xs font-medium text-fg-muted mb-1">Valid for</label>
          <input id={ttlId} type="number" min={1} value={ttlValue}
            onChange={(e) => setTtlValue(parseInt(e.target.value, 10) || 0)} className="input text-sm w-24" />
        </div>
        <div>
          <label htmlFor={unitId} className="block text-xs font-medium text-fg-muted mb-1">Unit</label>
          <select id={unitId} value={ttlUnit} className="input text-sm w-auto"
            onChange={(e) => setTtlUnit(e.target.value as OpenApiKeyTtlUnit)}>
            <option value="minutes">Minutes</option>
            <option value="hours">Hours</option>
            <option value="days">Days</option>
          </select>
        </div>
        <button type="submit" disabled={saving} className="btn-primary text-sm">
          {saving ? 'Saving...' : 'Save key'}
        </button>
        <button type="button" onClick={onCancel} disabled={saving} className="btn-secondary text-sm">
          Cancel
        </button>
      </div>
      <p className="text-xs text-fg-subtle">
        Stored write-only and never shown again. The longest lifetime is 30 days.
      </p>
      {error && <p className="text-xs text-danger-600 dark:text-danger-400">{error}</p>}
    </form>
  );
};


interface OpenApiMyKeyControlProps {
  slug: string;
  targetName: string;
  /** The caller's key status, or null while it is unknown. */
  status: MyCredentialStatus | null;
  /** Called after a key is saved or removed, so the caller can reload. */
  onChanged: () => void;
  onShowToast?: ToastFn;
}


/**
 * The current user's own key for one per-user target: its status, Add key or
 * Replace key (opens a small form), and Remove key.
 */
const OpenApiMyKeyControl: React.FC<OpenApiMyKeyControlProps> = ({
  slug,
  targetName,
  status,
  onChanged,
  onShowToast,
}) => {
  const [formOpen, setFormOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const configured = !!status?.configured;

  const handleSave = async (secret: string, ttlValue: number, ttlUnit: OpenApiKeyTtlUnit) => {
    setSaving(true);
    setError(null);
    try {
      await setMyCredential(slug, targetName, { secret, ttl_value: ttlValue, ttl_unit: ttlUnit });
      setFormOpen(false);
      onShowToast?.(`Your key for "${targetName}" is saved.`, 'success');
      onChanged();
    } catch (err: unknown) {
      setError(keyErrorText(err, 'Failed to save your key'));
    } finally {
      setSaving(false);
    }
  };

  const handleRemove = async () => {
    setSaving(true);
    setError(null);
    try {
      await deleteMyCredential(slug, targetName);
      onShowToast?.(`Your key for "${targetName}" is removed.`, 'success');
      onChanged();
    } catch (err: unknown) {
      setError(keyErrorText(err, 'Failed to remove your key'));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-2" data-testid={`openapi-my-key-${targetName}`}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-mono text-fg">{targetName}</span>
        <span className="text-xs text-fg-muted" data-testid={`openapi-my-key-status-${targetName}`}>
          {_statusText(status)}
        </span>
        {!formOpen && (
          <button type="button" onClick={() => setFormOpen(true)} disabled={saving}
            className="px-2 py-1 text-xs font-medium rounded border border-line hover:bg-surface-hover text-fg-muted">
            {configured ? 'Replace key' : 'Add key'}
          </button>
        )}
        {configured && !formOpen && (
          <button type="button" onClick={handleRemove} disabled={saving}
            className="px-2 py-1 text-xs font-medium rounded text-danger-700 dark:text-danger-300 hover:bg-danger-50 dark:hover:bg-danger-900/20">
            Remove key
          </button>
        )}
      </div>
      {formOpen && (
        <_KeyForm targetName={targetName} saving={saving} onSubmit={handleSave}
          onCancel={() => { setFormOpen(false); setError(null); }} />
      )}
      {error && (
        <p role="alert" className="text-xs text-danger-600 dark:text-danger-400">{error}</p>
      )}
    </div>
  );
};


export default OpenApiMyKeyControl;
