import React from 'react';
import FormField from './FormField';
import { fieldClass, FIELD_FOCUS } from './formClasses';

/** One upstream custom header row in the editor. */
export interface UpstreamHeader {
  name: string;
  /** Operator value. Blank = no default (caller-only when overridable). */
  value: string;
  /** When true the CALLER may supply/override this header on the request. */
  overridable: boolean;
}

interface UpstreamHeadersFieldProps {
  headers: UpstreamHeader[];
  onChange: (headers: UpstreamHeader[]) => void;
  accent?: keyof typeof FIELD_FOCUS;
  /** Max rows (mirrors the backend MAX_CUSTOM_HEADERS_PER_SERVER = 10). */
  max?: number;
  /**
   * On edit, values are write-only (the backend never returns them). We show the
   * registered names with an empty value and a "keep existing" placeholder so the
   * operator can see/rotate them without the plaintext ever being echoed back.
   */
  editMode?: boolean;
}

const MAX_DEFAULT = 10;

// Header names the gateway manages; a custom upstream header may not use them.
// Mirrors registry/constants.py RESERVED_CUSTOM_HEADER_NAMES (the security
// boundary is the backend — this is a fail-fast UX hint, not the enforcement).
// Authorization is intentionally NOT reserved here: it is allowed as a
// caller-overridable header (see the overridable toggle below).
const RESERVED_LOWER = new Set(
  [
    'x-authorization',
    'proxy-authorization',
    'content-type',
    'content-length',
    'accept',
    'host',
    'connection',
    'keep-alive',
    'te',
    'trailer',
    'transfer-encoding',
    'upgrade',
    'x-forwarded-for',
    'x-forwarded-proto',
    'x-forwarded-host',
    'x-real-ip',
    'cookie',
    'set-cookie',
    'x-user',
    'x-username',
    'x-scopes',
    'x-auth-method',
    'x-groups',
    'x-client-id',
    'x-user-pool-id',
    'x-region',
    'x-original-url',
    'x-server-name',
    'x-tool-name',
    'x-internal-token',
    'x-internal-token-generic',
    'x-internal-token-registry',
    'x-generic-proxy-kind',
    'x-generic-streaming',
    'x-generic-has-upstream-auth',
    'x-resolved-upstream',
    'x-resolved-generic-upstream',
    'x-upstream-url',
    'x-body',
    'x-body-uninspectable',
  ],
);

/**
 * Compute the inline validation message for a single header row, or null when
 * the row is valid. Exported so callers can GATE submit on the same policy the
 * field shows inline (the backend re-validates — it is the security boundary).
 *
 * On edit, values are write-only: a blank value means "keep the stored value",
 * so a value-less fixed header is legitimate (its ciphertext persists). Pass
 * `editMode` to suppress the "fixed header needs a value" requirement; the
 * reserved-name and fixed-Authorization checks still apply.
 */
export function upstreamHeaderRowError(
  h: UpstreamHeader,
  editMode = false,
): string | null {
  const name = h.name.trim();
  if (!name) {
    // An all-blank row is ignored on submit, not an error.
    return h.value.trim() || h.overridable ? 'Header name is required.' : null;
  }
  const lower = name.toLowerCase();
  if (lower === 'authorization' && !h.overridable) {
    return 'Authorization is only allowed as a caller-overridable header — enable "Caller can override", or use the egress credential vault for a fixed token.';
  }
  if (RESERVED_LOWER.has(lower)) {
    return `"${name}" is managed by the gateway and cannot be a custom upstream header.`;
  }
  if (!h.value.trim() && !h.overridable && !editMode) {
    return 'A fixed header needs a value (or enable "Caller can override" for a caller-supplied slot).';
  }
  return null;
}

/**
 * Editor for a proxied entity's upstream custom headers, using the per-header
 * `overridable` model:
 *
 * - value + not overridable  = FIXED operator credential (immutable by callers).
 * - value + overridable      = operator DEFAULT the caller may override.
 * - no value + overridable   = caller-only passthrough slot.
 *
 * Shared across the skill / custom-entity forms so the editor behaves
 * identically. The backend re-validates the full policy (reserved-name deny,
 * count cap, fixed-Authorization reject); this control is a convenience, not the
 * security boundary. Header values are write-only: never echoed back on read.
 */
const UpstreamHeadersField: React.FC<UpstreamHeadersFieldProps> = ({
  headers,
  onChange,
  accent = 'purple',
  max = MAX_DEFAULT,
  editMode = false,
}) => {
  const update = (idx: number, patch: Partial<UpstreamHeader>) => {
    const next = headers.map((h, i) => (i === idx ? { ...h, ...patch } : h));
    onChange(next);
  };
  const remove = (idx: number) => onChange(headers.filter((_, i) => i !== idx));
  const add = () =>
    onChange([...headers, { name: '', value: '', overridable: false }]);

  return (
    <FormField
      label="Upstream headers"
      hint="Headers the gateway presents to the proxied backend. A value is an operator-set header; enable “Caller can override” to let the request supply or replace it (a value-less overridable row is a caller-only slot). Values are encrypted and never shown again."
    >
      <div className="space-y-3">
        {headers.map((h, idx) => {
          const err = upstreamHeaderRowError(h, editMode);
          const isAuth = h.name.trim().toLowerCase() === 'authorization';
          return (
            <div
              key={idx}
              className="rounded-md border border-gray-200 dark:border-gray-700 p-3 space-y-2"
            >
              <div className="flex gap-2">
                <input
                  type="text"
                  aria-label={`Header name (row ${idx + 1})`}
                  placeholder="X-Api-Key"
                  value={h.name}
                  onChange={(e) => update(idx, { name: e.target.value })}
                  className={`flex-1 ${fieldClass(accent, !!err)} text-sm`}
                />
                <button
                  type="button"
                  aria-label={`Remove header (row ${idx + 1})`}
                  onClick={() => remove(idx)}
                  className="px-3 py-2 text-sm text-red-600 hover:text-red-800 dark:text-red-400"
                >
                  Remove
                </button>
              </div>
              <input
                type="password"
                aria-label={`Header value (row ${idx + 1})`}
                placeholder={
                  editMode
                    ? // On edit a blank value keeps the stored value (write-only).
                      // To drop a stored value entirely, Remove the row and re-add
                      // it (as an overridable caller-only slot if desired).
                      'Blank = keep stored value'
                    : h.overridable
                      ? 'Optional default value (blank = caller-supplied only)'
                      : 'Header value'
                }
                value={h.value}
                onChange={(e) => update(idx, { value: e.target.value })}
                className={`w-full ${fieldClass(accent)} text-sm`}
              />
              <label className="inline-flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={h.overridable}
                  onChange={(e) => update(idx, { overridable: e.target.checked })}
                  className="h-4 w-4 rounded border-gray-300 dark:border-gray-600"
                />
                <span className="text-xs text-gray-600 dark:text-gray-400">
                  Caller can override / supply this header
                </span>
              </label>
              {isAuth && h.overridable && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  The caller’s Authorization is forwarded to the backend. The
                  gateway credential is never leaked (an equal-token guard rejects
                  a request that reuses it).
                </p>
              )}
              {err && (
                <p className="text-xs text-red-600 dark:text-red-400">{err}</p>
              )}
            </div>
          );
        })}
        {headers.length < max && (
          <button
            type="button"
            onClick={add}
            className="text-sm text-purple-600 hover:text-purple-800 dark:text-purple-400"
          >
            + Add header
          </button>
        )}
      </div>
    </FormField>
  );
};

export default UpstreamHeadersField;
