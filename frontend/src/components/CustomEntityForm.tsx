import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { XMarkIcon } from '@heroicons/react/24/outline';
import {
  CustomEntityCreate,
  CustomEntityFieldError,
  CustomEntityRecord,
  CustomEntityUpdate,
  CustomFieldDescriptor,
  CustomTypeDescriptor,
  MAX_STRING_LEN,
  MAX_TEXT_LEN,
} from '../types/customEntity';
import { labelFor } from '../utils/humanize';
import {
  FormField,
  fieldClass,
  ProxyField,
  UpstreamHeadersField,
  type UpstreamHeader,
} from './formFields';

interface CustomEntityFormProps {
  descriptor: CustomTypeDescriptor;
  /** Existing record for edit mode; null/undefined for create. */
  record?: CustomEntityRecord | null;
  /**
   * Persist the record. ``customHeaders`` carries the upstream-header editor
   * rows (only when proxied); the parent routes them to the create payload
   * (create) or the dedicated rotation PATCH (edit). Undefined = not proxied /
   * no header edits.
   */
  onSave: (
    body: CustomEntityCreate | CustomEntityUpdate,
    customHeaders?: UpstreamHeader[],
  ) => Promise<void>;
  onCancel: () => void;
}

const VISIBILITIES = ['public', 'private', 'group-restricted'] as const;

// Gateway-proxy keys live in the attributes bag on the wire but are edited via
// the dedicated ProxyField widget. If an admin's custom-type descriptor happens
// to define a field with one of these names, skip its descriptor-driven widget
// so it can't collide with (and clobber) the ProxyField-owned value.
const PROXY_RESERVED_ATTR_KEYS = new Set(['is_proxied', 'proxy_target_url']);

// Custom-entity forms use a teal focus accent.
const INPUT_CLASS = fieldClass('teal');

/** Comma/enter chip input — matches the agents allowed_groups widget. */
function ChipInput({
  value,
  onChange,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
}) {
  const [draft, setDraft] = useState('');

  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed && !value.includes(trimmed)) {
      onChange([...value, trimmed]);
    }
    setDraft('');
  };

  return (
    <div>
      <div className="flex flex-wrap gap-1.5 mb-1.5">
        {value.map((chip) => (
          <span
            key={chip}
            className="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-full
                       bg-teal-100 dark:bg-teal-900/30 text-teal-700 dark:text-teal-300"
          >
            {chip}
            <button
              type="button"
              onClick={() => onChange(value.filter((c) => c !== chip))}
              className="hover:text-red-500"
              aria-label={`Remove ${chip}`}
            >
              <XMarkIcon className="h-3 w-3" />
            </button>
          </span>
        ))}
      </div>
      <input
        type="text"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault();
            commit();
          }
        }}
        onBlur={commit}
        placeholder={placeholder}
        className={INPUT_CLASS}
      />
    </div>
  );
}

const CustomEntityForm: React.FC<CustomEntityFormProps> = ({
  descriptor,
  record,
  onSave,
  onCancel,
}) => {
  const isEditMode = !!record;

  // --- envelope state ---
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [visibility, setVisibility] = useState<string>('private');
  const [allowedGroups, setAllowedGroups] = useState<string[]>([]);
  const [tags, setTags] = useState<string[]>([]);

  // --- gateway-proxy opt-in (lives in the attributes bag on the wire) ---
  // A custom entity has no native backend URL, so proxy_target_url is required
  // when is_proxied is true.
  const [isProxied, setIsProxied] = useState(false);
  const [proxyTargetUrl, setProxyTargetUrl] = useState('');
  // Read-only, server-derived client path (present on edit once saved).
  const [proxyClientUrl, setProxyClientUrl] = useState('');
  // Upstream headers (per-header overridable). On edit these are the registered
  // names with blank values (write-only); the backend preserves unchanged ones.
  const [customHeaders, setCustomHeaders] = useState<UpstreamHeader[]>([]);

  // --- attributes state (per-type) ---
  const [attributes, setAttributes] = useState<Record<string, unknown>>({});

  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  // Map of field name -> error message (envelope keys and attribute keys share namespace).
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (record) {
      setName(record.name);
      setDescription(record.description ?? '');
      setVisibility(record.visibility);
      setAllowedGroups(record.allowed_groups ?? []);
      setTags(record.tags ?? []);
      // Proxy fields are TOP-LEVEL (mixin) fields on the wire, edited via the
      // dedicated ProxyField widget. Read them from the record envelope; keep a
      // fallback read from the attributes bag for any record written before the
      // fields moved top-level, and strip them from the descriptor-driven
      // attribute state so no duplicate raw widget can render.
      const attrs = { ...(record.attributes ?? {}) };
      setIsProxied(
        record.is_proxied ?? !!attrs.is_proxied,
      );
      setProxyTargetUrl(
        record.proxy_target_url ??
          (typeof attrs.proxy_target_url === 'string' ? attrs.proxy_target_url : ''),
      );
      setProxyClientUrl(record.proxy_client_url ?? '');
      // Rebuild header editor rows from the registered NAMES (values write-only).
      setCustomHeaders(
        (record.custom_header_names ?? []).map((name) => ({
          name,
          value: '',
          overridable: (record.custom_header_overridable_names ?? []).includes(name),
        })),
      );
      delete attrs.is_proxied;
      delete attrs.proxy_target_url;
      setAttributes(attrs);
    }
  }, [record]);

  const setAttr = (key: string, val: unknown) => {
    setAttributes((prev) => ({ ...prev, [key]: val }));
  };

  // Render one attribute widget from its descriptor. The default arm renders
  // an unknown datatype as read-only text rather than crashing the form.
  const renderField = (field: CustomFieldDescriptor) => {
    const raw = attributes[field.name];
    const err = fieldErrors[field.name];

    let widget: React.ReactNode;
    switch (field.datatype) {
      case 'string':
        widget = (
          <input
            type="text"
            maxLength={MAX_STRING_LEN}
            value={(raw as string) ?? ''}
            onChange={(e) => setAttr(field.name, e.target.value)}
            className={INPUT_CLASS}
          />
        );
        break;
      case 'text':
        widget = (
          <textarea
            rows={4}
            maxLength={MAX_TEXT_LEN}
            value={(raw as string) ?? ''}
            onChange={(e) => setAttr(field.name, e.target.value)}
            className={INPUT_CLASS}
          />
        );
        break;
      case 'number':
        widget = (
          <input
            type="number"
            value={raw === undefined || raw === null ? '' : (raw as number)}
            onChange={(e) => {
              // Guard against NaN: inputs like "1e" or "1.2.3" parse to NaN,
              // which would serialize to null and silently drop the value.
              // Treat any non-finite parse as empty (null) so the field stays
              // predictable rather than confusingly failing server-side.
              const text = e.target.value;
              if (text === '') {
                setAttr(field.name, null);
                return;
              }
              const parsed = Number(text);
              setAttr(field.name, Number.isFinite(parsed) ? parsed : null);
            }}
            className={INPUT_CLASS}
          />
        );
        break;
      case 'bool':
        widget = (
          <input
            type="checkbox"
            checked={!!raw}
            onChange={(e) => setAttr(field.name, e.target.checked)}
            className="h-4 w-4 rounded border-gray-300 text-teal-600 focus:ring-teal-500"
          />
        );
        break;
      case 'enum':
        widget = (
          <select
            value={(raw as string) ?? ''}
            onChange={(e) => setAttr(field.name, e.target.value || null)}
            className={INPUT_CLASS}
          >
            <option value="">— select —</option>
            {(field.enum_values ?? []).map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        );
        break;
      case 'date':
        widget = (
          <input
            type="date"
            value={(raw as string) ?? ''}
            onChange={(e) => setAttr(field.name, e.target.value || null)}
            className={INPUT_CLASS}
          />
        );
        break;
      case 'array<string>':
        widget = (
          <ChipInput
            value={Array.isArray(raw) ? (raw as string[]) : []}
            onChange={(next) => setAttr(field.name, next)}
            placeholder="Type a value, press Enter"
          />
        );
        break;
      default:
        // Unknown datatype (older frontend vs newer descriptor): show read-only.
        widget = (
          <div className="px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-900/50 text-sm text-gray-500 dark:text-gray-400">
            {raw === undefined || raw === null ? '—' : String(raw)}
            <span className="ml-2 italic">(unsupported field type — read only)</span>
          </div>
        );
    }

    return (
      <FormField key={field.name} label={labelFor(field)} required={field.required} error={err}>
        {widget}
      </FormField>
    );
  };

  const handleSubmit = async () => {
    setFormError(null);
    setFieldErrors({});

    if (!name.trim()) {
      setFieldErrors({ name: 'Name is required' });
      return;
    }
    if (visibility === 'group-restricted' && allowedGroups.length === 0) {
      setFieldErrors({ allowed_groups: 'At least one group is required' });
      return;
    }
    // A custom entity has no native URL to fall back on, so a proxied entity
    // must carry an explicit target.
    if (isProxied && !proxyTargetUrl.trim()) {
      setFieldErrors({ proxy_target_url: 'Proxy target URL is required when proxying is enabled' });
      return;
    }

    // Materialize bool field defaults. The checkbox widget renders an untouched
    // bool as unchecked but never writes into `attributes` until toggled, so a
    // bool left at its default false would be OMITTED from the payload — and a
    // required bool would then fail server-side for input that looked valid.
    // Default any absent bool field to false before submitting.
    const attributesWithDefaults = { ...attributes };
    descriptor.fields.forEach((field) => {
      if (PROXY_RESERVED_ATTR_KEYS.has(field.name)) {
        // Owned by the ProxyField; never materialize a descriptor default for
        // these (they are sent top-level, not in the attributes bag).
        return;
      }
      if (
        field.datatype === 'bool' &&
        attributesWithDefaults[field.name] === undefined
      ) {
        attributesWithDefaults[field.name] = false;
      }
    });

    // Gateway-proxy opt-in: TOP-LEVEL (mixin) fields on the wire, NOT in the
    // attributes bag. Always send is_proxied so toggling off persists; only send
    // the target when proxying (proxy_client_url is server-derived, never sent).
    const payload = {
      name: name.trim(),
      description: description.trim() || null,
      visibility,
      allowed_groups: allowedGroups,
      tags,
      attributes: attributesWithDefaults,
      is_proxied: isProxied,
      ...(isProxied ? { proxy_target_url: proxyTargetUrl.trim() } : {}),
    };

    // Upstream headers only apply when proxied. Drop fully-blank rows; a blank
    // value is preserved server-side (write-only UX) on edit, or must be a
    // caller-only overridable slot on create.
    const upstreamHeaders = isProxied
      ? customHeaders
          .filter((h) => h.name.trim())
          .map((h) => ({
            name: h.name.trim(),
            value: h.value,
            overridable: h.overridable,
          }))
      : [];

    setSaving(true);
    try {
      await onSave(payload, upstreamHeaders);
    } catch (err: unknown) {
      const detail = axios.isAxiosError(err) ? err.response?.data?.detail : undefined;
      if (Array.isArray(detail)) {
        // Multi-error 400 body: [{ field, message }, ...] — highlight every field.
        const map: Record<string, string> = {};
        (detail as CustomEntityFieldError[]).forEach((e) => {
          if (e.field) map[e.field] = e.message;
        });
        setFieldErrors(map);
        setFormError('Please fix the highlighted fields.');
      } else {
        const message = err instanceof Error ? err.message : 'Failed to save record';
        setFormError(typeof detail === 'string' ? detail : message);
      }
    } finally {
      setSaving(false);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !saving) onCancel();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onCancel, saving]);

  const typeLabel = descriptor.display_name || labelFor({ name: descriptor.name });

  // Descriptor fields to render as raw widgets, minus any that collide with the
  // ProxyField-owned reserved keys (see PROXY_RESERVED_ATTR_KEYS).
  const renderableFields = descriptor.fields.filter(
    (field) => !PROXY_RESERVED_ATTR_KEYS.has(field.name),
  );

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div
        className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] flex flex-col"
        role="dialog"
        aria-modal="true"
        aria-label={isEditMode ? `Edit ${typeLabel}` : `Create ${typeLabel}`}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            {isEditMode ? `Edit ${typeLabel}` : `Create ${typeLabel}`}
          </h2>
          <button
            onClick={onCancel}
            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg transition-colors"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {formError && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-sm text-red-700 dark:text-red-300">{formError}</p>
            </div>
          )}

          {/* --- Envelope (fixed, separate from attributes) --- */}
          <FormField label="Name" required error={fieldErrors.name}>
            <input
              type="text"
              value={name}
              maxLength={MAX_STRING_LEN}
              onChange={(e) => setName(e.target.value)}
              className={INPUT_CLASS}
            />
          </FormField>

          <FormField label="Description">
            <textarea
              rows={3}
              maxLength={MAX_TEXT_LEN}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className={INPUT_CLASS}
            />
          </FormField>

          <FormField label="Visibility">
            <select
              value={visibility}
              onChange={(e) => setVisibility(e.target.value)}
              className={INPUT_CLASS}
            >
              {VISIBILITIES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </FormField>

          {visibility === 'group-restricted' && (
            <FormField label="Allowed Groups" required error={fieldErrors.allowed_groups}>
              <ChipInput
                value={allowedGroups}
                onChange={setAllowedGroups}
                placeholder="Type a group name, press Enter"
              />
            </FormField>
          )}

          <FormField label="Tags">
            <ChipInput value={tags} onChange={setTags} placeholder="Type a tag, press Enter" />
          </FormField>

          <ProxyField
            isProxied={isProxied}
            onIsProxiedChange={setIsProxied}
            proxyTargetUrl={proxyTargetUrl}
            onProxyTargetUrlChange={setProxyTargetUrl}
            targetRequired
            accent="teal"
            error={fieldErrors.proxy_target_url}
            clientUrl={proxyClientUrl}
          />

          {isProxied && (
            <UpstreamHeadersField
              headers={customHeaders}
              onChange={setCustomHeaders}
              accent="teal"
              editMode={!!record}
            />
          )}

          {/* --- Per-type attributes (descriptor-driven) --- */}
          {renderableFields.length > 0 && (
            <div className="space-y-6 pt-2 border-t border-gray-200 dark:border-gray-700">
              {renderableFields.map(renderField)}
            </div>
          )}
        </div>

        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
          <button
            type="button"
            onClick={onCancel}
            disabled={saving}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300
                       bg-gray-100 dark:bg-gray-700 rounded-lg
                       hover:bg-gray-200 dark:hover:bg-gray-800 transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={saving}
            className="px-4 py-2 text-sm font-medium text-white bg-teal-600 rounded-lg
                       hover:bg-teal-700 transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {saving && (
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            )}
            {isEditMode ? 'Save Changes' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default CustomEntityForm;
