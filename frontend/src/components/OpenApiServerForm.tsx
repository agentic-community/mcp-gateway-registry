import React, { useState } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import {
  OpenApiServer,
  OpenApiApiError,
  OpenApiVisibility,
  CreateOpenApiServerRequest,
  UpdateOpenApiServerRequest,
} from '../types/openApiServer';
import { parseOpenApiError } from '../hooks/useOpenApiServers';
import { pathFromName } from '../utils/slug';
import useEscapeKey from '../hooks/useEscapeKey';


/**
 * Props for the OpenApiServerForm component.
 */
interface OpenApiServerFormProps {
  /** The server to edit, or null to create one. */
  server?: OpenApiServer | null;
  onSave: (
    data: CreateOpenApiServerRequest | UpdateOpenApiServerRequest,
  ) => Promise<void>;
  onCancel: () => void;
}


const VISIBILITY_OPTIONS: Array<{ value: OpenApiVisibility; label: string }> = [
  { value: 'private', label: 'Private (admins only)' },
  { value: 'group-restricted', label: 'Restricted to groups' },
  { value: 'public', label: 'Public (everyone)' },
];


/** Split a comma-separated input into trimmed, non-empty values. */
function _splitList(value: string): string[] {
  return value
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean);
}


/** Errors that belong beside the name field rather than in the banner. */
function _isNameError(error: OpenApiApiError): boolean {
  return error.code === 'duplicate_name' || (error.status === 400 && !error.code);
}


/** The banner text for an error that is not shown beside a field. */
function _bannerText(error: OpenApiApiError): string {
  if (error.code === 'limit_reached') {
    const limit = error.limit !== undefined ? ` of ${error.limit}` : '';
    return `The registry already has the maximum${limit} OpenAPI MCP servers ` +
      `(OPENAPI_MCP_MAX_SERVERS). Delete one, or ask an operator to raise the limit. ` +
      error.message;
  }
  if (error.code === 'container_unavailable') {
    return `The openapi-mcp container did not accept the server, so nothing was ` +
      `created. Check the container and try again. ${error.message}`;
  }
  return error.message;
}


/**
 * OpenApiServerForm creates or edits an OpenAPI MCP server.
 *
 * On create it previews the path the name produces. On edit the name is
 * disabled, because the path cannot change.
 */
const OpenApiServerForm: React.FC<OpenApiServerFormProps> = ({
  server,
  onSave,
  onCancel,
}) => {
  const isEditMode = !!server;

  const [name, setName] = useState(server?.name || '');
  const [description, setDescription] = useState(server?.description || '');
  // The registry adds the "openapi" tag itself, so the form does not show it.
  const [tags, setTags] = useState(
    (server?.tags || []).filter((t) => t !== 'openapi').join(', '),
  );
  const [visibility, setVisibility] = useState<OpenApiVisibility>(
    server?.visibility || 'private',
  );
  const [allowedGroups, setAllowedGroups] = useState(
    (server?.allowed_groups || []).join(', '),
  );
  const [saving, setSaving] = useState(false);
  const [apiError, setApiError] = useState<OpenApiApiError | null>(null);
  const [nameError, setNameError] = useState<string | null>(null);

  useEscapeKey(onCancel, !saving);

  const pathPreview = isEditMode ? server?.path || '' : pathFromName(name);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setApiError(null);
    setNameError(null);

    if (!isEditMode && !name.trim()) {
      setNameError('Enter a name. It becomes the server path.');
      return;
    }
    if (visibility === 'group-restricted' && _splitList(allowedGroups).length === 0) {
      setApiError({
        status: null,
        code: null,
        message: 'Name at least one group, or choose another visibility.',
      });
      return;
    }

    const common = {
      description: description.trim(),
      tags: _splitList(tags),
      visibility,
      allowed_groups: visibility === 'group-restricted' ? _splitList(allowedGroups) : [],
    };

    setSaving(true);
    try {
      if (isEditMode) {
        await onSave(common);
      } else {
        await onSave({ name: name.trim(), ...common });
      }
    } catch (err: unknown) {
      const parsed = parseOpenApiError(err, 'Failed to save the OpenAPI MCP server');
      if (_isNameError(parsed)) {
        setNameError(parsed.message);
      } else {
        setApiError(parsed);
      }
    } finally {
      setSaving(false);
    }
  };

  const title = isEditMode ? 'Edit OpenAPI MCP Server' : 'New OpenAPI MCP Server';

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <form
        onSubmit={handleSubmit}
        className="bg-surface-overlay border border-line rounded-xl shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] flex flex-col"
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-line flex-shrink-0">
          <h2 className="text-lg font-semibold text-fg">{title}</h2>
          <button type="button" onClick={onCancel} className="btn-icon" aria-label="Close">
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {apiError && (
            <div
              role="alert"
              className="p-3 bg-danger-50 dark:bg-danger-900/20 border border-danger-200 dark:border-danger-800 rounded-lg"
            >
              <p className="text-sm text-danger-700 dark:text-danger-300">{_bannerText(apiError)}</p>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="openapi-server-name" className="block text-sm font-medium text-fg-muted mb-1">
                Name <span className="text-danger-500">*</span>
              </label>
              <input
                id="openapi-server-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Weather APIs"
                disabled={isEditMode}
                maxLength={64}
                className="input"
              />
              {pathPreview && (
                <p className="mt-1 text-xs text-fg-subtle" data-testid="openapi-path-preview">
                  Path: <code className="font-mono">{pathPreview}</code>
                  {isEditMode && ' (the path cannot change)'}
                </p>
              )}
              {nameError && (
                <p className="mt-1 text-xs text-danger-600 dark:text-danger-400" data-testid="openapi-name-error">
                  {nameError}
                </p>
              )}
            </div>

            <div>
              <label htmlFor="openapi-server-visibility" className="block text-sm font-medium text-fg-muted mb-1">
                Visibility
              </label>
              <select
                id="openapi-server-visibility"
                value={visibility}
                onChange={(e) => setVisibility(e.target.value as OpenApiVisibility)}
                className="input"
              >
                {VISIBILITY_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
              <p className="mt-1 text-xs text-fg-subtle">
                Each API on this server is called with one shared credential. Open it to
                groups or everyone when it is ready.
              </p>
            </div>
          </div>

          {visibility === 'group-restricted' && (
            <div>
              <label htmlFor="openapi-server-groups" className="block text-sm font-medium text-fg-muted mb-1">
                Allowed groups
              </label>
              <input
                id="openapi-server-groups"
                type="text"
                value={allowedGroups}
                onChange={(e) => setAllowedGroups(e.target.value)}
                placeholder="weather-team, data-science (comma-separated)"
                className="input"
              />
            </div>
          )}

          <div>
            <label htmlFor="openapi-server-description" className="block text-sm font-medium text-fg-muted mb-1">
              Description
            </label>
            <textarea
              id="openapi-server-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What the APIs on this server do"
              rows={3}
              maxLength={1000}
              className="input"
            />
          </div>

          <div>
            <label htmlFor="openapi-server-tags" className="block text-sm font-medium text-fg-muted mb-1">
              Tags
            </label>
            <input
              id="openapi-server-tags"
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="weather, forecast (comma-separated)"
              className="input"
            />
          </div>
        </div>

        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-line flex-shrink-0">
          <button type="button" onClick={onCancel} disabled={saving} className="btn-secondary">
            Cancel
          </button>
          <button type="submit" disabled={saving} className="btn-primary">
            {saving ? 'Saving...' : isEditMode ? 'Save Changes' : 'Create Server'}
          </button>
        </div>
      </form>
    </div>
  );
};


export default OpenApiServerForm;
