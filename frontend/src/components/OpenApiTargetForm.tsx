import React, { useState } from 'react';
import { XMarkIcon, PlusIcon, TrashIcon } from '@heroicons/react/24/outline';
import {
  OpenApiTarget,
  OpenApiApiError,
  OpenApiAuthScheme,
  OpenApiTargetAuthInput,
  OpenApiTargetWriteResponse,
  CreateOpenApiTargetRequest,
  UpdateOpenApiTargetRequest,
} from '../types/openApiServer';
import {
  addOpenApiTarget,
  updateOpenApiTarget,
  parseOpenApiError,
} from '../hooks/useOpenApiServers';
import useEscapeKey from '../hooks/useEscapeKey';


/** What the form is for: a new target, an edit, or only a credential rotation. */
export type OpenApiTargetFormIntent = 'create' | 'edit' | 'rotate';


interface OpenApiTargetFormProps {
  serverSlug: string;
  /** The target to edit or rotate. Null or missing for create. */
  target?: OpenApiTarget | null;
  intent: OpenApiTargetFormIntent;
  onSaved: (result: OpenApiTargetWriteResponse) => void;
  onCancel: () => void;
}


/** The form fields an API error can be shown beside. */
type FieldKey = 'name' | 'spec_url' | 'base_url' | 'auth' | 'operations' | 'overrides';


interface OverrideRow {
  key: string;
  value: string;
}


const TARGET_NAME_PATTERN: RegExp = /^[a-z][a-z0-9]{1,23}$/;


const SCHEME_OPTIONS: Array<{ value: OpenApiAuthScheme; label: string }> = [
  { value: 'none', label: 'None' },
  { value: 'api_key_header', label: 'API key in a header' },
  { value: 'api_key_query', label: 'API key in a query parameter' },
  { value: 'bearer', label: 'Bearer token' },
  { value: 'basic', label: 'Basic (username and password)' },
];


const CREDENTIAL_LABELS: Record<OpenApiAuthScheme, string> = {
  none: '',
  api_key_header: 'API key',
  api_key_query: 'API key',
  bearer: 'Bearer token',
  basic: 'Password',
};


/** One sentence per error code, shown before the server's own message. */
const ERROR_SENTENCES: Record<string, string> = {
  duplicate_name: 'This server already has an API with this name.',
  url_blocked: 'The registry does not allow calls to this URL.',
  spec_invalid: 'The spec is not a valid OpenAPI 3 document.',
  spec_too_large: 'The spec is larger than the registry accepts.',
  spec_fetch_failed: 'The registry could not fetch the spec.',
  conversion_failed: 'The spec could not be turned into tools.',
  too_many_tools: 'This spec produces more tools than one API may have. Choose the tags or operations to include.',
  tool_name_invalid: 'Some generated tool names are too long or collide. Rename them with tool name overrides.',
};


function _rowsFromMap(map: Record<string, string> | undefined): OverrideRow[] {
  return Object.entries(map || {}).map(([key, value]) => ({ key, value }));
}


function _mapFromRows(rows: OverrideRow[]): Record<string, string> {
  const result: Record<string, string> = {};
  for (const row of rows) {
    if (row.key.trim()) {
      result[row.key.trim()] = row.value;
    }
  }
  return result;
}


function _linesToList(text: string): string[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}


/** Which field an API error belongs beside, or null for the banner. */
function _fieldForError(error: OpenApiApiError): FieldKey | null {
  switch (error.code) {
    case 'duplicate_name':
      return 'name';
    case 'url_blocked':
      return error.field === 'spec_url' ? 'spec_url' : 'base_url';
    case 'spec_invalid':
    case 'spec_too_large':
    case 'spec_fetch_failed':
    case 'conversion_failed':
      return 'spec_url';
    case 'too_many_tools':
      return 'operations';
    case 'tool_name_invalid':
      return 'overrides';
    default:
      return null;
  }
}


function _errorSentence(error: OpenApiApiError): string {
  const sentence = error.code ? ERROR_SENTENCES[error.code] : undefined;
  if (!sentence) {
    return error.message;
  }
  return error.message ? `${sentence} ${error.message}` : sentence;
}


function _bannerText(error: OpenApiApiError): string {
  if (error.code === 'limit_reached') {
    const limit = error.limit !== undefined ? ` of ${error.limit}` : '';
    return `This server already has the maximum${limit} APIs ` +
      `(OPENAPI_MCP_MAX_TARGETS_PER_SERVER). Delete one first. ${error.message}`;
  }
  return error.message;
}


/** Tool names without the target prefix, for the override key suggestions. */
function _toolKeySuggestions(target: OpenApiTarget | null | undefined): string[] {
  if (!target?.tool_names) {
    return [];
  }
  const prefix = `${target.name}_`;
  return target.tool_names.map((n) => (n.startsWith(prefix) ? n.slice(prefix.length) : n));
}


/** Build the auth body. Returns an error message when a required field is empty. */
function _buildAuth(
  scheme: OpenApiAuthScheme,
  headerName: string,
  queryParam: string,
  username: string,
  credential: string,
): { auth: OpenApiTargetAuthInput; error: string | null } {
  if (scheme === 'none') {
    return { auth: { scheme }, error: null };
  }
  const auth: OpenApiTargetAuthInput = { scheme, credential };
  if (scheme === 'api_key_header') {
    auth.header_name = headerName.trim();
    if (!auth.header_name) return { auth, error: 'Enter the header name.' };
  }
  if (scheme === 'api_key_query') {
    auth.query_param = queryParam.trim();
    if (!auth.query_param) return { auth, error: 'Enter the query parameter name.' };
  }
  if (scheme === 'basic') {
    auth.username = username.trim();
    if (!auth.username) return { auth, error: 'Enter the username.' };
  }
  if (!credential) {
    return { auth, error: `Enter the ${CREDENTIAL_LABELS[scheme].toLowerCase()}.` };
  }
  return { auth, error: null };
}


const _FieldError: React.FC<{ field: FieldKey; message?: string }> = ({ field, message }) => {
  if (!message) return null;
  return (
    <p
      className="mt-1 text-xs text-danger-600 dark:text-danger-400"
      data-testid={`openapi-target-error-${field}`}
    >
      {message}
    </p>
  );
};


interface AuthFieldsProps {
  scheme: OpenApiAuthScheme;
  setScheme: (value: OpenApiAuthScheme) => void;
  headerName: string;
  setHeaderName: (value: string) => void;
  queryParam: string;
  setQueryParam: (value: string) => void;
  username: string;
  setUsername: (value: string) => void;
  credential: string;
  setCredential: (value: string) => void;
  credentialSet: boolean;
  error?: string;
}


/** The auth scheme select and only the fields that scheme needs. */
const _AuthFields: React.FC<AuthFieldsProps> = (props) => {
  const { scheme } = props;
  return (
    <div className="space-y-3">
      <div>
        <label htmlFor="openapi-target-scheme" className="block text-sm font-medium text-fg-muted mb-1">
          Auth scheme
        </label>
        <select
          id="openapi-target-scheme"
          value={scheme}
          onChange={(e) => props.setScheme(e.target.value as OpenApiAuthScheme)}
          className="input"
        >
          {SCHEME_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>
      {scheme === 'api_key_header' && (
        <div>
          <label htmlFor="openapi-target-header" className="block text-sm font-medium text-fg-muted mb-1">
            Header name
          </label>
          <input id="openapi-target-header" type="text" value={props.headerName}
            onChange={(e) => props.setHeaderName(e.target.value)} placeholder="X-API-Key"
            maxLength={128} className="input" autoComplete="off" />
        </div>
      )}
      {scheme === 'api_key_query' && (
        <div>
          <label htmlFor="openapi-target-query" className="block text-sm font-medium text-fg-muted mb-1">
            Query parameter
          </label>
          <input id="openapi-target-query" type="text" value={props.queryParam}
            onChange={(e) => props.setQueryParam(e.target.value)} placeholder="api_key"
            maxLength={128} className="input" autoComplete="off" />
        </div>
      )}
      {scheme === 'basic' && (
        <div>
          <label htmlFor="openapi-target-username" className="block text-sm font-medium text-fg-muted mb-1">
            Username
          </label>
          <input id="openapi-target-username" type="text" value={props.username}
            onChange={(e) => props.setUsername(e.target.value)}
            maxLength={256} className="input" autoComplete="off" />
        </div>
      )}
      {scheme !== 'none' && (
        <div>
          <label htmlFor="openapi-target-credential" className="block text-sm font-medium text-fg-muted mb-1">
            {CREDENTIAL_LABELS[scheme]}
          </label>
          <input id="openapi-target-credential" type="password" value={props.credential}
            onChange={(e) => props.setCredential(e.target.value)}
            autoComplete="new-password" className="input"
            placeholder={props.credentialSet ? 'Leave empty to keep the stored credential' : ''} />
          <p className="mt-1 text-xs text-fg-subtle">
            Stored encrypted. The registry never shows it again.
          </p>
        </div>
      )}
      <_FieldError field="auth" message={props.error} />
    </div>
  );
};


interface OperationsSectionProps {
  onlyThese: boolean;
  setOnlyThese: (value: boolean) => void;
  tagCounts: Record<string, number | null>;
  selectedTags: string[];
  toggleTag: (tag: string) => void;
  operationsText: string;
  setOperationsText: (value: string) => void;
  error?: string;
}


/** "All operations" or "Only these", with the spec's tags and an operation list. */
const _OperationsSection: React.FC<OperationsSectionProps> = (props) => {
  const tagNames = Object.keys(props.tagCounts).sort();
  return (
    <fieldset className="space-y-3" data-testid="openapi-operations-section">
      <legend className="block text-sm font-medium text-fg-muted mb-1">Operations</legend>
      <div className="flex gap-6 text-sm text-fg">
        <label className="flex items-center gap-2">
          <input type="radio" name="openapi-ops" checked={!props.onlyThese}
            onChange={() => props.setOnlyThese(false)} />
          All operations
        </label>
        <label className="flex items-center gap-2">
          <input type="radio" name="openapi-ops" checked={props.onlyThese}
            onChange={() => props.setOnlyThese(true)} />
          Only these
        </label>
      </div>
      <_FieldError field="operations" message={props.error} />
      {props.onlyThese && (
        <div className="space-y-3">
          {tagNames.length > 0 ? (
            <div className="flex flex-wrap gap-2" data-testid="openapi-tag-list">
              {tagNames.map((tag) => (
                <label key={tag} className="flex items-center gap-1.5 px-2 py-1 text-xs border border-line rounded">
                  <input type="checkbox" checked={props.selectedTags.includes(tag)}
                    onChange={() => props.toggleTag(tag)} />
                  <span>{tag}</span>
                  {props.tagCounts[tag] !== null && (
                    <span className="text-fg-subtle">({props.tagCounts[tag]})</span>
                  )}
                </label>
              ))}
            </div>
          ) : (
            <p className="text-xs text-fg-subtle">
              The spec&apos;s tags appear here if the spec has more operations than one API may
              have. You can also list tags or operations by hand below.
            </p>
          )}
          <div>
            <label htmlFor="openapi-target-operations" className="block text-xs font-medium text-fg-muted mb-1">
              Operations to include, one per line (an operationId, or METHOD /path)
            </label>
            <textarea id="openapi-target-operations" value={props.operationsText}
              onChange={(e) => props.setOperationsText(e.target.value)} rows={3}
              placeholder={'getInventory\nGET /v1/forecast'} className="input font-mono text-xs" />
          </div>
        </div>
      )}
    </fieldset>
  );
};


interface OverridesEditorProps {
  label: string;
  rows: OverrideRow[];
  setRows: (rows: OverrideRow[]) => void;
  suggestionsId: string;
  valuePlaceholder: string;
}


/** A list of key and value pairs. Keys are suggested from the target's tool names. */
const _OverridesEditor: React.FC<OverridesEditorProps> = ({
  label,
  rows,
  setRows,
  suggestionsId,
  valuePlaceholder,
}) => {
  const updateRow = (index: number, change: Partial<OverrideRow>) => {
    setRows(rows.map((row, i) => (i === index ? { ...row, ...change } : row)));
  };
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-fg-muted">{label}</span>
        <button type="button" onClick={() => setRows([...rows, { key: '', value: '' }])}
          className="btn-ghost text-xs flex items-center gap-1">
          <PlusIcon className="h-3.5 w-3.5" /> Add
        </button>
      </div>
      {rows.map((row, index) => (
        <div key={index} className="flex gap-2 mb-2">
          <input type="text" value={row.key} list={suggestionsId} placeholder="Tool name"
            aria-label={`${label} key ${index + 1}`}
            onChange={(e) => updateRow(index, { key: e.target.value })} className="input text-xs font-mono" />
          <input type="text" value={row.value} placeholder={valuePlaceholder}
            aria-label={`${label} value ${index + 1}`}
            onChange={(e) => updateRow(index, { value: e.target.value })} className="input text-xs" />
          <button type="button" className="btn-icon" aria-label={`Remove ${label} ${index + 1}`}
            onClick={() => setRows(rows.filter((_, i) => i !== index))}>
            <TrashIcon className="h-4 w-4" />
          </button>
        </div>
      ))}
    </div>
  );
};


/**
 * OpenApiTargetForm adds, edits, or rotates the credential of one REST target.
 *
 * The credential field is never pre-filled. On edit an empty credential keeps
 * the stored one. Each API error code is shown beside the field it concerns.
 */
const OpenApiTargetForm: React.FC<OpenApiTargetFormProps> = ({
  serverSlug,
  target,
  intent,
  onSaved,
  onCancel,
}) => {
  const isCreate = intent === 'create';
  const isRotate = intent === 'rotate';
  const initialScheme: OpenApiAuthScheme = target?.auth?.scheme || 'none';

  const [name, setName] = useState(target?.name || '');
  const [description, setDescription] = useState(target?.description || '');
  const [specUrl, setSpecUrl] = useState(target?.spec_url || '');
  const [baseUrl, setBaseUrl] = useState(target?.base_url || '');
  const [scheme, setScheme] = useState<OpenApiAuthScheme>(initialScheme);
  const [headerName, setHeaderName] = useState(target?.auth?.header_name || '');
  const [queryParam, setQueryParam] = useState(target?.auth?.query_param || '');
  const [username, setUsername] = useState(target?.auth?.username || '');
  // The credential is write-only, so it always starts empty.
  const [credential, setCredential] = useState('');
  const [onlyThese, setOnlyThese] = useState(
    (target?.include_tags?.length || 0) > 0 || (target?.include_operations?.length || 0) > 0,
  );
  const [tagCounts, setTagCounts] = useState<Record<string, number | null>>(() => {
    const counts: Record<string, number | null> = {};
    for (const tag of target?.include_tags || []) counts[tag] = null;
    return counts;
  });
  const [selectedTags, setSelectedTags] = useState<string[]>(target?.include_tags || []);
  const [operationsText, setOperationsText] = useState(
    (target?.include_operations || []).join('\n'),
  );
  const [nameOverrides, setNameOverrides] = useState(_rowsFromMap(target?.tool_name_overrides));
  const [descOverrides, setDescOverrides] = useState(_rowsFromMap(target?.description_overrides));
  const [keepOutputSchemas, setKeepOutputSchemas] = useState(!!target?.keep_output_schemas);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [saving, setSaving] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<Partial<Record<FieldKey, string>>>({});
  const [banner, setBanner] = useState<string | null>(null);
  const [invalidNames, setInvalidNames] = useState<string[]>([]);

  useEscapeKey(onCancel, !saving);

  const toggleTag = (tag: string) => {
    setSelectedTags((prev) => (prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]));
  };

  const authChanged = scheme !== initialScheme ||
    headerName !== (target?.auth?.header_name || '') ||
    queryParam !== (target?.auth?.query_param || '') ||
    username !== (target?.auth?.username || '');

  /** Decide the auth to send. `undefined` means leave the stored auth alone. */
  const resolveAuth = (): { auth?: OpenApiTargetAuthInput; error: string | null } => {
    const built = _buildAuth(scheme, headerName, queryParam, username, credential);
    if (isCreate || isRotate) {
      return built;
    }
    if (!credential && !authChanged) {
      return { auth: undefined, error: null };
    }
    if (!credential && scheme !== 'none') {
      return { auth: undefined, error: 'Enter the credential again to change the auth settings.' };
    }
    return built;
  };

  const filters = () => ({
    include_tags: onlyThese ? selectedTags : [],
    include_operations: onlyThese ? _linesToList(operationsText) : [],
  });

  const buildCreateBody = (auth: OpenApiTargetAuthInput): CreateOpenApiTargetRequest => ({
    name: name.trim(),
    description: description.trim(),
    spec_url: specUrl.trim(),
    base_url: baseUrl.trim(),
    auth,
    tool_name_overrides: _mapFromRows(nameOverrides),
    description_overrides: _mapFromRows(descOverrides),
    ...filters(),
    keep_output_schemas: keepOutputSchemas,
  });

  const buildUpdateBody = (auth: OpenApiTargetAuthInput | undefined): UpdateOpenApiTargetRequest => {
    if (isRotate) {
      return { auth };
    }
    const body: UpdateOpenApiTargetRequest = {
      description: description.trim(),
      tool_name_overrides: _mapFromRows(nameOverrides),
      description_overrides: _mapFromRows(descOverrides),
      ...filters(),
      keep_output_schemas: keepOutputSchemas,
    };
    if (baseUrl.trim() !== (target?.base_url || '')) body.base_url = baseUrl.trim();
    if (auth) body.auth = auth;
    return body;
  };

  const validateCreate = (): Partial<Record<FieldKey, string>> => {
    const errors: Partial<Record<FieldKey, string>> = {};
    if (!TARGET_NAME_PATTERN.test(name.trim())) {
      errors.name = 'Use 2 to 24 lowercase letters and digits, starting with a letter. It becomes the tool prefix.';
    }
    if (!specUrl.trim()) errors.spec_url = 'Enter the URL of the OpenAPI spec.';
    if (!baseUrl.trim()) errors.base_url = 'Enter the base URL the API is called at.';
    return errors;
  };

  const showApiError = (error: OpenApiApiError) => {
    const field = _fieldForError(error);
    if (!field) {
      setBanner(_bannerText(error));
      return;
    }
    setFieldErrors({ [field]: _errorSentence(error) });
    if (error.code === 'too_many_tools' && error.tags) {
      setTagCounts(error.tags);
      setOnlyThese(true);
    }
    if (error.code === 'tool_name_invalid') {
      setInvalidNames(error.names || []);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBanner(null);
    setInvalidNames([]);
    const errors = isCreate ? validateCreate() : {};
    const resolved = resolveAuth();
    if (resolved.error) errors.auth = resolved.error;
    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) return;

    setSaving(true);
    try {
      const result = isCreate
        ? await addOpenApiTarget(serverSlug, buildCreateBody(resolved.auth as OpenApiTargetAuthInput))
        : await updateOpenApiTarget(serverSlug, target?.name || '', buildUpdateBody(resolved.auth));
      onSaved(result);
    } catch (err: unknown) {
      showApiError(parseOpenApiError(err, 'Failed to save the API'));
    } finally {
      setSaving(false);
    }
  };

  const title = isCreate ? 'Add API' : isRotate ? `Rotate credential: ${target?.name}` : `Edit API: ${target?.name}`;
  const suggestions = _toolKeySuggestions(target);

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-[60]">
      <form onSubmit={handleSubmit} role="dialog" aria-modal="true" aria-label={title}
        className="bg-surface-overlay border border-line rounded-xl shadow-xl max-w-3xl w-full mx-4 max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-4 border-b border-line flex-shrink-0">
          <h2 className="text-lg font-semibold text-fg truncate">{title}</h2>
          <button type="button" onClick={onCancel} className="btn-icon" aria-label="Close">
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {banner && (
            <div role="alert" className="p-3 bg-danger-50 dark:bg-danger-900/20 border border-danger-200 dark:border-danger-800 rounded-lg">
              <p className="text-sm text-danger-700 dark:text-danger-300">{banner}</p>
            </div>
          )}

          {!isRotate && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="openapi-target-name" className="block text-sm font-medium text-fg-muted mb-1">
                    Name <span className="text-danger-500">*</span>
                  </label>
                  <input id="openapi-target-name" type="text" value={name} disabled={!isCreate}
                    onChange={(e) => setName(e.target.value)} placeholder="petstore" maxLength={24}
                    className="input font-mono" autoComplete="off" />
                  <_FieldError field="name" message={fieldErrors.name} />
                </div>
                <div>
                  <label htmlFor="openapi-target-description" className="block text-sm font-medium text-fg-muted mb-1">
                    Description
                  </label>
                  <input id="openapi-target-description" type="text" value={description}
                    onChange={(e) => setDescription(e.target.value)} maxLength={1000} className="input" />
                </div>
              </div>

              <div>
                <label htmlFor="openapi-target-spec-url" className="block text-sm font-medium text-fg-muted mb-1">
                  Spec URL <span className="text-danger-500">*</span>
                </label>
                <input id="openapi-target-spec-url" type="url" value={specUrl} disabled={!isCreate}
                  onChange={(e) => setSpecUrl(e.target.value)} className="input font-mono text-sm"
                  placeholder="https://petstore3.swagger.io/api/v3/openapi.json" autoComplete="off" />
                {!isCreate && (
                  <p className="mt-1 text-xs text-fg-subtle">
                    To load a new version of the spec, use Refresh spec in the server details.
                  </p>
                )}
                <_FieldError field="spec_url" message={fieldErrors.spec_url} />
              </div>

              <div>
                <label htmlFor="openapi-target-base-url" className="block text-sm font-medium text-fg-muted mb-1">
                  Base URL <span className="text-danger-500">*</span>
                </label>
                <input id="openapi-target-base-url" type="url" value={baseUrl}
                  onChange={(e) => setBaseUrl(e.target.value)} className="input font-mono text-sm"
                  placeholder="https://petstore3.swagger.io/api/v3" autoComplete="off" />
                <_FieldError field="base_url" message={fieldErrors.base_url} />
              </div>

              <_OperationsSection onlyThese={onlyThese} setOnlyThese={setOnlyThese}
                tagCounts={tagCounts} selectedTags={selectedTags} toggleTag={toggleTag}
                operationsText={operationsText} setOperationsText={setOperationsText}
                error={fieldErrors.operations} />
            </>
          )}

          <_AuthFields scheme={scheme} setScheme={setScheme} headerName={headerName}
            setHeaderName={setHeaderName} queryParam={queryParam} setQueryParam={setQueryParam}
            username={username} setUsername={setUsername} credential={credential}
            setCredential={setCredential} credentialSet={!!target?.auth?.credential_set}
            error={fieldErrors.auth} />

          {!isRotate && (
            <div className="border-t border-line pt-4 space-y-4">
              {fieldErrors.overrides && (
                <div data-testid="openapi-tool-name-error">
                  <_FieldError field="overrides" message={fieldErrors.overrides} />
                  {invalidNames.length > 0 && (
                    <ul className="mt-1 text-xs font-mono text-danger-600 dark:text-danger-400 list-disc pl-5">
                      {invalidNames.map((n) => <li key={n}>{n}</li>)}
                    </ul>
                  )}
                  <button type="button" onClick={() => setShowAdvanced(true)}
                    className="mt-1 text-xs text-primary-600 dark:text-primary-400 underline">
                    Open tool name overrides
                  </button>
                </div>
              )}
              <button type="button" onClick={() => setShowAdvanced(!showAdvanced)}
                aria-expanded={showAdvanced} className="text-sm font-medium text-fg-muted hover:text-fg">
                {showAdvanced ? 'Hide advanced settings' : 'Show advanced settings'}
              </button>
              {showAdvanced && (
                <div className="space-y-4" data-testid="openapi-advanced-section">
                  <datalist id="openapi-tool-keys">
                    {suggestions.map((s) => <option key={s} value={s} />)}
                  </datalist>
                  <_OverridesEditor label="Tool name overrides" rows={nameOverrides}
                    setRows={setNameOverrides} suggestionsId="openapi-tool-keys" valuePlaceholder="New name" />
                  <_OverridesEditor label="Description overrides" rows={descOverrides}
                    setRows={setDescOverrides} suggestionsId="openapi-tool-keys" valuePlaceholder="New description" />
                  <label className="flex items-start gap-2 text-sm text-fg">
                    <input type="checkbox" checked={keepOutputSchemas} className="mt-0.5"
                      onChange={(e) => setKeepOutputSchemas(e.target.checked)} />
                    <span>
                      Keep output schemas
                      <span className="block text-xs text-fg-subtle">
                        Off by default. Many real APIs return responses that break their own
                        spec, and clients reject a tool result that does not match its schema.
                      </span>
                    </span>
                  </label>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-line flex-shrink-0">
          <button type="button" onClick={onCancel} disabled={saving} className="btn-secondary">
            Cancel
          </button>
          <button type="submit" disabled={saving} className="btn-primary">
            {saving ? 'Saving...' : isCreate ? 'Add API' : isRotate ? 'Rotate credential' : 'Save Changes'}
          </button>
        </div>
      </form>
    </div>
  );
};


export default OpenApiTargetForm;
