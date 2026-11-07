import React from 'react';
import type {
  LocalRuntimeFormData,
  LocalRuntimeFormType,
} from '../utils/localRuntime';
import { CHECKBOX, FIELD, FIELD_ERROR, LABEL } from './formFields';


interface LocalRuntimeFormPanelProps {
  /** Current form state. */
  runtime: LocalRuntimeFormData;
  /** Called with the next state on any field change. */
  onChange: (next: LocalRuntimeFormData) => void;
  /** Optional inline error messages keyed by field name. */
  errors?: {
    package?: string;
    image_digest?: string;
    env?: string;
  };
  /**
   * Tailwind class string applied to text/select inputs. Lets the embedding
   * page reuse its own input style; defaults to a sensible standalone style.
   */
  inputClass?: string;
  /**
   * Tailwind class string applied to field labels. Defaults match the
   * registration form.
   */
  labelClass?: string;
  /** Tailwind class string applied to inline error text. */
  errorClass?: string;
}


const DEFAULT_INPUT_CLASS = FIELD;
const DEFAULT_LABEL_CLASS = LABEL;
const DEFAULT_ERROR_CLASS = FIELD_ERROR;


/**
 * Shared local-runtime form panel used by both the registration page and the
 * Dashboard edit modal. Renders the runtime-type select, package/version/
 * digest inputs, args list editor, and env-rows editor with required toggle.
 *
 * State management is hoisted to the parent — this component is purely
 * controlled. The parent decides where the panel lives in its broader form
 * (deployment toggle, validation timing, submit handling).
 */
const LocalRuntimeFormPanel: React.FC<LocalRuntimeFormPanelProps> = ({
  runtime,
  onChange,
  errors = {},
  inputClass = DEFAULT_INPUT_CLASS,
  labelClass = DEFAULT_LABEL_CLASS,
  errorClass = DEFAULT_ERROR_CLASS,
}) => {
  const update = (patch: Partial<LocalRuntimeFormData>) => onChange({ ...runtime, ...patch });

  const updateEnvRow = (idx: number, patch: Partial<LocalRuntimeFormData['envRows'][number]>) => {
    const rows = [...runtime.envRows];
    rows[idx] = { ...rows[idx], ...patch };
    onChange({ ...runtime, envRows: rows });
  };

  const removeEnvRow = (idx: number) => {
    onChange({ ...runtime, envRows: runtime.envRows.filter((_, i) => i !== idx) });
  };

  const addEnvRow = () => {
    onChange({
      ...runtime,
      envRows: [...runtime.envRows, { key: '', value: '', required: false }],
    });
  };

  const updateArg = (idx: number, value: string) => {
    const next = [...runtime.argList];
    next[idx] = value;
    onChange({ ...runtime, argList: next });
  };

  const removeArg = (idx: number) => {
    onChange({ ...runtime, argList: runtime.argList.filter((_, i) => i !== idx) });
  };

  const addArg = () => {
    onChange({ ...runtime, argList: [...runtime.argList, ''] });
  };

  const packageLabel =
    runtime.type === 'docker' ? 'Image Reference *'
    : runtime.type === 'command' ? 'Command Path *'
    : 'Package Name *';

  const packagePlaceholder =
    runtime.type === 'docker' ? 'acme/weather-mcp:1.2.0'
    : runtime.type === 'command' ? '/usr/local/bin/my-mcp'
    : '@acme/weather-mcp';

  return (
    <div className="border border-purple-200 dark:border-purple-800 rounded-lg p-4 bg-purple-50/40 dark:bg-purple-900/10">
      <h4 className="font-medium text-gray-900 dark:text-white mb-3">Launch Recipe</h4>

      {runtime.type === 'command' && (
        <div className="mb-3 p-3 rounded-md bg-warning-50 dark:bg-warning-900/20 border border-warning-200 dark:border-warning-800 text-sm text-warning-900 dark:text-warning-100">
          Warning: command type executes an arbitrary binary on every developer&apos;s machine.
          Only register commands you trust.
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className={labelClass}>Runtime Type</label>
          <select
            className={inputClass}
            value={runtime.type}
            onChange={(e) => update({ type: e.target.value as LocalRuntimeFormType })}
          >
            <option value="npx">npx</option>
            <option value="docker">docker</option>
            <option value="uvx">uvx</option>
            <option value="command">command (admin-only)</option>
          </select>
        </div>

        <div>
          <label className={labelClass}>{packageLabel}</label>
          <input
            type="text"
            className={`${inputClass} ${errors.package ? 'border-danger-500' : ''}`}
            value={runtime.package}
            onChange={(e) => update({ package: e.target.value })}
            placeholder={packagePlaceholder}
          />
          {errors.package && <p className={errorClass}>{errors.package}</p>}
        </div>

        {(runtime.type === 'npx' || runtime.type === 'uvx') && (
          <div>
            <label className={labelClass}>Version Pin (recommended)</label>
            <input
              type="text"
              className={inputClass}
              value={runtime.version}
              onChange={(e) => update({ version: e.target.value })}
              placeholder="1.2.0"
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Unpinned packages are tagged &lsquo;unpinned-version&rsquo; for visibility.
            </p>
          </div>
        )}

        {runtime.type === 'docker' && (
          <div>
            <label className={labelClass}>Image Digest (recommended)</label>
            <input
              type="text"
              className={`${inputClass} ${errors.image_digest ? 'border-danger-500' : ''}`}
              value={runtime.image_digest}
              onChange={(e) => update({ image_digest: e.target.value })}
              placeholder="sha256:..."
            />
            {errors.image_digest && <p className={errorClass}>{errors.image_digest}</p>}
          </div>
        )}

        <div className="md:col-span-2">
          <label className={labelClass}>Args</label>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
            Argv-style list — one entry per argument. No shell interpolation.
            Args may contain commas, equals signs, etc. (e.g. <code>--label=k=v,w=x</code>).
          </p>
          {runtime.argList.map((arg, idx) => (
            <div key={idx} className="flex gap-2 mb-2 items-center">
              <input
                type="text"
                className={`${inputClass} flex-1`}
                value={arg}
                onChange={(e) => updateArg(idx, e.target.value)}
                placeholder={`arg ${idx + 1}`}
              />
              <button
                type="button"
                onClick={() => removeArg(idx)}
                className="text-danger-500 hover:text-danger-700 dark:text-danger-400 dark:hover:text-danger-300 px-2"
              >
                ×
              </button>
            </div>
          ))}
          <button
            type="button"
            className="mt-1 px-3 py-1 text-sm bg-purple-100 dark:bg-purple-800 text-purple-700 dark:text-purple-200 rounded hover:bg-purple-200 dark:hover:bg-purple-700"
            onClick={addArg}
          >
            + Add arg
          </button>
        </div>

        <div className="md:col-span-2">
          <label className={labelClass}>Environment Variables</label>
          <div className="mb-2 p-3 rounded-md bg-warning-50 dark:bg-warning-900/20 border border-warning-200 dark:border-warning-800 text-xs text-warning-900 dark:text-warning-100">
            <strong>Visibility note:</strong> values you put here are visible
            to every authenticated registry user (they need them to render
            the Connect modal), and may appear in process listings
            (<code>ps aux</code>, <code>/proc/&lt;pid&gt;/environ</code>) on
            developer machines once launched. Use literal values only for
            non-secret defaults (e.g. <code>LOG_LEVEL=info</code>). For
            secrets, use <code>${'$'}{'{VAR}'}</code> placeholders or mark
            the row &lsquo;required from user&rsquo; so the user supplies
            it at connect time.
          </div>
          {runtime.envRows.length === 0 && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
              None. Add rows for env vars. Mark a row &lsquo;required from user&rsquo;
              for secrets the user provides at connect time.
            </p>
          )}
          {runtime.envRows.map((row, idx) => (
            <div key={idx} className="flex gap-2 mb-2 items-center">
              <input
                type="text"
                className={`${inputClass} flex-1`}
                placeholder="KEY"
                value={row.key}
                onChange={(e) => updateEnvRow(idx, { key: e.target.value })}
              />
              <input
                type="text"
                className={`${inputClass} flex-1 ${row.required ? 'opacity-50' : ''}`}
                placeholder={row.required ? '(provided by user at connect)' : 'value or ${VAR}'}
                disabled={row.required}
                value={row.value}
                onChange={(e) => updateEnvRow(idx, { value: e.target.value })}
              />
              <label className="flex items-center gap-1 text-xs text-gray-700 dark:text-gray-300 whitespace-nowrap">
                <input
                  type="checkbox"
                  className={CHECKBOX}
                  checked={row.required}
                  onChange={(e) => updateEnvRow(idx, {
                    required: e.target.checked,
                    value: e.target.checked ? '' : row.value,
                  })}
                />
                required from user
              </label>
              <button
                type="button"
                onClick={() => removeEnvRow(idx)}
                className="text-danger-500 hover:text-danger-700 dark:text-danger-400 dark:hover:text-danger-300 px-2"
              >
                ×
              </button>
            </div>
          ))}
          {errors.env && <p className={errorClass}>{errors.env}</p>}
          <button
            type="button"
            className="mt-2 px-3 py-1 text-sm bg-purple-100 dark:bg-purple-800 text-purple-700 dark:text-purple-200 rounded hover:bg-purple-200 dark:hover:bg-purple-700"
            onClick={addEnvRow}
          >
            + Add env var
          </button>
        </div>
      </div>
    </div>
  );
};


export default LocalRuntimeFormPanel;
