import React, { useState, useEffect, useCallback } from 'react';
import { PlusIcon, ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';
import {
  OpenApiServer,
  OpenApiTarget,
  OpenApiTargetWriteResponse,
} from '../types/openApiServer';
import {
  getOpenApiServer,
  updateOpenApiTarget,
  deleteOpenApiTarget,
  parseOpenApiError,
} from '../hooks/useOpenApiServers';
import { EntityModal, CopyButton } from './modals';
import { SafeLink } from './SafeLink';
import ConfirmModal from './ConfirmModal';
import OpenApiTargetForm, { OpenApiTargetFormIntent } from './OpenApiTargetForm';
import { OpenApiStatusBadge, openApiServerStatus } from './OpenApiServerCard';


/** Poll every 5 seconds while a target is pending. */
export const OPENAPI_POLL_INTERVAL_MS: number = 5000;
/** Stop polling after 2 minutes. */
export const OPENAPI_POLL_MAX_MS: number = 120000;
const MAX_POLLS: number = OPENAPI_POLL_MAX_MS / OPENAPI_POLL_INTERVAL_MS;
/** How many tool names an expanded row shows before "Show all". */
const TOOL_PREVIEW_COUNT: number = 50;


type ToastFn = (message: string, type: 'success' | 'error' | 'info') => void;


interface OpenApiServerDetailsModalProps {
  slug: string;
  isOpen: boolean;
  onClose: () => void;
  onShowToast?: ToastFn;
  /** Called after any write, so the caller can refresh its list. */
  onChanged?: () => void;
}


/** The tool names, filter and error message of one expanded target row. */
const _TargetToolList: React.FC<{ target: OpenApiTarget }> = ({ target }) => {
  const [filter, setFilter] = useState('');
  const [showAll, setShowAll] = useState(false);
  const names = target.tool_names || [];
  const matching = filter
    ? names.filter((n) => n.toLowerCase().includes(filter.toLowerCase()))
    : names;
  const shown = showAll ? matching : matching.slice(0, TOOL_PREVIEW_COUNT);

  return (
    <div className="px-4 py-3 bg-surface-muted space-y-2" data-testid={`openapi-target-tools-${target.name}`}>
      {target.status === 'error' && target.error && (
        <p className="text-xs text-danger-600 dark:text-danger-400 whitespace-pre-wrap">{target.error}</p>
      )}
      {target.spec_url && (
        <p className="text-xs text-fg-muted">
          Spec:{' '}
          <SafeLink href={target.spec_url} target="_blank" rel="noopener noreferrer nofollow"
            className="font-mono text-primary-600 dark:text-primary-400 break-all">
            {target.spec_url}
          </SafeLink>
        </p>
      )}
      {target.description && <p className="text-xs text-fg-muted">{target.description}</p>}
      <input type="text" value={filter} onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter tools" aria-label={`Filter tools of ${target.name}`}
        className="input text-xs py-1" />
      <ul className="text-xs font-mono text-fg grid grid-cols-1 md:grid-cols-2 gap-x-4">
        {shown.map((n) => <li key={n} className="truncate">{n}</li>)}
      </ul>
      {matching.length > TOOL_PREVIEW_COUNT && (
        <button type="button" onClick={() => setShowAll(!showAll)}
          className="text-xs text-primary-600 dark:text-primary-400 underline">
          {showAll ? 'Show fewer' : `Show all ${matching.length} tools`}
        </button>
      )}
      {names.length === 0 && <p className="text-xs text-fg-subtle">No tools.</p>}
    </div>
  );
};


interface TargetRowProps {
  target: OpenApiTarget;
  expanded: boolean;
  onToggle: () => void;
  onEdit: () => void;
  onRotate: () => void;
  onRefresh: () => void;
  onDelete: () => void;
  busy: boolean;
}


const _TargetRow: React.FC<TargetRowProps> = (props) => {
  const { target } = props;
  const specLabel = [target.spec_title, target.spec_version].filter(Boolean).join(' ');
  const actionClass = 'px-2 py-1 text-xs font-medium rounded hover:bg-surface-hover text-fg-muted';
  return (
    <>
      <tr className="table-row">
        <td className="px-3 py-2 whitespace-nowrap">
          <button type="button" onClick={props.onToggle} aria-expanded={props.expanded}
            className="flex items-center gap-1 text-sm font-medium text-fg font-mono">
            {props.expanded ? <ChevronDownIcon className="h-3 w-3" /> : <ChevronRightIcon className="h-3 w-3" />}
            {target.name}
          </button>
        </td>
        <td className="px-3 py-2 text-xs text-fg-muted">{specLabel || '-'}</td>
        <td className="px-3 py-2 text-xs font-mono max-w-[200px] truncate">
          {target.base_url ? (
            <SafeLink href={target.base_url} target="_blank" rel="noopener noreferrer nofollow"
              className="text-primary-600 dark:text-primary-400">{target.base_url}</SafeLink>
          ) : '-'}
        </td>
        <td className="px-3 py-2 text-xs text-fg-muted">{target.auth?.scheme || 'none'}</td>
        <td className="px-3 py-2 text-xs text-fg-muted" data-testid={`openapi-credential-set-${target.name}`}>
          {target.auth?.credential_set ? 'yes' : 'no'}
        </td>
        <td className="px-3 py-2 text-xs text-fg-muted">{target.tool_names?.length || 0}</td>
        <td className="px-3 py-2"><OpenApiStatusBadge status={target.status || 'ok'} /></td>
        <td className="px-3 py-2 whitespace-nowrap text-right">
          <button type="button" className={actionClass} onClick={props.onEdit} disabled={props.busy}>Edit</button>
          {target.auth?.scheme && target.auth.scheme !== 'none' && (
            <button type="button" className={actionClass} onClick={props.onRotate} disabled={props.busy}>
              Rotate credential
            </button>
          )}
          <button type="button" className={actionClass} onClick={props.onRefresh} disabled={props.busy}>
            Refresh spec
          </button>
          <button type="button" onClick={props.onDelete} disabled={props.busy}
            className="px-2 py-1 text-xs font-medium rounded text-danger-700 dark:text-danger-300 hover:bg-danger-50 dark:hover:bg-danger-900/20">
            Delete
          </button>
        </td>
      </tr>
      {props.expanded && (
        <tr>
          <td colSpan={8} className="p-0"><_TargetToolList target={target} /></td>
        </tr>
      )}
    </>
  );
};


/** The toast text after a target write: sync status plus any warnings. */
function _savedMessage(
  verb: string,
  result: OpenApiTargetWriteResponse,
): string {
  const pending = result.sync_status === 'pending'
    ? ' The container has not confirmed it yet; the registry will finish the sync.'
    : '';
  const warnings = result.warnings && result.warnings.length > 0
    ? ` Warnings: ${result.warnings.join('; ')}`
    : '';
  return `API "${result.name}" ${verb}.${pending}${warnings}`;
}


/**
 * OpenApiServerDetailsModal shows one OpenAPI MCP server and a table of its
 * targets, with Add, Edit, Rotate credential, Refresh spec and Delete. While
 * anything is pending it polls the server every 5 seconds for up to 2 minutes.
 */
const OpenApiServerDetailsModal: React.FC<OpenApiServerDetailsModalProps> = ({
  slug,
  isOpen,
  onClose,
  onShowToast,
  onChanged,
}) => {
  const [server, setServer] = useState<OpenApiServer | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [pollCount, setPollCount] = useState(0);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [form, setForm] = useState<{ intent: OpenApiTargetFormIntent; target: OpenApiTarget | null } | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<OpenApiTarget | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async (showSpinner: boolean) => {
    if (showSpinner) setLoading(true);
    try {
      setServer(await getOpenApiServer(slug));
      setLoadError(null);
    } catch (err: unknown) {
      setLoadError(parseOpenApiError(err, 'Failed to load the OpenAPI MCP server').message);
    } finally {
      if (showSpinner) setLoading(false);
    }
  }, [slug]);

  useEffect(() => {
    if (!isOpen) {
      setServer(null);
      return;
    }
    setPollCount(0);
    load(true);
  }, [isOpen, load]);

  const pending = !!server && openApiServerStatus(server) === 'pending';
  const pollTimedOut = pending && pollCount >= MAX_POLLS;

  // Poll while pending. Each finished poll bumps pollCount, which schedules the next.
  useEffect(() => {
    if (!isOpen || !pending || pollCount >= MAX_POLLS) {
      return;
    }
    const timer = setTimeout(async () => {
      await load(false);
      setPollCount((c) => c + 1);
    }, OPENAPI_POLL_INTERVAL_MS);
    return () => clearTimeout(timer);
  }, [isOpen, pending, pollCount, load]);

  const afterWrite = async () => {
    setPollCount(0);
    await load(false);
    onChanged?.();
  };

  const handleSaved = async (result: OpenApiTargetWriteResponse) => {
    const verb = form?.intent === 'create' ? 'added' : 'updated';
    setForm(null);
    onShowToast?.(_savedMessage(verb, result), result.sync_status === 'pending' ? 'info' : 'success');
    await afterWrite();
  };

  const handleRefreshSpec = async (target: OpenApiTarget) => {
    setBusy(true);
    try {
      const result = await updateOpenApiTarget(slug, target.name, { refresh_spec: true });
      onShowToast?.(_savedMessage('refreshed', { ...result, name: result.name || target.name }), 'success');
      await afterWrite();
    } catch (err: unknown) {
      onShowToast?.(parseOpenApiError(err, 'Failed to refresh the spec').message, 'error');
    } finally {
      setBusy(false);
    }
  };

  const handleDeleteTarget = async () => {
    if (!deleteTarget) return;
    setBusy(true);
    try {
      await deleteOpenApiTarget(slug, deleteTarget.name);
      onShowToast?.(`API "${deleteTarget.name}" deleted`, 'success');
      setDeleteTarget(null);
      await afterWrite();
    } catch (err: unknown) {
      onShowToast?.(parseOpenApiError(err, 'Failed to delete the API').message, 'error');
    } finally {
      setBusy(false);
    }
  };

  const title = (
    <div>
      <div className="flex items-center gap-2">
        <h3 className="text-lg font-semibold text-fg">{server?.name || slug}</h3>
        {server && <OpenApiStatusBadge status={openApiServerStatus(server)} />}
      </div>
      <p className="text-sm text-fg-subtle font-mono">{server?.path || `/${slug}`}</p>
    </div>
  );

  return (
    <>
      <EntityModal isOpen={isOpen} onClose={form || deleteTarget ? () => undefined : onClose}
        title={title} loading={loading} error={server ? null : loadError} maxWidth="4xl" layout="flush">
        {server && (
          <div className="space-y-4">
            {pollTimedOut && (
              <div role="status" className="p-3 bg-warning-50 dark:bg-warning-900/20 border border-warning-200 dark:border-warning-800 rounded-lg text-sm text-warning-800 dark:text-warning-300">
                Still syncing. Check the openapi-mcp container.
              </div>
            )}
            <p className="text-sm text-fg-muted whitespace-pre-wrap">
              {server.description || 'No description available.'}
            </p>
            {server.mcp_url && (
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-xs font-semibold text-fg-subtle uppercase">MCP URL</span>
                <SafeLink href={server.mcp_url} target="_blank" rel="noopener noreferrer"
                  className="text-xs font-mono text-primary-600 dark:text-primary-400 truncate">
                  {server.mcp_url}
                </SafeLink>
                <CopyButton getText={() => server.mcp_url || ''} label="Copy" variant="subtle" />
              </div>
            )}
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold text-fg-subtle uppercase tracking-wide">
                APIs ({server.targets.length})
              </p>
              <button type="button" onClick={() => setForm({ intent: 'create', target: null })}
                className="btn-primary flex items-center gap-1 text-sm">
                <PlusIcon className="h-4 w-4" /> Add API
              </button>
            </div>
            {server.targets.length === 0 ? (
              <p className="text-sm text-fg-subtle">No APIs yet. Add one to give this server tools.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-line">
                  <thead className="bg-surface-muted">
                    <tr>
                      {['Name', 'Spec', 'Base URL', 'Auth', 'Credential set', 'Tools', 'Status', ''].map((h) => (
                        <th key={h} className="px-3 py-2 text-left text-xs font-medium text-fg-subtle uppercase">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-surface divide-y divide-line">
                    {server.targets.map((t) => (
                      <_TargetRow key={t.name} target={t} busy={busy}
                        expanded={!!expanded[t.name]}
                        onToggle={() => setExpanded((p) => ({ ...p, [t.name]: !p[t.name] }))}
                        onEdit={() => setForm({ intent: 'edit', target: t })}
                        onRotate={() => setForm({ intent: 'rotate', target: t })}
                        onRefresh={() => handleRefreshSpec(t)}
                        onDelete={() => setDeleteTarget(t)} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </EntityModal>

      {form && (
        <OpenApiTargetForm serverSlug={slug} target={form.target} intent={form.intent}
          onSaved={handleSaved} onCancel={() => setForm(null)} />
      )}

      <ConfirmModal
        isOpen={!!deleteTarget}
        onClose={() => setDeleteTarget(null)}
        onConfirm={handleDeleteTarget}
        title="Delete API"
        message={deleteTarget
          ? `Delete the API "${deleteTarget.name}"? Its ${deleteTarget.tool_names?.length || 0} tools are removed from ${server?.path || `/${slug}`}.`
          : ''}
        confirmLabel="Delete"
        loadingLabel="Deleting..."
        isDestructive
        isLoading={busy}
      />
    </>
  );
};


export default OpenApiServerDetailsModal;
