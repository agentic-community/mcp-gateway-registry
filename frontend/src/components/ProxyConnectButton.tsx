import React, { useEffect, useRef, useState } from 'react';
import { LinkIcon } from '@heroicons/react/24/outline';
import { CopyButton } from './modals';
import { gatewayClientUrl } from '../utils/gatewayClientUrl';

// Shown when the resource is not served through the gateway, so there is no
// client URL to hand out. Deliberately short: on the icon-only card control it
// renders as a tooltip, where there is no room for prose.
const DISABLED_HINT =
  'Not routed through the gateway. Enable gateway routing on this resource to connect.';
// Visible companion to DISABLED_HINT for the inline (labelled) variant, where a
// few words fit next to the label without crowding the row.
const DISABLED_SUFFIX = 'enable gateway routing';

interface ProxyConnectButtonProps {
  /** Server-derived client path, e.g. "/gateway/skill/pdf". Absent = not proxied. */
  clientUrl?: string | null;
  /** Backend origin the gateway forwards to (shown for context only). */
  targetUrl?: string | null;
  /** Optional operator-authored usage notes; rendered only when present. */
  connectNotes?: string | null;
  /**
   * Render as an inline icon+text control (for modal action rows) instead of the
   * icon-only card button. The popover then aligns left, matching the row.
   */
  label?: string;
}

/**
 * "Connect" affordance for a registry entity. Opens a small popover with the full
 * client-facing URL a caller uses to reach the resource through the gateway, a
 * one-line auth note, and a copy button.
 *
 * The URL is the same-origin client path (window.location.origin + clientUrl).
 * It is shown as selectable text plus copy, never a live link, because a caller
 * must append their own API sub-path and send auth headers, so the base URL is
 * not directly navigable.
 *
 * The control is always rendered so its absence never has to be interpreted: an
 * entity that is not proxied has no client URL, so the trigger is disabled and
 * explains what would make it work. Presence of clientUrl is the single source of
 * truth for "is this reachable through the gateway" -- the server derives it only
 * for a proxied entity, so no separate is_proxied prop can drift out of sync.
 */
const ProxyConnectButton: React.FC<ProxyConnectButtonProps> = ({
  clientUrl,
  targetUrl,
  connectNotes,
  label,
}) => {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDocMouseDown = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDocMouseDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const enabled = Boolean(clientUrl);
  const inline = Boolean(label);
  const fullUrl = gatewayClientUrl(clientUrl);

  const triggerClass = inline
    ? `flex items-center gap-1 text-sm transition-colors ${
        enabled
          ? 'text-cyan-700 dark:text-cyan-300 hover:underline'
          : 'text-gray-400 dark:text-gray-500 cursor-not-allowed'
      }`
    : `p-2 rounded-lg transition-colors ${
        enabled
          ? 'text-gray-400 hover:text-cyan-600 dark:hover:text-cyan-400'
          : 'text-gray-300 dark:text-gray-600 cursor-not-allowed'
      }`;

  return (
    <div
      className={inline ? 'relative inline-flex items-center gap-1' : 'relative'}
      ref={wrapperRef}
    >
      <button
        type="button"
        disabled={!enabled}
        aria-disabled={!enabled}
        onClick={() => setOpen((v) => !v)}
        className={triggerClass}
        title={enabled ? 'Show the client URL for this resource' : DISABLED_HINT}
        aria-label={
          enabled
            ? 'Connect: show the client URL for this proxied resource'
            : `Connect unavailable: ${DISABLED_HINT}`
        }
        aria-expanded={enabled ? open : undefined}
        aria-haspopup={enabled ? 'dialog' : undefined}
      >
        <LinkIcon className="h-4 w-4" />
        {inline && <span>{label}</span>}
      </button>
      {inline && !enabled && (
        <span className="text-xs text-gray-400 dark:text-gray-500" title={DISABLED_HINT}>
          &mdash; {DISABLED_SUFFIX}
        </span>
      )}
      {open && enabled && (
        <div
          role="dialog"
          aria-label="Connection details"
          className={`absolute ${
            inline ? 'left-0' : 'right-0'
          } z-20 mt-1 w-80 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-lg p-3 text-left`}
        >
          <p className="text-xs font-semibold text-gray-900 dark:text-white mb-1">Connect</p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Clients connect at:</p>
          <div className="flex items-start gap-2">
            <code className="flex-1 text-xs break-all bg-gray-50 dark:bg-gray-900 rounded px-2 py-1 text-gray-800 dark:text-gray-200">
              {fullUrl}
            </code>
            <CopyButton
              variant="subtle"
              label="Copy"
              copiedLabel="Copied"
              getText={() => fullUrl}
              title="Copy the client URL"
            />
          </div>
          {targetUrl && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 break-all">
              Forwards to {targetUrl}
            </p>
          )}
          {connectNotes && (
            <p
              className="text-xs text-gray-700 dark:text-gray-300 mt-2 whitespace-pre-wrap break-words max-h-40 overflow-y-auto rounded border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 px-2 py-1"
              title={connectNotes}
            >
              {connectNotes}
            </p>
          )}
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            Send the gateway token in the X-Authorization header; the backend credential goes in
            Authorization.
          </p>
        </div>
      )}
    </div>
  );
};

export default ProxyConnectButton;
