import React, { useState, useEffect, useCallback } from 'react';
import { MyCredentialStatus } from '../types/openApiServer';
import { listMyCredentials } from '../hooks/useOpenApiServers';
import { EntityModal } from './modals';
import OpenApiMyKeyControl, { keyErrorText } from './OpenApiMyKeyControl';


type ToastFn = (message: string, type: 'success' | 'error' | 'info') => void;


interface OpenApiMyKeysModalProps {
  slug: string;
  serverName: string;
  isOpen: boolean;
  onClose: () => void;
  onShowToast?: ToastFn;
}


/**
 * OpenApiMyKeysModal lets any user who can see an OpenAPI MCP server manage
 * their own keys for its per-user targets. It lists the targets from
 * GET .../my-credentials, with add, replace and remove for each.
 */
const OpenApiMyKeysModal: React.FC<OpenApiMyKeysModalProps> = ({
  slug,
  serverName,
  isOpen,
  onClose,
  onShowToast,
}) => {
  const [targets, setTargets] = useState<MyCredentialStatus[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (showSpinner: boolean) => {
    if (showSpinner) setLoading(true);
    try {
      setTargets(await listMyCredentials(slug));
      setError(null);
    } catch (err: unknown) {
      setError(keyErrorText(err, 'Failed to load your API keys'));
    } finally {
      if (showSpinner) setLoading(false);
    }
  }, [slug]);

  useEffect(() => {
    if (isOpen) {
      load(true);
    }
  }, [isOpen, load]);

  return (
    <EntityModal isOpen={isOpen} onClose={onClose} title={`Your API keys: ${serverName}`}
      loading={loading} error={error} maxWidth="2xl">
      <div className="space-y-4">
        <p className="text-sm text-fg-muted">
          These APIs call the provider with your own key. Get each key from the API provider
          and add it here. The registry never fetches keys.
        </p>
        {targets.length === 0 ? (
          <p className="text-sm text-fg-subtle">This server has no APIs that need your own key.</p>
        ) : (
          <ul className="divide-y divide-line">
            {targets.map((t) => (
              <li key={t.target} className="py-3">
                <OpenApiMyKeyControl slug={slug} targetName={t.target} status={t}
                  onChanged={() => load(false)} onShowToast={onShowToast} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </EntityModal>
  );
};


export default OpenApiMyKeysModal;
