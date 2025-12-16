import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Checkbox } from '@/components/ui';
import { CopyButton } from '@/components/ui/CopyButton';
import {
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClipboardDocumentIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/outline';
import { useMintToken, decodeToken } from './hooks';
import { formatRelativeTime } from '@/lib/format';
import type { EnforceAIAgent, MintTokenResponse } from '@/api/types';

// ============================================================================
// Form Schema
// ============================================================================

const mintTokenSchema = z.object({
  scopes: z.string().min(1, 'At least one scope is required'),
  ttl_preset: z.string().optional(),
  ttl_custom: z.string().optional(),
  expires_at: z.string().optional(),
}).refine(
  (data) => {
    // Either TTL or expires_at, not both
    const hasTtl = data.ttl_preset || data.ttl_custom;
    const hasExpiry = data.expires_at;
    return !(hasTtl && hasExpiry);
  },
  {
    message: 'Cannot specify both TTL and expiration date',
    path: ['expires_at'],
  }
);

type MintTokenFormData = z.infer<typeof mintTokenSchema>;

// TTL presets in seconds
const TTL_PRESETS = [
  { label: '1 hour', value: '3600' },
  { label: '1 day', value: '86400' },
  { label: '7 days', value: '604800' },
  { label: '30 days', value: '2592000' },
  { label: 'Custom', value: 'custom' },
];

// ============================================================================
// Component
// ============================================================================

export interface MintTokenModalProps {
  open: boolean;
  agent: EnforceAIAgent;
  onClose: () => void;
  onSuccess: () => void;
}

export function MintTokenModal({
  open,
  agent,
  onClose,
  onSuccess,
}: MintTokenModalProps) {
  const { mintToken, isMinting, error } = useMintToken();
  const [mintedToken, setMintedToken] = useState<MintTokenResponse | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);
  const [showDecoded, setShowDecoded] = useState(false);

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
    reset,
  } = useForm<MintTokenFormData>({
    resolver: zodResolver(mintTokenSchema),
    defaultValues: {
      scopes: agent.scopes.join(', '),
      ttl_preset: '86400', // 1 day default
      ttl_custom: '',
      expires_at: '',
    },
  });

  const ttlPreset = watch('ttl_preset');

  const onSubmit = async (data: MintTokenFormData) => {
    try {
      const scopes = data.scopes.split(',').map((s) => s.trim()).filter(Boolean);

      let ttl_seconds: number | null = null;
      let expires_at: string | null = null;

      if (data.ttl_preset === 'custom' && data.ttl_custom) {
        ttl_seconds = parseInt(data.ttl_custom, 10);
      } else if (data.ttl_preset && data.ttl_preset !== 'custom') {
        ttl_seconds = parseInt(data.ttl_preset, 10);
      } else if (data.expires_at) {
        expires_at = new Date(data.expires_at).toISOString();
      }

      const result = await mintToken(agent.agent_id, {
        scopes,
        ttl_seconds,
        expires_at,
      });

      setMintedToken(result);
    } catch {
      // Error handled by hook
    }
  };

  const handleClose = () => {
    if (mintedToken && !acknowledged) {
      return;
    }
    reset();
    setMintedToken(null);
    setAcknowledged(false);
    setShowDecoded(false);
    onClose();
  };

  const handleDone = () => {
    reset();
    setMintedToken(null);
    setAcknowledged(false);
    setShowDecoded(false);
    onSuccess();
  };

  const decoded = mintedToken ? decodeToken(mintedToken.token) : null;

  // If token was minted, show the token display
  if (mintedToken) {
    return (
      <Modal
        open={open}
        onClose={handleClose}
        title="Gateway Token Minted"
        size="lg"
      >
        <div className="space-y-4">
          {/* Success message */}
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                <CheckCircleIcon className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                Gateway token minted successfully
              </p>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                For agent: {agent.alias || agent.agent_id.slice(0, 8)}
              </p>
              {decoded && (
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Expires: {formatRelativeTime(new Date(decoded.payload.exp * 1000).toISOString())}
                </p>
              )}
            </div>
          </div>

          {/* Critical warning */}
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-4">
            <div className="flex">
              <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-300">
                  Save this token now
                </h3>
                <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-400">
                  This is the only time you will see this token. Store it securely before closing.
                </p>
              </div>
            </div>
          </div>

          {/* Token display */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Gateway Token
            </label>
            <div className="relative">
              <code className="block w-full p-3 pr-24 bg-gray-100 dark:bg-gray-800 rounded-md text-xs font-mono text-gray-900 dark:text-gray-100 break-all border border-gray-200 dark:border-gray-700 max-h-32 overflow-y-auto">
                {mintedToken.token}
              </code>
              <div className="absolute right-2 top-2">
                <CopyButton text={mintedToken.token} />
              </div>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex space-x-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                navigator.clipboard.writeText(
                  `Authorization: Bearer ${mintedToken.token}`
                );
              }}
              leftIcon={<ClipboardDocumentIcon className="w-4 h-4" />}
            >
              Copy as Authorization Header
            </Button>
          </div>

          {/* Decode section */}
          <div className="border-t border-gray-200 dark:border-gray-700 pt-3">
            <button
              type="button"
              onClick={() => setShowDecoded(!showDecoded)}
              className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
            >
              {showDecoded ? (
                <ChevronUpIcon className="w-4 h-4 mr-1" />
              ) : (
                <ChevronDownIcon className="w-4 h-4 mr-1" />
              )}
              Decode Token (local only)
            </button>

            {showDecoded && decoded && (
              <div className="mt-3 space-y-3">
                <div>
                  <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                    Header
                  </label>
                  <pre className="p-2 bg-gray-50 dark:bg-gray-900 rounded text-xs font-mono overflow-x-auto">
                    {JSON.stringify(decoded.header, null, 2)}
                  </pre>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                    Payload
                  </label>
                  <pre className="p-2 bg-gray-50 dark:bg-gray-900 rounded text-xs font-mono overflow-x-auto">
                    {JSON.stringify(decoded.payload, null, 2)}
                  </pre>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 italic">
                  Note: This decoding is done locally in your browser. The token is not sent anywhere.
                </p>
              </div>
            )}
          </div>

          {/* Acknowledgment checkbox */}
          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <Checkbox
              label="I have securely saved this gateway token"
              checked={acknowledged}
              onChange={(e) => setAcknowledged(e.target.checked)}
            />
          </div>
        </div>

        <ModalFooter>
          <Button
            type="button"
            variant="primary"
            onClick={handleDone}
            disabled={!acknowledged}
          >
            Done
          </Button>
        </ModalFooter>
      </Modal>
    );
  }

  // Form to mint token
  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Mint Gateway Token"
      size="md"
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        {/* Agent info */}
        <div className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-md">
          <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
            <ShieldCheckIcon className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              Minting token for: {agent.alias || 'Unnamed Agent'}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 font-mono">
              {agent.agent_id}
            </p>
          </div>
        </div>

        {/* Scopes */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Scopes <span className="text-red-500">*</span>
          </label>
          <Input
            {...register('scopes')}
            placeholder="scope1, scope2"
            error={errors.scopes?.message}
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Comma-separated list. Must be subset of agent scopes.
          </p>
          {agent.scopes.length > 0 && (
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Agent scopes: {agent.scopes.join(', ')}
            </p>
          )}
        </div>

        {/* TTL Preset */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Token Lifetime
          </label>
          <select
            {...register('ttl_preset')}
            className="block w-full rounded-md border-gray-300 dark:border-gray-600 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm dark:bg-gray-700 dark:text-gray-100"
          >
            {TTL_PRESETS.map((preset) => (
              <option key={preset.value} value={preset.value}>
                {preset.label}
              </option>
            ))}
          </select>
        </div>

        {/* Custom TTL */}
        {ttlPreset === 'custom' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Custom TTL (seconds)
            </label>
            <Input
              type="number"
              {...register('ttl_custom')}
              placeholder="e.g., 7200 for 2 hours"
              error={errors.ttl_custom?.message}
            />
          </div>
        )}

        {/* OR Expiration date */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-300 dark:border-gray-600" />
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-white dark:bg-gray-800 text-gray-500 dark:text-gray-400">
              OR set explicit expiry
            </span>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Expires At
          </label>
          <Input
            type="datetime-local"
            {...register('expires_at')}
            error={errors.expires_at?.message}
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Cannot be combined with TTL presets.
          </p>
        </div>

        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-600 dark:text-red-400">
              Failed to mint token. Please try again.
            </p>
          </div>
        )}

        <ModalFooter>
          <Button
            type="button"
            variant="secondary"
            onClick={handleClose}
            disabled={isMinting}
          >
            Cancel
          </Button>
          <Button
            type="submit"
            variant="primary"
            loading={isMinting}
          >
            Mint Token
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}
