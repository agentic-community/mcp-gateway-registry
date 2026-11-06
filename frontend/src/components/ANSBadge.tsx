import React, { useState, useEffect, useCallback } from 'react';
import { ShieldCheckIcon, ExclamationTriangleIcon, XCircleIcon, CodeBracketIcon, XMarkIcon } from '@heroicons/react/24/solid';
import { SafeLink } from './SafeLink';

interface ANSFunction {
  id?: string;
  name?: string;
  tags?: string[] | null;
}

interface ANSEndpoint {
  type?: string;
  url?: string;
  protocol?: string;
  transports?: string[];
  functions?: ANSFunction[];
}

interface ANSLink {
  rel?: string;
  href?: string;
}

interface ANSMetadata {
  ans_agent_id: string;
  status: 'verified' | 'expired' | 'revoked' | 'not_found' | 'pending';
  domain?: string;
  organization?: string;
  ans_name?: string;
  ans_display_name?: string;
  ans_description?: string;
  ans_version?: string;
  registered_with_ans_at?: string;
  certificate?: {
    not_after?: string;
    not_before?: string;
    subject_dn?: string;
    issuer_dn?: string;
    serial_number?: string;
  };
  endpoints?: ANSEndpoint[];
  links?: ANSLink[];
  raw_ans_response?: Record<string, unknown>;
  last_verified?: string;
}

interface ANSBadgeProps {
  ansMetadata: ANSMetadata | null | undefined;
  compact?: boolean;
}

const STATUS_CONFIG = {
  verified: {
    label: 'ANS VERIFIED',
    Icon: ShieldCheckIcon,
    badgeClasses: 'bg-gradient-to-r from-success-100 to-success-100 text-success-700 ' +
      'dark:from-success-900/30 dark:to-success-900/30 dark:text-success-300 ' +
      'border border-success-200 dark:border-success-600',
    iconColor: 'text-success-600 dark:text-success-400',
    modalBadgeClasses: 'bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-300',
  },
  expired: {
    label: 'ANS EXPIRED',
    Icon: ExclamationTriangleIcon,
    badgeClasses: 'bg-gradient-to-r from-warning-100 to-warning-100 text-warning-700 ' +
      'dark:from-warning-900/30 dark:to-warning-900/30 dark:text-warning-300 ' +
      'border border-warning-200 dark:border-warning-600',
    iconColor: 'text-warning-600 dark:text-warning-400',
    modalBadgeClasses: 'bg-warning-100 text-warning-700 dark:bg-warning-900/30 dark:text-warning-300',
  },
  revoked: {
    label: 'ANS REVOKED',
    Icon: XCircleIcon,
    badgeClasses: 'bg-gradient-to-r from-danger-100 to-danger-100 text-danger-700 ' +
      'dark:from-danger-900/30 dark:to-danger-900/30 dark:text-danger-300 ' +
      'border border-danger-200 dark:border-danger-600',
    iconColor: 'text-danger-600 dark:text-danger-400',
    modalBadgeClasses: 'bg-danger-100 text-danger-700 dark:bg-danger-900/30 dark:text-danger-300',
  },
  not_found: {
    label: 'ANS NOT FOUND',
    Icon: ExclamationTriangleIcon,
    badgeClasses: 'bg-gradient-to-r from-gray-100 to-gray-100 text-gray-700 ' +
      'dark:from-gray-900/30 dark:to-gray-900/30 dark:text-gray-300 ' +
      'border border-gray-200 dark:border-gray-600',
    iconColor: 'text-gray-600 dark:text-gray-400',
    modalBadgeClasses: 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-300',
  },
  pending: {
    label: 'ANS PENDING',
    Icon: ShieldCheckIcon,
    badgeClasses: 'bg-gradient-to-r from-info-100 to-info-100 text-info-700 ' +
      'dark:from-info-900/30 dark:to-info-900/30 dark:text-info-300 ' +
      'border border-info-200 dark:border-info-600',
    iconColor: 'text-info-600 dark:text-info-400',
    modalBadgeClasses: 'bg-info-100 text-info-700 dark:bg-info-900/30 dark:text-info-300',
  },
};

const LINK_LABELS: Record<string, string> = {
  'self': 'ANS Agent API',
  'server-certificates': 'Server Certificates',
  'identity-certificates': 'Identity Certificates',
  'agent-details': 'Agent Details',
};


export const ANSBadge: React.FC<ANSBadgeProps> = ({ ansMetadata }) => {
  const [showModal, setShowModal] = useState(false);

  if (!ansMetadata) return null;

  const config = STATUS_CONFIG[ansMetadata.status] || STATUS_CONFIG.pending;
  const { label, Icon, badgeClasses, iconColor } = config;

  return (
    <>
      <span
        className={`px-2 py-0.5 text-xs font-semibold rounded-full flex-shrink-0
          cursor-pointer inline-flex items-center gap-1 ${badgeClasses}`}
        title={`ANS: ${ansMetadata.domain || ansMetadata.ans_agent_id}`}
        onClick={() => setShowModal(true)}
      >
        <Icon className={`h-3.5 w-3.5 ${iconColor}`} />
        {label}
      </span>

      {showModal && (
        <ANSCertificateModal
          ansMetadata={ansMetadata}
          onClose={() => setShowModal(false)}
        />
      )}
    </>
  );
};


interface ANSCertificateModalProps {
  ansMetadata: ANSMetadata;
  onClose: () => void;
}

const ANSCertificateModal: React.FC<ANSCertificateModalProps> = ({ ansMetadata, onClose }) => {
  const [showRawJson, setShowRawJson] = useState(false);
  const config = STATUS_CONFIG[ansMetadata.status] || STATUS_CONFIG.pending;
  const { label, Icon, iconColor, modalBadgeClasses } = config;

  // Close on ESC key
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  }, [onClose]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const hasCertDetails = ansMetadata.certificate && (
    ansMetadata.certificate.subject_dn ||
    ansMetadata.certificate.issuer_dn ||
    ansMetadata.certificate.not_after
  );

  const hasEndpoints = ansMetadata.endpoints && ansMetadata.endpoints.length > 0;
  const hasLinks = ansMetadata.links && ansMetadata.links.length > 0;

  // Collect all unique functions across endpoints
  const allFunctions = (ansMetadata.endpoints || [])
    .flatMap(ep => ep.functions || [])
    .filter(fn => fn && fn.id);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
         onClick={onClose}>
      <div className="bg-surface-overlay border border-line rounded-xl shadow-2xl max-w-lg w-full mx-4 p-6 max-h-[85vh] overflow-y-auto"
           onClick={(e) => e.stopPropagation()}>

        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
            ANS Certificate Details
          </h3>
          <div className="flex items-center gap-2">
            {ansMetadata.raw_ans_response && (
              <button
                onClick={() => setShowRawJson(!showRawJson)}
                className={`p-1.5 rounded-lg transition-colors ${
                  showRawJson
                    ? 'bg-info-100 text-info-700 dark:bg-info-900/30 dark:text-info-400'
                    : 'text-fg-subtle hover:text-fg hover:bg-surface-hover'
                }`}
                title="View raw ANS JSON"
              >
                <CodeBracketIcon className="h-4 w-4" />
              </button>
            )}
            <button
              onClick={onClose}
              className="btn-icon"
              aria-label="Close"
            >
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Raw JSON View */}
        {showRawJson && ansMetadata.raw_ans_response && (
          <div className="mb-4">
            <pre className="text-[11px] font-mono bg-gray-950 text-success-400 p-4 rounded-lg overflow-x-auto max-h-[60vh]">
              {JSON.stringify(ansMetadata.raw_ans_response, null, 2)}
            </pre>
          </div>
        )}

        {/* Normal View */}
        {!showRawJson && (
          <div className="space-y-4 text-sm text-gray-700 dark:text-gray-300">

            {/* Status */}
            <div>
              <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                Status
              </div>
              <span className={`px-2.5 py-1 text-xs font-semibold rounded-full inline-flex items-center gap-1 ${modalBadgeClasses}`}>
                <Icon className={`h-3.5 w-3.5 ${iconColor}`} />
                {label}
              </span>
            </div>

            {/* ANS Display Name */}
            {ansMetadata.ans_display_name && (
              <div>
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                  ANS Registered Name
                </div>
                <span>{ansMetadata.ans_display_name}</span>
              </div>
            )}

            {/* Agent ID */}
            <div>
              <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                Agent ID
              </div>
              <code className="text-xs font-mono bg-gray-500/15 px-2 py-1 rounded break-all">
                {ansMetadata.ans_agent_id}
              </code>
            </div>

            {/* Domain */}
            {ansMetadata.domain && (
              <div>
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                  Domain
                </div>
                <a
                  href={`https://${ansMetadata.domain}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-info-600 dark:text-info-400 hover:underline text-sm"
                >
                  {ansMetadata.domain}
                </a>
              </div>
            )}

            {/* Agent Card URL */}
            {ansMetadata.domain && (
              <div>
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                  Agent Card
                </div>
                <a
                  href={`https://${ansMetadata.domain}/.well-known/agent-card.json`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-info-600 dark:text-info-400 hover:underline text-xs font-mono break-all"
                >
                  https://{ansMetadata.domain}/.well-known/agent-card.json
                </a>
              </div>
            )}

            {/* Organization */}
            {ansMetadata.organization && (
              <div>
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                  Organization
                </div>
                <span>{ansMetadata.organization}</span>
              </div>
            )}

            {/* Version */}
            {ansMetadata.ans_version && (
              <div>
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                  ANS Version
                </div>
                <span className="font-mono text-xs">{ansMetadata.ans_version}</span>
              </div>
            )}

            {/* ANS Registration Date */}
            {ansMetadata.registered_with_ans_at && (
              <div>
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                  Registered with ANS
                </div>
                <span>{new Date(ansMetadata.registered_with_ans_at).toLocaleString()}</span>
              </div>
            )}

            {/* Certificate Section */}
            {hasCertDetails && (
              <div className="border-t border-line pt-3">
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  Certificate
                </div>
                <div className="space-y-1.5 text-xs">
                  {ansMetadata.certificate?.subject_dn && (
                    <div>
                      <span className="font-medium text-gray-600 dark:text-gray-400">Subject:</span>{' '}
                      <span className="font-mono">{ansMetadata.certificate.subject_dn}</span>
                    </div>
                  )}
                  {ansMetadata.certificate?.issuer_dn && (
                    <div>
                      <span className="font-medium text-gray-600 dark:text-gray-400">Issuer:</span>{' '}
                      <span className="font-mono">{ansMetadata.certificate.issuer_dn}</span>
                    </div>
                  )}
                  {ansMetadata.certificate?.not_after && (
                    <div>
                      <span className="font-medium text-gray-600 dark:text-gray-400">Expires:</span>{' '}
                      <span>{new Date(ansMetadata.certificate.not_after).toLocaleDateString()}</span>
                    </div>
                  )}
                  {ansMetadata.certificate?.serial_number && (
                    <div>
                      <span className="font-medium text-gray-600 dark:text-gray-400">Serial:</span>{' '}
                      <span className="font-mono">{ansMetadata.certificate.serial_number}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Endpoints Section */}
            {hasEndpoints && (
              <div className="border-t border-line pt-3">
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  Endpoints
                </div>
                <div className="space-y-2.5">
                  {ansMetadata.endpoints!.map((ep, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex items-center gap-2 text-xs">
                        <span className="px-1.5 py-0.5 bg-gray-500/15 rounded font-medium uppercase text-[10px]">
                          {ep.type || 'HTTP'}
                        </span>
                        <SafeLink
                          href={ep.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-info-600 dark:text-info-400 hover:underline font-mono truncate"
                        >
                          {ep.url}
                        </SafeLink>
                        {ep.protocol && (
                          <span className="px-1.5 py-0.5 bg-info-50 dark:bg-info-900/30 text-info-600 dark:text-info-400 rounded text-[10px] font-medium flex-shrink-0">
                            {ep.protocol}
                          </span>
                        )}
                      </div>
                      {ep.transports && ep.transports.length > 0 && (
                        <div className="flex items-center gap-1 ml-12">
                          <span className="text-[10px] text-gray-400 dark:text-gray-500">Transport:</span>
                          {ep.transports.map((t, ti) => (
                            <span key={ti} className="px-1 py-0.5 bg-gray-500/10 text-fg-muted rounded text-[10px]">
                              {t}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Functions/Skills Section */}
            {allFunctions.length > 0 && (
              <div className="border-t border-line pt-3">
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  Functions
                </div>
                <div className="space-y-1.5">
                  {allFunctions.map((fn, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-xs">
                      <span className="font-medium text-gray-700 dark:text-gray-300">{fn.name || fn.id}</span>
                      {fn.tags && fn.tags.length > 0 && (
                        <div className="flex gap-1">
                          {fn.tags.map((tag, ti) => (
                            <span key={ti} className="px-1 py-0.5 bg-info-50 dark:bg-info-900/30 text-info-600 dark:text-info-400 rounded text-[10px]">
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ANS API Links Section */}
            {hasLinks && (
              <div className="border-t border-line pt-3">
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  ANS API Links
                </div>
                <div className="space-y-1.5">
                  {ansMetadata.links!.map((link, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-xs">
                      <span className="font-medium text-gray-600 dark:text-gray-400 min-w-[130px]">
                        {LINK_LABELS[link.rel || ''] || link.rel}:
                      </span>
                      <SafeLink
                        href={link.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-info-600 dark:text-info-400 hover:underline font-mono truncate"
                      >
                        {link.href}
                      </SafeLink>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Description from ANS */}
            {ansMetadata.ans_description && (
              <div className="border-t border-line pt-3">
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                  ANS Description
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">{ansMetadata.ans_description}</p>
              </div>
            )}

            {/* Last Verified */}
            {ansMetadata.last_verified && (
              <div className="border-t border-line pt-3 text-xs text-fg-subtle">
                Last Verified: {new Date(ansMetadata.last_verified).toLocaleString()}
              </div>
            )}
          </div>
        )}

        {/* Close button */}
        <div className="mt-5 flex justify-end">
          <button
            onClick={onClose}
            className="btn-secondary"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default ANSBadge;
