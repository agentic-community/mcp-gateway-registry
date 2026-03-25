import React, { useState, useEffect, useCallback } from 'react';
import { ShieldCheckIcon, ExclamationTriangleIcon, XCircleIcon, CodeBracketIcon } from '@heroicons/react/24/solid';

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
    badgeClasses: 'bg-gradient-to-r from-emerald-100 to-green-100 text-emerald-700 ' +
      'dark:from-emerald-900/30 dark:to-green-900/30 dark:text-emerald-300 ' +
      'border border-emerald-200 dark:border-emerald-600',
    iconColor: 'text-emerald-600 dark:text-emerald-400',
    modalBadgeClasses: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300',
  },
  expired: {
    label: 'ANS EXPIRED',
    Icon: ExclamationTriangleIcon,
    badgeClasses: 'bg-gradient-to-r from-yellow-100 to-amber-100 text-yellow-700 ' +
      'dark:from-yellow-900/30 dark:to-amber-900/30 dark:text-yellow-300 ' +
      'border border-yellow-200 dark:border-yellow-600',
    iconColor: 'text-yellow-600 dark:text-yellow-400',
    modalBadgeClasses: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300',
  },
  revoked: {
    label: 'ANS REVOKED',
    Icon: XCircleIcon,
    badgeClasses: 'bg-gradient-to-r from-red-100 to-rose-100 text-red-700 ' +
      'dark:from-red-900/30 dark:to-rose-900/30 dark:text-red-300 ' +
      'border border-red-200 dark:border-red-600',
    iconColor: 'text-red-600 dark:text-red-400',
    modalBadgeClasses: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300',
  },
  not_found: {
    label: 'ANS NOT FOUND',
    Icon: ExclamationTriangleIcon,
    badgeClasses: 'bg-gradient-to-r from-gray-100 to-slate-100 text-gray-700 ' +
      'dark:from-gray-900/30 dark:to-slate-900/30 dark:text-gray-300 ' +
      'border border-gray-200 dark:border-gray-600',
    iconColor: 'text-gray-600 dark:text-gray-400',
    modalBadgeClasses: 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-300',
  },
  pending: {
    label: 'ANS PENDING',
    Icon: ShieldCheckIcon,
    badgeClasses: 'bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-700 ' +
      'dark:from-blue-900/30 dark:to-indigo-900/30 dark:text-blue-300 ' +
      'border border-blue-200 dark:border-blue-600',
    iconColor: 'text-blue-600 dark:text-blue-400',
    modalBadgeClasses: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300',
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
      <div className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl max-w-lg w-full mx-4 p-6 max-h-[85vh] overflow-y-auto"
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
                    ? 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-400'
                    : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
                title="View raw ANS JSON"
              >
                <CodeBracketIcon className="h-4 w-4" />
              </button>
            )}
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 text-xl leading-none"
            >
              &times;
            </button>
          </div>
        </div>

        {/* Raw JSON View */}
        {showRawJson && ansMetadata.raw_ans_response && (
          <div className="mb-4">
            <pre className="text-[11px] font-mono bg-gray-950 text-green-400 p-4 rounded-lg overflow-x-auto max-h-[60vh]">
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
              <code className="text-xs font-mono bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded break-all">
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
                  className="text-cyan-600 dark:text-cyan-400 hover:underline text-sm"
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
                  className="text-cyan-600 dark:text-cyan-400 hover:underline text-xs font-mono break-all"
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
              <div className="border-t dark:border-gray-700 pt-3">
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
              <div className="border-t dark:border-gray-700 pt-3">
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  Endpoints
                </div>
                <div className="space-y-2.5">
                  {ansMetadata.endpoints!.map((ep, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex items-center gap-2 text-xs">
                        <span className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 rounded font-medium uppercase text-[10px]">
                          {ep.type || 'HTTP'}
                        </span>
                        <a
                          href={ep.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-cyan-600 dark:text-cyan-400 hover:underline font-mono truncate"
                        >
                          {ep.url}
                        </a>
                        {ep.protocol && (
                          <span className="px-1.5 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 rounded text-[10px] font-medium flex-shrink-0">
                            {ep.protocol}
                          </span>
                        )}
                      </div>
                      {ep.transports && ep.transports.length > 0 && (
                        <div className="flex items-center gap-1 ml-12">
                          <span className="text-[10px] text-gray-400 dark:text-gray-500">Transport:</span>
                          {ep.transports.map((t, ti) => (
                            <span key={ti} className="px-1 py-0.5 bg-gray-50 dark:bg-gray-800/80 text-gray-500 dark:text-gray-400 rounded text-[10px]">
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
              <div className="border-t dark:border-gray-700 pt-3">
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
                            <span key={ti} className="px-1 py-0.5 bg-cyan-50 dark:bg-cyan-900/30 text-cyan-600 dark:text-cyan-400 rounded text-[10px]">
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
              <div className="border-t dark:border-gray-700 pt-3">
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  ANS API Links
                </div>
                <div className="space-y-1.5">
                  {ansMetadata.links!.map((link, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-xs">
                      <span className="font-medium text-gray-600 dark:text-gray-400 min-w-[130px]">
                        {LINK_LABELS[link.rel || ''] || link.rel}:
                      </span>
                      <a
                        href={link.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-cyan-600 dark:text-cyan-400 hover:underline font-mono truncate"
                      >
                        {link.href}
                      </a>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Description from ANS */}
            {ansMetadata.ans_description && (
              <div className="border-t dark:border-gray-700 pt-3">
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                  ANS Description
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">{ansMetadata.ans_description}</p>
              </div>
            )}

            {/* Last Verified */}
            {ansMetadata.last_verified && (
              <div className="border-t dark:border-gray-700 pt-3 text-xs text-gray-500 dark:text-gray-400">
                Last Verified: {new Date(ansMetadata.last_verified).toLocaleString()}
              </div>
            )}
          </div>
        )}

        {/* Close button */}
        <div className="mt-5 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium bg-gray-100 hover:bg-gray-200
              dark:bg-gray-800 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default ANSBadge;
