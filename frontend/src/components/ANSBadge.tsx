import React, { useState } from 'react';
import { ShieldCheckIcon, ExclamationTriangleIcon, XCircleIcon } from '@heroicons/react/24/solid';

interface ANSMetadata {
  ans_agent_id: string;
  status: 'verified' | 'expired' | 'revoked' | 'not_found' | 'pending';
  domain?: string;
  organization?: string;
  certificate?: {
    not_after?: string;
    subject_dn?: string;
    issuer_dn?: string;
  };
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
  },
  expired: {
    label: 'ANS EXPIRED',
    Icon: ExclamationTriangleIcon,
    badgeClasses: 'bg-gradient-to-r from-yellow-100 to-amber-100 text-yellow-700 ' +
      'dark:from-yellow-900/30 dark:to-amber-900/30 dark:text-yellow-300 ' +
      'border border-yellow-200 dark:border-yellow-600',
    iconColor: 'text-yellow-600 dark:text-yellow-400',
  },
  revoked: {
    label: 'ANS REVOKED',
    Icon: XCircleIcon,
    badgeClasses: 'bg-gradient-to-r from-red-100 to-rose-100 text-red-700 ' +
      'dark:from-red-900/30 dark:to-rose-900/30 dark:text-red-300 ' +
      'border border-red-200 dark:border-red-600',
    iconColor: 'text-red-600 dark:text-red-400',
  },
  not_found: {
    label: 'ANS NOT FOUND',
    Icon: ExclamationTriangleIcon,
    badgeClasses: 'bg-gradient-to-r from-gray-100 to-slate-100 text-gray-700 ' +
      'dark:from-gray-900/30 dark:to-slate-900/30 dark:text-gray-300 ' +
      'border border-gray-200 dark:border-gray-600',
    iconColor: 'text-gray-600 dark:text-gray-400',
  },
  pending: {
    label: 'ANS PENDING',
    Icon: ShieldCheckIcon,
    badgeClasses: 'bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-700 ' +
      'dark:from-blue-900/30 dark:to-indigo-900/30 dark:text-blue-300 ' +
      'border border-blue-200 dark:border-blue-600',
    iconColor: 'text-blue-600 dark:text-blue-400',
  },
};

export const ANSBadge: React.FC<ANSBadgeProps> = ({ ansMetadata, compact = false }) => {
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
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
         onClick={onClose}>
      <div className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl max-w-md w-full mx-4 p-6"
           onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
            ANS Certificate Details
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            &times;
          </button>
        </div>

        <div className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
          <div>
            <span className="font-medium">ANS ID:</span>{' '}
            <span className="font-mono text-xs">{ansMetadata.ans_agent_id}</span>
          </div>
          <div>
            <span className="font-medium">Status:</span>{' '}
            <span className="capitalize">{ansMetadata.status}</span>
          </div>
          {ansMetadata.domain && (
            <div><span className="font-medium">Domain:</span> {ansMetadata.domain}</div>
          )}
          {ansMetadata.organization && (
            <div><span className="font-medium">Organization:</span> {ansMetadata.organization}</div>
          )}
          {ansMetadata.certificate && (
            <div className="border-t pt-3 mt-3 dark:border-gray-700">
              <div className="font-semibold mb-2">Certificate</div>
              {ansMetadata.certificate.subject_dn && (
                <div className="text-xs"><span className="font-medium">Subject:</span> {ansMetadata.certificate.subject_dn}</div>
              )}
              {ansMetadata.certificate.issuer_dn && (
                <div className="text-xs"><span className="font-medium">Issuer:</span> {ansMetadata.certificate.issuer_dn}</div>
              )}
              {ansMetadata.certificate.not_after && (
                <div className="text-xs"><span className="font-medium">Expires:</span> {new Date(ansMetadata.certificate.not_after).toLocaleDateString()}</div>
              )}
            </div>
          )}
          {ansMetadata.last_verified && (
            <div className="text-xs text-gray-500">
              Last verified: {new Date(ansMetadata.last_verified).toLocaleString()}
            </div>
          )}
        </div>

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
