/**
 * EgressAllowlistModal - Modal for creating/editing egress allowlist entries
 */

import { useState, useEffect, useMemo } from 'react';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Textarea } from '@/components/ui/Textarea';
import { Button } from '@/components/ui/Button';
import { useToast } from '@/components/ui/Toast';
import {
  createEgressAllowlistEntry,
  updateEgressAllowlistEntry,
} from '@/api/admin';
import type {
  EgressAllowlistEntry,
  EgressAllowlistEntryKind,
  CreateEgressAllowlistEntryRequest,
  UpdateEgressAllowlistEntryRequest,
} from '@/api/types';

/**
 * Check if an allowlist entry may be risky in production.
 * Returns a warning message for potentially sensitive internal network targets.
 */
function getEntrySecurityWarning(
  kind: EgressAllowlistEntryKind,
  value: string
): string | null {
  const trimmedValue = value.trim().toLowerCase();
  if (!trimmedValue) return null;

  if (kind === 'ip-cidr') {
    if (trimmedValue === '0.0.0.0/0' || trimmedValue === '::/0') {
      return 'This CIDR allows all IPs. This may pose a security risk (SSRF) in production environments.';
    }
  }

  const internalPatterns: { pattern: RegExp; description: string }[] = [
    { pattern: /^localhost$/i, description: 'localhost' },
    { pattern: /^127\./i, description: 'loopback addresses (127.x.x.x)' },
    { pattern: /^0\.0\.0\.0$/i, description: 'all interfaces (0.0.0.0)' },
    { pattern: /^10\./i, description: 'private network (10.x.x.x)' },
    { pattern: /^172\.(1[6-9]|2[0-9]|3[0-1])\./i, description: 'private network (172.16-31.x.x.x)' },
    { pattern: /^192\.168\./i, description: 'private network (192.168.x.x.x)' },
    { pattern: /^::1$/i, description: 'IPv6 loopback (::1)' },
    { pattern: /^fe80:/i, description: 'IPv6 link-local (fe80::)' },
    { pattern: /^fc00:/i, description: 'IPv6 unique local (fc00::)' },
    { pattern: /^fd00:/i, description: 'IPv6 unique local (fd00::)' },
    { pattern: /^169\.254\./i, description: 'link-local (169.254.x.x.x)' },
  ];

  for (const { pattern, description } of internalPatterns) {
    if (pattern.test(trimmedValue)) {
      return `This entry includes ${description}. This is typically safe for local development but may pose a security risk (SSRF) in production environments.`;
    }
  }

  return null;
}

interface EgressAllowlistModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
  entry?: EgressAllowlistEntry | null;
}

/**
 * Modal for creating or editing egress allowlist entries
 */
export function EgressAllowlistModal({
  open,
  onClose,
  onSuccess,
  entry,
}: EgressAllowlistModalProps) {
  const { addToast } = useToast();
  const isEditMode = Boolean(entry);

  // Form state
  const [kind, setKind] = useState<EgressAllowlistEntryKind>('hostname');
  const [value, setValue] = useState('');
  const [comment, setComment] = useState('');
  const [expiresAt, setExpiresAt] = useState('');
  const [errors, setErrors] = useState<{
    value?: string;
    expiresAt?: string;
  }>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Reset form when modal opens/closes or entry changes
  useEffect(() => {
    if (open) {
      if (entry) {
        setKind(entry.kind);
        setValue(entry.value);
        setComment(entry.comment || '');
        // Convert ISO string to datetime-local format
        if (entry.expires_at) {
          const date = new Date(entry.expires_at);
          const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
          setExpiresAt(localDate.toISOString().slice(0, 16));
        } else {
          setExpiresAt('');
        }
      } else {
        setKind('hostname');
        setValue('');
        setComment('');
        setExpiresAt('');
      }
      setErrors({});
    }
  }, [open, entry]);

  const validateValue = (
    kind: EgressAllowlistEntryKind,
    value: string
  ): string | undefined => {
    const trimmed = value.trim();
    if (!trimmed) return 'Value is required';
    if (/[\r\n]/.test(trimmed)) return 'Value must not contain newline characters';

    if (trimmed.includes('://') || trimmed.includes('/')) {
      return 'Value must be a hostname, domain suffix, or CIDR (not a full URL)';
    }

    if (kind !== 'ip-cidr' && trimmed.includes(':')) {
      return 'Do not include a port; enter only the hostname/domain (e.g. host.docker.internal)';
    }

    if (kind === 'domain-suffix' && trimmed.includes('*')) {
      return 'Domain suffix must not include *';
    }

    if (kind === 'ip-cidr' && !trimmed.includes('/')) {
      return 'CIDR must include a prefix length (e.g. 10.0.0.0/8)';
    }

    return undefined;
  };

  // Validate expiration date
  const validateExpiresAt = (value: string): string | undefined => {
    if (value) {
      const selectedDate = new Date(value);
      const now = new Date();

      if (selectedDate <= now) {
        return 'Expiration date must be in the future';
      }
    }

    return undefined;
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validate form
    const valueError = validateValue(kind, value);
    const expiresAtError = validateExpiresAt(expiresAt);

    if (valueError || expiresAtError) {
      setErrors({
        value: valueError,
        expiresAt: expiresAtError,
      });
      return;
    }

    setIsSubmitting(true);

    try {
      if (isEditMode && entry) {
        // Update existing entry
        const updateData: UpdateEgressAllowlistEntryRequest = {
          kind,
          value: value.trim(),
          comment: comment.trim() || undefined,
          expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
        };

        await updateEgressAllowlistEntry(entry.entry_id, updateData);

        addToast({
          type: 'success',
          title: 'Entry updated',
          message: `${kind} "${value}" has been updated`,
        });
      } else {
        // Create new entry
        const createData: CreateEgressAllowlistEntryRequest = {
          kind,
          value: value.trim(),
          comment: comment.trim() || undefined,
          expires_at: expiresAt ? new Date(expiresAt).toISOString() : undefined,
        };

        await createEgressAllowlistEntry(createData);

        addToast({
          type: 'success',
          title: 'Entry created',
          message: `${kind} "${value}" has been added to the allowlist`,
        });
      }

      onSuccess();
      onClose();
    } catch (err: any) {
      console.error('Failed to save entry:', err);
      addToast({
        type: 'error',
        title: isEditMode ? 'Failed to update entry' : 'Failed to create entry',
        message: err.message || 'An error occurred',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleValueChange = (value: string) => {
    setValue(value);
    if (errors.value) {
      const error = validateValue(kind, value);
      setErrors({ ...errors, value: error });
    }
  };

  const entryWarning = useMemo(() => {
    return getEntrySecurityWarning(kind, value);
  }, [kind, value]);

  // Handle expiration date change with validation
  const handleExpiresAtChange = (value: string) => {
    setExpiresAt(value);
    if (errors.expiresAt) {
      const error = validateExpiresAt(value);
      setErrors({ ...errors, expiresAt: error });
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={isEditMode ? 'Edit Allowlist Entry' : 'Add Allowlist Entry'}
      description={
        isEditMode
          ? 'Update the destination, comment, or expiration date'
          : 'Allow a destination host/network for upstream proxying'
      }
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Kind + Value */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div>
            <label
              htmlFor="kind"
              className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
            >
              Kind <span className="text-red-500">*</span>
            </label>
            <select
              id="kind"
              className="mt-1 block w-full rounded-md border-gray-300 text-sm focus:border-primary-500 focus:ring-primary-500 dark:bg-gray-800 dark:border-gray-600 dark:text-gray-200"
              value={kind}
              onChange={(e) => {
                const nextKind = e.target.value as EgressAllowlistEntryKind;
                setKind(nextKind);
                const nextError = validateValue(nextKind, value);
                setErrors((prev) => ({ ...prev, value: nextError }));
              }}
              disabled={isSubmitting}
            >
              <option value="hostname">Hostname</option>
              <option value="domain-suffix">Domain Suffix</option>
              <option value="ip-cidr">IP CIDR</option>
            </select>
          </div>

          <div className="sm:col-span-2">
            <label
              htmlFor="value"
              className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
            >
              Value <span className="text-red-500">*</span>
            </label>
            <Input
              id="value"
              type="text"
              value={value}
              onChange={(e) => handleValueChange(e.target.value)}
              placeholder={
                kind === 'hostname'
                  ? 'host.docker.internal'
                  : kind === 'domain-suffix'
                    ? 'example.com'
                    : '10.0.0.0/8'
              }
              error={errors.value}
              disabled={isSubmitting}
              required
            />
            {errors.value && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.value}</p>
            )}
            {entryWarning && !errors.value && (
              <div className="mt-2 flex items-start gap-2 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-md">
                <ExclamationTriangleIcon className="h-5 w-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-amber-700 dark:text-amber-300">{entryWarning}</p>
              </div>
            )}
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Enter a hostname/domain/CIDR only (no scheme, port, or path).
            </p>
          </div>
        </div>

        {/* Comment Input */}
        <div>
          <label
            htmlFor="comment"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
          >
            Comment
          </label>
          <Textarea
            id="comment"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Optional: why this destination is allowed"
            rows={3}
            disabled={isSubmitting}
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Optional: Describe the purpose of this allowlist entry
          </p>
        </div>

        {/* Expiration Date Input */}
        <div>
          <label
            htmlFor="expiresAt"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
          >
            Expiration Date
          </label>
          <Input
            id="expiresAt"
            type="datetime-local"
            value={expiresAt}
            onChange={(e) => handleExpiresAtChange(e.target.value)}
            error={errors.expiresAt}
            disabled={isSubmitting}
          />
          {errors.expiresAt && (
            <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.expiresAt}</p>
          )}
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Optional: Set an expiration date for temporary access. Leave empty for permanent
            access.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end space-x-3 pt-4">
          <Button variant="ghost" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" loading={isSubmitting}>
            {isEditMode ? 'Update Entry' : 'Add Entry'}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
