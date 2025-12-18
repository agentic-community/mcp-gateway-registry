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
  CreateEgressAllowlistEntryRequest,
  UpdateEgressAllowlistEntryRequest,
} from '@/api/types';

/**
 * Check if a pattern matches potentially sensitive internal network targets.
 * Returns a warning message if the pattern is potentially dangerous in production.
 */
function getPatternSecurityWarning(pattern: string): string | null {
  const trimmedPattern = pattern.trim().toLowerCase();

  // Skip empty patterns
  if (!trimmedPattern) {
    return null;
  }

  // Patterns that are typically safe only in development
  const internalPatterns: { pattern: RegExp; description: string }[] = [
    { pattern: /^localhost(:|$)/i, description: 'localhost' },
    { pattern: /^127\./i, description: 'loopback addresses (127.x.x.x)' },
    { pattern: /^0\.0\.0\.0/i, description: 'all interfaces (0.0.0.0)' },
    { pattern: /^10\./i, description: 'private network (10.x.x.x)' },
    { pattern: /^172\.(1[6-9]|2[0-9]|3[0-1])\./i, description: 'private network (172.16-31.x.x)' },
    { pattern: /^192\.168\./i, description: 'private network (192.168.x.x)' },
    { pattern: /^\[?::1\]?/i, description: 'IPv6 loopback (::1)' },
    { pattern: /^\[?fe80:/i, description: 'IPv6 link-local (fe80::)' },
    { pattern: /^\[?fc00:/i, description: 'IPv6 unique local (fc00::)' },
    { pattern: /^\[?fd00:/i, description: 'IPv6 unique local (fd00::)' },
    { pattern: /^169\.254\./i, description: 'link-local (169.254.x.x)' },
    { pattern: /^\*$/i, description: 'wildcard matching all hosts' },
  ];

  for (const { pattern: regex, description } of internalPatterns) {
    if (regex.test(trimmedPattern)) {
      return `This pattern matches ${description}. This is typically safe for local development but may pose a security risk (SSRF) in production environments.`;
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
  const [pattern, setPattern] = useState('');
  const [description, setDescription] = useState('');
  const [expiresAt, setExpiresAt] = useState('');
  const [errors, setErrors] = useState<{
    pattern?: string;
    expiresAt?: string;
  }>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Reset form when modal opens/closes or entry changes
  useEffect(() => {
    if (open) {
      if (entry) {
        setPattern(entry.pattern);
        setDescription(entry.description || '');
        // Convert ISO string to datetime-local format
        if (entry.expires_at) {
          const date = new Date(entry.expires_at);
          const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
          setExpiresAt(localDate.toISOString().slice(0, 16));
        } else {
          setExpiresAt('');
        }
      } else {
        setPattern('');
        setDescription('');
        setExpiresAt('');
      }
      setErrors({});
    }
  }, [open, entry]);

  // Validate pattern
  const validatePattern = (value: string): string | undefined => {
    if (!value.trim()) {
      return 'Pattern is required';
    }

    // Basic validation for hostname/URL pattern
    // Allow wildcards (*), hostnames, URLs, and IP addresses
    const validPatternRegex = /^[\w\-.*:/?#[\]@!$&'()+,;=%]+$/;
    if (!validPatternRegex.test(value)) {
      return 'Invalid pattern format';
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
    const patternError = validatePattern(pattern);
    const expiresAtError = validateExpiresAt(expiresAt);

    if (patternError || expiresAtError) {
      setErrors({
        pattern: patternError,
        expiresAt: expiresAtError,
      });
      return;
    }

    setIsSubmitting(true);

    try {
      if (isEditMode && entry) {
        // Update existing entry
        const updateData: UpdateEgressAllowlistEntryRequest = {
          pattern: pattern.trim(),
          description: description.trim() || undefined,
          expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
        };

        await updateEgressAllowlistEntry(entry.entry_id, updateData);

        addToast({
          type: 'success',
          title: 'Entry updated',
          message: `Pattern "${pattern}" has been updated`,
        });
      } else {
        // Create new entry
        const createData: CreateEgressAllowlistEntryRequest = {
          pattern: pattern.trim(),
          description: description.trim() || undefined,
          expires_at: expiresAt ? new Date(expiresAt).toISOString() : undefined,
        };

        await createEgressAllowlistEntry(createData);

        addToast({
          type: 'success',
          title: 'Entry created',
          message: `Pattern "${pattern}" has been added to the allowlist`,
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

  // Handle pattern change with validation
  const handlePatternChange = (value: string) => {
    setPattern(value);
    if (errors.pattern) {
      const error = validatePattern(value);
      setErrors({ ...errors, pattern: error });
    }
  };

  // Compute security warning for the current pattern
  const patternWarning = useMemo(() => {
    return getPatternSecurityWarning(pattern);
  }, [pattern]);

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
          ? 'Update the pattern, description, or expiration date'
          : 'Add a new pattern to control which upstream servers can be proxied'
      }
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Pattern Input */}
        <div>
          <label
            htmlFor="pattern"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
          >
            Pattern <span className="text-red-500">*</span>
          </label>
          <Input
            id="pattern"
            type="text"
            value={pattern}
            onChange={(e) => handlePatternChange(e.target.value)}
            placeholder="e.g., localhost:*, *.example.com, https://api.example.com/*"
            error={errors.pattern}
            disabled={isSubmitting}
            required
          />
          {errors.pattern && (
            <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.pattern}</p>
          )}
          {patternWarning && !errors.pattern && (
            <div className="mt-2 flex items-start gap-2 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-md">
              <ExclamationTriangleIcon className="h-5 w-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-amber-700 dark:text-amber-300">{patternWarning}</p>
            </div>
          )}
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Use wildcards (*) for flexible matching. Examples: localhost:*, *.example.com,
            192.168.*.*
          </p>
        </div>

        {/* Description Input */}
        <div>
          <label
            htmlFor="description"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
          >
            Description
          </label>
          <Textarea
            id="description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Optional description for this allowlist entry"
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
