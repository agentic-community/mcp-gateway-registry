import React, { useState } from 'react';
import {
  ClipboardDocumentIcon,
  ClipboardDocumentCheckIcon,
  CheckIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';

interface CopyButtonProps {
  /** Returns the text to place on the clipboard (called on click). */
  getText: () => string;
  /** Optional async copy delegate (e.g. parent shows a toast). Falls back to navigator.clipboard. */
  onCopy?: (text: string) => Promise<void> | void;
  /** Label shown next to the icon. */
  label?: string;
  /** Label shown for ~2s after a successful copy. */
  copiedLabel?: string;
  /** `solid` = filled blue→green button; `subtle` = muted gray chip. */
  variant?: 'solid' | 'subtle';
  disabled?: boolean;
  title?: string;
}

/**
 * Copy-to-clipboard button with the blue→green "Copied" feedback that was
 * duplicated across ServerDetailsModal, CustomEntityDetail, and SkillCard's
 * inline modal. Centralizing it keeps the feedback timing and styling identical
 * everywhere.
 */
const CopyButton: React.FC<CopyButtonProps> = ({
  getText,
  onCopy,
  label = 'Copy JSON',
  copiedLabel = 'Copied',
  variant = 'solid',
  disabled = false,
  title,
}) => {
  const [copied, setCopied] = useState(false);

  const handleClick = async () => {
    try {
      const text = getText();
      if (onCopy) {
        await onCopy(text);
      } else {
        await navigator.clipboard.writeText(text);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  };

  const solid = copied
    ? 'btn-base bg-green-600 text-white shadow-sm px-3'
    : 'btn-primary px-3';
  const subtle = clsx(
    'inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg transition-colors',
    'text-fg-muted bg-gray-100 hover:bg-gray-200 hover:text-fg dark:bg-gray-700 dark:hover:bg-gray-600 disabled:opacity-50',
  );

  const iconClass = variant === 'subtle' ? 'h-4 w-4' : 'h-4 w-4';

  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      title={title}
      className={variant === 'solid' ? solid : subtle}
    >
      {copied ? (
        <>
          {variant === 'subtle' ? (
            <ClipboardDocumentCheckIcon
              className={clsx(iconClass, 'text-green-600 dark:text-green-400')}
            />
          ) : (
            <CheckIcon className={iconClass} />
          )}
          {copiedLabel}
        </>
      ) : (
        <>
          <ClipboardDocumentIcon className={iconClass} />
          {label}
        </>
      )}
    </button>
  );
};

export default CopyButton;
