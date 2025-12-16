import { useState, useCallback } from 'react';
import {
  ClipboardIcon,
  ClipboardDocumentCheckIcon,
} from '@heroicons/react/24/outline';
import { cn } from '../../lib/cn';
import { Button, type ButtonProps } from './Button';

export interface CopyButtonProps extends Omit<ButtonProps, 'onClick'> {
  text: string;
  onCopy?: () => void;
  successDuration?: number;
  label?: string;
  copiedLabel?: string;
}

export function CopyButton({
  text,
  onCopy,
  successDuration = 2000,
  label = 'Copy',
  copiedLabel = 'Copied!',
  variant = 'ghost',
  size = 'sm',
  className,
  ...props
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      onCopy?.();
      setTimeout(() => setCopied(false), successDuration);
    } catch {
      // Fallback for environments without clipboard API
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        setCopied(true);
        onCopy?.();
        setTimeout(() => setCopied(false), successDuration);
      } finally {
        document.body.removeChild(textArea);
      }
    }
  }, [text, onCopy, successDuration]);

  return (
    <Button
      type="button"
      variant={variant}
      size={size}
      onClick={handleCopy}
      className={cn(
        copied && 'text-green-600 dark:text-green-400',
        className
      )}
      leftIcon={
        copied ? (
          <ClipboardDocumentCheckIcon className="h-4 w-4" />
        ) : (
          <ClipboardIcon className="h-4 w-4" />
        )
      }
      {...props}
    >
      {copied ? copiedLabel : label}
    </Button>
  );
}

export interface CopyIconButtonProps {
  text: string;
  onCopy?: () => void;
  successDuration?: number;
  className?: string;
  'aria-label'?: string;
}

export function CopyIconButton({
  text,
  onCopy,
  successDuration = 2000,
  className,
  'aria-label': ariaLabel = 'Copy to clipboard',
}: CopyIconButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      onCopy?.();
      setTimeout(() => setCopied(false), successDuration);
    } catch {
      // Fallback
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        setCopied(true);
        onCopy?.();
        setTimeout(() => setCopied(false), successDuration);
      } finally {
        document.body.removeChild(textArea);
      }
    }
  }, [text, onCopy, successDuration]);

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={cn(
        'p-1 rounded',
        'text-gray-500 hover:text-gray-700',
        'dark:text-gray-400 dark:hover:text-gray-200',
        'focus:outline-none focus:ring-2 focus:ring-primary-500',
        copied && 'text-green-600 dark:text-green-400',
        className
      )}
      aria-label={ariaLabel}
    >
      {copied ? (
        <ClipboardDocumentCheckIcon className="h-5 w-5" />
      ) : (
        <ClipboardIcon className="h-5 w-5" />
      )}
    </button>
  );
}
