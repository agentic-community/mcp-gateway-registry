import React from 'react';
import clsx from 'clsx';
import { LABEL, FIELD_ERROR } from './formClasses';

interface FormFieldProps {
  label: React.ReactNode;
  /** Renders a required asterisk after the label. */
  required?: boolean;
  /** Validation error shown under the control (also styles the control border via the child). */
  error?: string | null;
  /** Helper/hint text shown under the control (suppressed when an error is present). */
  hint?: React.ReactNode;
  /** Associates the label with a control id (optional). */
  htmlFor?: string;
  className?: string;
  /** The control (input/select/textarea/custom widget). */
  children: React.ReactNode;
}

/**
 * Label + control + error/hint wrapper shared by every form. Replaces the
 * `<div><label/>...<input/>{error && <p/>}</div>` block repeated ~50 times
 * across the form components. The control itself is passed as children so this
 * works for inputs, selects, textareas, and custom widgets alike.
 */
const FormField: React.FC<FormFieldProps> = ({
  label,
  required = false,
  error,
  hint,
  htmlFor,
  className,
  children,
}) => {
  return (
    <div className={className}>
      <label className={LABEL} htmlFor={htmlFor}>
        {label}
        {required && ' *'}
      </label>
      {children}
      {error ? (
        <p className={FIELD_ERROR}>{error}</p>
      ) : hint ? (
        <p className="mt-1 text-xs text-fg-subtle">{hint}</p>
      ) : null}
    </div>
  );
};

export default FormField;
