/**
 * Shared Tailwind class strings for form controls.
 *
 * These were redefined (with minor drift) in 8+ form components:
 * RegisterPage, the entity edit/register modals, CustomEntityForm,
 * VirtualServerForm, AddRegistryEntryModal, LocalRuntimeFormPanel, etc. This is
 * the single source of truth; forms import these instead of declaring their own.
 *
 * Every field uses the `.input` component class from index.css, so inputs look
 * the same in every form and in both color modes, with one focus ring.
 */

/** Input/select/textarea classes. */
export const FIELD = 'input';

/** Checkbox classes (styled by @tailwindcss/forms), legible in both modes. */
export const CHECKBOX =
  'h-4 w-4 rounded border-line-strong bg-surface text-accent focus:ring-focus';

/** Standard field label. */
export const LABEL = 'block text-sm font-medium text-fg-muted mb-1';

/** Inline validation-error text shown under a field. */
export const FIELD_ERROR = 'mt-1 text-sm text-red-600 dark:text-red-400';

/** Border applied to a field when it has a validation error. */
export const FIELD_ERROR_BORDER = 'border-red-500 dark:border-red-400';

/** Build field classes, optionally flagging an error state. */
export function fieldClass(hasError = false): string {
  return hasError ? `${FIELD} ${FIELD_ERROR_BORDER}` : FIELD;
}
