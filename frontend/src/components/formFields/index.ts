/**
 * Shared form-field primitives.
 *
 * One source of truth for the label/input/error markup and the field class
 * strings that were redefined across every form component (RegisterPage, the
 * entity edit/register modals, IAM forms, federation forms, custom-entity form).
 * Compose these instead of hand-rolling `<div><label/><input/>{error}</div>`.
 */
export { default as FormField } from './FormField';
export { default as TagsField } from './TagsField';
export { default as StatusField } from './StatusField';
export type { LifecycleStatus } from './StatusField';
export { default as VisibilityField } from './VisibilityField';
export type { Visibility } from './VisibilityField';
export { default as ProxyField } from './ProxyField';
export { default as UpstreamHeadersField, upstreamHeaderRowError } from './UpstreamHeadersField';
export type { UpstreamHeader } from './UpstreamHeadersField';
export { default as MetadataField } from './MetadataField';
export { default as AuthSchemeFields } from './AuthSchemeFields';
export type { AuthScheme } from './AuthSchemeFields';
export {
  FIELD,
  FIELD_BASE,
  FIELD_FOCUS,
  LABEL,
  FIELD_ERROR,
  FIELD_ERROR_BORDER,
  fieldClass,
} from './formClasses';
