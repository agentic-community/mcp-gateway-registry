import React from 'react';
import FormField from './FormField';
import { FIELD } from './formClasses';

export type Visibility = 'public' | 'private' | 'group-restricted';

interface VisibilityFieldProps {
  value: Visibility;
  onChange: (visibility: Visibility) => void;
  /** Comma-separated allowed groups (shown only for group-restricted). */
  allowedGroups: string;
  onAllowedGroupsChange: (value: string) => void;
}

/**
 * Visibility select plus the conditional "Allowed Groups" input that appears
 * only for group-restricted visibility — the cascade duplicated in the agent
 * form and RegisterPage. (Skills use a simpler 3-option select without groups;
 * custom entities use a chip widget — those keep their own controls.)
 */
const VisibilityField: React.FC<VisibilityFieldProps> = ({
  value,
  onChange,
  allowedGroups,
  onAllowedGroupsChange,
}) => {
  return (
    <>
      <FormField label="Visibility">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value as Visibility)}
          className={FIELD}
        >
          <option value="private">Private</option>
          <option value="public">Public</option>
          <option value="group-restricted">Group Restricted</option>
        </select>
      </FormField>

      {value === 'group-restricted' && (
        <FormField
          label="Allowed Groups"
          hint="Comma-separated list of groups that can access this resource"
        >
          <input
            type="text"
            value={allowedGroups}
            onChange={(e) => onAllowedGroupsChange(e.target.value)}
            className={FIELD}
            placeholder="e.g. finance-team, engineering"
          />
          {allowedGroups.trim() === '' && (
            <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
              At least one group is required for group-restricted visibility
            </p>
          )}
        </FormField>
      )}
    </>
  );
};

export default VisibilityField;
