import React from 'react';
import FormField from './FormField';
import { FIELD } from './formClasses';

export type LifecycleStatus = 'active' | 'draft' | 'deprecated' | 'beta';

interface StatusFieldProps {
  value: LifecycleStatus;
  onChange: (status: LifecycleStatus) => void;
  label?: string;
}

/**
 * The lifecycle-status select (Active/Draft/Beta/Deprecated) shared by the
 * server, agent, and skill forms.
 */
const StatusField: React.FC<StatusFieldProps> = ({
  value,
  onChange,
  label = 'Lifecycle Status',
}) => {
  return (
    <FormField label={label}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as LifecycleStatus)}
        className={FIELD}
      >
        <option value="active">Active</option>
        <option value="draft">Draft</option>
        <option value="beta">Beta</option>
        <option value="deprecated">Deprecated</option>
      </select>
    </FormField>
  );
};

export default StatusField;
