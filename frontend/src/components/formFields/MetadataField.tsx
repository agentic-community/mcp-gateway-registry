import React from 'react';
import FormField from './FormField';
import { FIELD } from './formClasses';

interface MetadataFieldProps {
  value: string;
  onChange: (value: string) => void;
  label?: string;
  hint?: React.ReactNode;
  placeholder?: string;
  rows?: number;
}

const DEFAULT_HINT =
  'Custom key-value pairs in JSON format for searchable metadata';

/**
 * The "Custom Metadata (JSON, optional)" textarea shared by the server, agent,
 * and skill forms — a monospace textarea with a hint line.
 */
const MetadataField: React.FC<MetadataFieldProps> = ({
  value,
  onChange,
  label = 'Custom Metadata (JSON, optional)',
  hint = DEFAULT_HINT,
  placeholder = '{"team": "platform", "owner": "alice@example.com"}',
  rows = 4,
}) => {
  return (
    <FormField label={label} hint={hint}>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={rows}
        className={`${FIELD} font-mono`}
        placeholder={placeholder}
      />
    </FormField>
  );
};

export default MetadataField;
