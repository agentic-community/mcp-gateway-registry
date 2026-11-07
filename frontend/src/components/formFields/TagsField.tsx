import React from 'react';
import FormField from './FormField';
import { FIELD } from './formClasses';

interface TagsFieldProps {
  /** Current tags. */
  value: string[];
  /** Called with the parsed tag array (comma-split, trimmed, empties dropped). */
  onChange: (tags: string[]) => void;
  label?: string;
  hint?: React.ReactNode;
  placeholder?: string;
}

/**
 * Comma-separated tags input bound to a string[] — the pattern repeated across
 * the server/agent/virtual-server forms (`value={tags.join(',')}` +
 * split/trim/filter on change). Forms that keep tags as a raw string (e.g. the
 * skill form, which parses on save) use a plain input instead.
 */
const TagsField: React.FC<TagsFieldProps> = ({
  value,
  onChange,
  label = 'Tags',
  hint,
  placeholder = 'tag1,tag2,tag3',
}) => {
  return (
    <FormField label={label} hint={hint}>
      <input
        type="text"
        value={value.join(',')}
        onChange={(e) =>
          onChange(
            e.target.value
              .split(',')
              .map((t) => t.trim())
              .filter((t) => t),
          )
        }
        className={FIELD}
        placeholder={placeholder}
      />
    </FormField>
  );
};

export default TagsField;
