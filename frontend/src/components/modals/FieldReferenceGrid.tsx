import React from 'react';

export interface FieldRef {
  /** Field name, rendered as inline code. */
  name: string;
  /** Short human description shown after the code. */
  description: string;
}

interface FieldReferenceColumn {
  heading: string;
  fields: FieldRef[];
}

interface FieldReferenceGridProps {
  /** Two columns of field docs (e.g. Core Fields / Metadata Fields). */
  columns: [FieldReferenceColumn, FieldReferenceColumn];
}

/**
 * The "Field Reference" doc block shared by the agent and server detail modals:
 * a titled gray panel with two columns of `code` + description rows. Pass the
 * entity-specific field lists; the layout is shared.
 */
const FieldReferenceGrid: React.FC<FieldReferenceGridProps> = ({ columns }) => {
  return (
    <div className="bg-surface-muted border border-line rounded-lg p-4">
      <h4 className="font-medium text-fg mb-3">Field Reference</h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        {columns.map((col) => (
          <div key={col.heading}>
            <h5 className="font-medium text-fg-muted mb-2">
              {col.heading}
            </h5>
            <ul className="space-y-1 text-fg-muted">
              {col.fields.map((field) => (
                <li key={field.name}>
                  <code className="bg-surface-hover text-fg px-1 rounded">
                    {field.name}
                  </code>{' '}
                  - {field.description}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FieldReferenceGrid;
