import React from 'react';

import Badge from './Badge';
import { ToolAnnotations } from '../types/toolAnnotations';

interface ToolAnnotationBadgesProps {
  annotations?: ToolAnnotations | null;
}

const READ_ONLY_TOOLTIP =
  'The server declares this tool read-only. The registry does not verify this.';
const DESTRUCTIVE_TOOLTIP =
  'The server declares this tool can delete or overwrite data. The registry does not verify this.';


/**
 * READ ONLY and DESTRUCTIVE badges for a tool, shown only for hints the server
 * declared. The spec's defaults (destructive unless told otherwise) are not
 * applied: the UI shows what the server said, not a guess. A tool that claims
 * both read-only and destructive shows only READ ONLY, since the spec gives
 * destructiveHint meaning only for tools that are not read-only.
 */
const ToolAnnotationBadges: React.FC<ToolAnnotationBadgesProps> = ({ annotations }) => {
  if (!annotations) {
    return null;
  }
  const readOnly = annotations.readOnlyHint === true;
  const destructive = annotations.destructiveHint === true && !readOnly;

  return (
    <>
      {readOnly && (
        <Badge tone="green" shape="square" className="no-underline" title={READ_ONLY_TOOLTIP}>
          READ ONLY
        </Badge>
      )}
      {destructive && (
        <Badge tone="red" shape="square" className="no-underline" title={DESTRUCTIVE_TOOLTIP}>
          DESTRUCTIVE
        </Badge>
      )}
    </>
  );
};

export default ToolAnnotationBadges;
