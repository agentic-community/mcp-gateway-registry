import React from 'react';
import { ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';

interface CollapsibleSectionProps {
  /** Header label (left side of the toggle row). */
  title: React.ReactNode;
  /** Whether the body is expanded. */
  expanded: boolean;
  /** Toggle handler. */
  onToggle: () => void;
  /** Right-aligned badge/count shown in the header. */
  badge?: React.ReactNode;
  /** When false, the chevron is hidden and the header is not interactive. */
  collapsible?: boolean;
  children?: React.ReactNode;
}

/**
 * A bordered, header-toggled collapsible block — the backend/tools tree pattern
 * shared by the virtual-server details modal and the card tools modals. The
 * caller owns the expanded state (so groups can auto-expand the first item).
 */
const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  expanded,
  onToggle,
  badge,
  collapsible = true,
  children,
}) => {
  return (
    <div className="border border-line rounded-lg overflow-hidden">
      <button
        onClick={collapsible ? onToggle : undefined}
        disabled={!collapsible}
        className="w-full flex items-center justify-between px-4 py-3 bg-surface-muted hover:bg-surface-hover transition-colors text-left disabled:cursor-default disabled:hover:bg-surface-muted"
      >
        <div className="flex items-center gap-2 min-w-0">
          {collapsible ? (
            expanded ? (
              <ChevronDownIcon className="h-4 w-4 text-fg-subtle flex-shrink-0" />
            ) : (
              <ChevronRightIcon className="h-4 w-4 text-fg-subtle flex-shrink-0" />
            )
          ) : (
            <div className="w-4 flex-shrink-0" />
          )}
          {title}
        </div>
        {badge}
      </button>
      {expanded && children}
    </div>
  );
};

export default CollapsibleSection;
