import React from 'react';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}

const Pagination: React.FC<PaginationProps> = ({
  currentPage,
  totalPages,
  totalItems,
  pageSize,
  onPageChange,
}) => {
  if (totalPages <= 1) return null;

  const start = currentPage * pageSize + 1;
  const end = Math.min((currentPage + 1) * pageSize, totalItems);

  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="text-fg-subtle">
        {start}-{end} of {totalItems}
      </span>
      <div className="flex items-center gap-1">
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 0}
          className="px-2 py-1 rounded border border-line-strong text-fg-muted disabled:opacity-30 disabled:cursor-not-allowed hover:bg-surface-hover hover:text-fg transition-colors text-xs"
        >
          Prev
        </button>
        <span className="text-fg-muted px-1">
          {currentPage + 1}/{totalPages}
        </span>
        <button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage >= totalPages - 1}
          className="px-2 py-1 rounded border border-line-strong text-fg-muted disabled:opacity-30 disabled:cursor-not-allowed hover:bg-surface-hover hover:text-fg transition-colors text-xs"
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default Pagination;
