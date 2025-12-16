import { useState, type ReactNode, type ReactElement, cloneElement } from 'react';
import { cn } from '../../lib/cn';

export type TooltipPosition = 'top' | 'bottom' | 'left' | 'right';

export interface TooltipProps {
  content: ReactNode;
  children: ReactElement;
  position?: TooltipPosition;
  delay?: number;
  className?: string;
}

const positionStyles: Record<TooltipPosition, string> = {
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
  left: 'right-full top-1/2 -translate-y-1/2 mr-2',
  right: 'left-full top-1/2 -translate-y-1/2 ml-2',
};

const arrowStyles: Record<TooltipPosition, string> = {
  top: 'top-full left-1/2 -translate-x-1/2 border-t-gray-900 dark:border-t-gray-700 border-x-transparent border-b-transparent',
  bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-gray-900 dark:border-b-gray-700 border-x-transparent border-t-transparent',
  left: 'left-full top-1/2 -translate-y-1/2 border-l-gray-900 dark:border-l-gray-700 border-y-transparent border-r-transparent',
  right: 'right-full top-1/2 -translate-y-1/2 border-r-gray-900 dark:border-r-gray-700 border-y-transparent border-l-transparent',
};

export function Tooltip({
  content,
  children,
  position = 'top',
  delay = 200,
  className,
}: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const [timeoutId, setTimeoutId] = useState<ReturnType<typeof setTimeout> | null>(null);

  const handleMouseEnter = () => {
    const id = setTimeout(() => setVisible(true), delay);
    setTimeoutId(id);
  };

  const handleMouseLeave = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
      setTimeoutId(null);
    }
    setVisible(false);
  };

  const handleFocus = () => {
    setVisible(true);
  };

  const handleBlur = () => {
    setVisible(false);
  };

  return (
    <div className="relative inline-flex">
      {cloneElement(children, {
        onMouseEnter: handleMouseEnter,
        onMouseLeave: handleMouseLeave,
        onFocus: handleFocus,
        onBlur: handleBlur,
        'aria-describedby': visible ? 'tooltip' : undefined,
      })}
      {visible && (
        <div
          id="tooltip"
          role="tooltip"
          className={cn(
            'absolute z-50 px-2 py-1',
            'text-xs font-medium text-white whitespace-nowrap',
            'bg-gray-900 dark:bg-gray-700 rounded shadow-lg',
            'animate-fade-in',
            positionStyles[position],
            className
          )}
        >
          {content}
          <span
            className={cn(
              'absolute w-0 h-0',
              'border-4',
              arrowStyles[position]
            )}
          />
        </div>
      )}
    </div>
  );
}
