import React from 'react';

/**
 * An electrical-plug icon (two-prong plug), drawn inline because Heroicons —
 * the project's icon set — has no plug/socket glyph. Matches the Heroicons
 * 24x24 outline style (currentColor stroke, width 1.5) so it sits cleanly
 * alongside the other card action icons. Accepts a `className` for sizing/color
 * exactly like a Heroicons component.
 */
function PlugIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      strokeWidth={1.5}
      stroke="currentColor"
      className={className}
      aria-hidden="true"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 22v-5M9 8V2M15 8V2M18 8v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V8h12Z"
      />
    </svg>
  );
}

export default PlugIcon;
