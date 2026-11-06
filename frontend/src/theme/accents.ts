/**
 * Centralized accent token set for entity cards.
 *
 * Each entity type is assigned one accent token. The accent drives EVERY
 * shared card feature consistently — container, footer, divider, toggle, tags,
 * and the interactive "tool"/link color — so the same accent always looks the
 * same regardless of which card it's on. Card *identity* is the accent's color;
 * the *rule* (accent drives all features) is uniform across every type.
 *
 * This is the single source of truth for card color. The theming phase
 * (runtime CSS variables) replaces the Tailwind color literals below with
 * `var(--accent-*)` references without touching any card or primitive — the
 * token names and the per-feature structure stay exactly as they are here.
 */

export type AccentToken = 'primary' | 'cyan' | 'amber' | 'teal' | 'neutral';

/** Class fragments for every accent-driven surface of a card. */
export interface AccentClasses {
  /** Outer container: gradient background, border, hover border. */
  container: string;
  /** Footer bar: top border + background tint. */
  footer: string;
  /** Thin vertical divider between footer status items. */
  divider: string;
  /** Toggle switch background when enabled. */
  toggleOn: string;
  /** Tag pill background + text. */
  tag: string;
  /** Interactive accent for tool-count buttons, links, icon chips. */
  interactive: string;
  /** Soft background chip behind an icon (e.g. the tool-count icon). */
  iconChip: string;
}

export const ACCENTS: Record<AccentToken, AccentClasses> = {
  primary: {
    container:
      'bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 border-2 border-purple-200 dark:border-purple-700 hover:border-purple-300 dark:hover:border-purple-600',
    footer:
      'border-purple-100 dark:border-purple-800 bg-purple-50/50 dark:bg-purple-900/10',
    divider: 'bg-purple-200 dark:bg-purple-600',
    toggleOn: 'bg-purple-600',
    tag: 'bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300',
    interactive:
      'text-purple-600 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300 hover:bg-purple-50 dark:hover:bg-purple-900/20',
    iconChip: 'bg-purple-50 dark:bg-purple-900/30',
  },
  cyan: {
    container:
      'bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 border-2 border-cyan-200 dark:border-cyan-700 hover:border-cyan-300 dark:hover:border-cyan-600',
    footer:
      'border-cyan-100 dark:border-cyan-700 bg-cyan-50/50 dark:bg-cyan-900/30',
    divider: 'bg-cyan-200 dark:bg-cyan-600',
    toggleOn: 'bg-cyan-600',
    tag: 'bg-cyan-50 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-300',
    interactive:
      'text-cyan-600 hover:text-cyan-700 dark:text-cyan-400 dark:hover:text-cyan-300 hover:bg-cyan-50 dark:hover:bg-cyan-900/20',
    iconChip: 'bg-cyan-50 dark:bg-cyan-900/30',
  },
  amber: {
    container:
      'bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 border-2 border-amber-200 dark:border-amber-700 hover:border-amber-300 dark:hover:border-amber-600',
    footer:
      'border-amber-100 dark:border-amber-700 bg-amber-50/50 dark:bg-amber-900/30',
    divider: 'bg-amber-200 dark:bg-amber-600',
    toggleOn: 'bg-amber-600',
    tag: 'bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300',
    interactive:
      'text-amber-600 hover:text-amber-700 dark:text-amber-400 dark:hover:text-amber-300 hover:bg-amber-50 dark:hover:bg-amber-900/20',
    iconChip: 'bg-amber-50 dark:bg-amber-900/30',
  },
  teal: {
    container:
      'bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 border-2 border-teal-200 dark:border-teal-700 hover:border-teal-300 dark:hover:border-teal-600',
    footer:
      'border-teal-100 dark:border-teal-800 bg-teal-50/50 dark:bg-teal-900/10',
    divider: 'bg-teal-200 dark:bg-teal-600',
    toggleOn: 'bg-teal-600',
    tag: 'bg-teal-50 dark:bg-teal-900/30 text-teal-700 dark:text-teal-300',
    interactive:
      'text-teal-600 hover:text-teal-700 dark:text-teal-400 dark:hover:text-teal-300 hover:bg-teal-50 dark:hover:bg-teal-900/20',
    iconChip: 'bg-teal-50 dark:bg-teal-900/30',
  },
  neutral: {
    container:
      'bg-surface border border-line hover:border-line-strong',
    footer:
      'border-line bg-surface-muted/50',
    divider: 'bg-gray-200 dark:bg-gray-600',
    toggleOn: 'bg-blue-600',
    tag: 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300',
    interactive:
      'text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 hover:bg-blue-50 dark:hover:bg-blue-900/20',
    iconChip: 'bg-blue-50 dark:bg-blue-900/30',
  },
};

/**
 * Maps a registry entity type to its accent token, so every card of a given
 * type renders the same color-coded identity:
 *   server → primary (purple/indigo)   agent → cyan
 *   skill → amber                       virtualServer → teal
 *   custom → neutral
 *
 * Custom entity types share the neutral accent (they have no fixed brand color).
 */
export const ENTITY_ACCENTS = {
  server: 'primary',
  agent: 'cyan',
  skill: 'amber',
  virtualServer: 'teal',
  custom: 'neutral',
} as const satisfies Record<string, AccentToken>;
