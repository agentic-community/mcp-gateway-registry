/**
 * Centralized accent token set for entity cards.
 *
 * Each entity type is assigned one accent token. The accent drives EVERY
 * shared card feature consistently — container, footer, divider, toggle, tags,
 * and the interactive "tool"/link color — so the same accent always looks the
 * same regardless of which card it's on. Card *identity* is the accent's color;
 * the *rule* (accent drives all features) is uniform across every type.
 *
 * This is the single source of truth for card color. Each accent uses its
 * entity ramp (`server-*`, `agent-*`, `skill-*`, `virtual-*`, `custom-*`),
 * which resolves to CSS variables in index.css, so a branding theme.css can
 * recolor every card of a type without touching any card or primitive. The
 * token names are historical and stay as they are; they no longer name hues.
 */

export type AccentToken = 'primary' | 'cyan' | 'amber' | 'teal' | 'neutral' | 'custom';

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
      'bg-gradient-to-br from-server-50 to-server-50 dark:from-server-900/20 dark:to-server-900/20 border-2 border-server-200 dark:border-server-700 hover:border-server-300 dark:hover:border-server-600',
    footer:
      'border-server-100 dark:border-server-800 bg-server-50/50 dark:bg-server-900/10',
    divider: 'bg-server-200 dark:bg-server-600',
    toggleOn: 'bg-server-600',
    tag: 'bg-server-50 dark:bg-server-900/30 text-server-700 dark:text-server-300',
    interactive:
      'text-server-600 hover:text-server-700 dark:text-server-400 dark:hover:text-server-300 hover:bg-server-50 dark:hover:bg-server-900/20',
    iconChip: 'bg-server-50 dark:bg-server-900/30',
  },
  cyan: {
    container:
      'bg-gradient-to-br from-agent-50 to-agent-50 dark:from-agent-900/20 dark:to-agent-900/20 border-2 border-agent-200 dark:border-agent-700 hover:border-agent-300 dark:hover:border-agent-600',
    footer:
      'border-agent-100 dark:border-agent-700 bg-agent-50/50 dark:bg-agent-900/30',
    divider: 'bg-agent-200 dark:bg-agent-600',
    toggleOn: 'bg-agent-600',
    tag: 'bg-agent-50 dark:bg-agent-900/30 text-agent-700 dark:text-agent-300',
    interactive:
      'text-agent-600 hover:text-agent-700 dark:text-agent-400 dark:hover:text-agent-300 hover:bg-agent-50 dark:hover:bg-agent-900/20',
    iconChip: 'bg-agent-50 dark:bg-agent-900/30',
  },
  amber: {
    container:
      'bg-gradient-to-br from-skill-50 to-skill-50 dark:from-skill-900/20 dark:to-skill-900/20 border-2 border-skill-200 dark:border-skill-700 hover:border-skill-300 dark:hover:border-skill-600',
    footer:
      'border-skill-100 dark:border-skill-700 bg-skill-50/50 dark:bg-skill-900/30',
    divider: 'bg-skill-200 dark:bg-skill-600',
    toggleOn: 'bg-skill-600',
    tag: 'bg-skill-50 dark:bg-skill-900/30 text-skill-700 dark:text-skill-300',
    interactive:
      'text-skill-600 hover:text-skill-700 dark:text-skill-400 dark:hover:text-skill-300 hover:bg-skill-50 dark:hover:bg-skill-900/20',
    iconChip: 'bg-skill-50 dark:bg-skill-900/30',
  },
  teal: {
    container:
      'bg-gradient-to-br from-virtual-50 to-virtual-50 dark:from-virtual-900/20 dark:to-virtual-900/20 border-2 border-virtual-200 dark:border-virtual-700 hover:border-virtual-300 dark:hover:border-virtual-600',
    footer:
      'border-virtual-100 dark:border-virtual-800 bg-virtual-50/50 dark:bg-virtual-900/10',
    divider: 'bg-virtual-200 dark:bg-virtual-600',
    toggleOn: 'bg-virtual-600',
    tag: 'bg-virtual-50 dark:bg-virtual-900/30 text-virtual-700 dark:text-virtual-300',
    interactive:
      'text-virtual-600 hover:text-virtual-700 dark:text-virtual-400 dark:hover:text-virtual-300 hover:bg-virtual-50 dark:hover:bg-virtual-900/20',
    iconChip: 'bg-virtual-50 dark:bg-virtual-900/30',
  },
  // Plain cards with no entity identity (ordinary MCP servers, and the default
  // for card primitives). Its interactive color follows the info ramp.
  neutral: {
    container:
      'bg-surface border border-line hover:border-line-strong',
    footer:
      'border-line bg-surface-muted/50',
    divider: 'bg-gray-200 dark:bg-gray-600',
    toggleOn: 'bg-info-600',
    tag: 'bg-info-50 dark:bg-info-900/30 text-info-700 dark:text-info-300',
    interactive:
      'text-info-600 hover:text-info-700 dark:text-info-400 dark:hover:text-info-300 hover:bg-info-50 dark:hover:bg-info-900/20',
    iconChip: 'bg-info-50 dark:bg-info-900/30',
  },
  // Custom entity types: neutral surface, interactive color from the custom ramp.
  custom: {
    container:
      'bg-surface border border-line hover:border-line-strong',
    footer:
      'border-line bg-surface-muted/50',
    divider: 'bg-gray-200 dark:bg-gray-600',
    toggleOn: 'bg-custom-600',
    tag: 'bg-custom-50 dark:bg-custom-900/30 text-custom-700 dark:text-custom-300',
    interactive:
      'text-custom-600 hover:text-custom-700 dark:text-custom-400 dark:hover:text-custom-300 hover:bg-custom-50 dark:hover:bg-custom-900/20',
    iconChip: 'bg-custom-50 dark:bg-custom-900/30',
  },
};

/**
 * Maps a registry entity type to its accent token, so every card of a given
 * type renders the same color-coded identity:
 *   server → primary (purple/indigo)   agent → cyan
 *   skill → amber                       virtualServer → teal
 *   custom → custom
 *
 * Ordinary (non-Anthropic) server cards use the neutral accent; see ServerCard.
 */
export const ENTITY_ACCENTS = {
  server: 'primary',
  agent: 'cyan',
  skill: 'amber',
  virtualServer: 'teal',
  custom: 'custom',
} as const satisfies Record<string, AccentToken>;
