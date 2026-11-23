/**
 * Every color below resolves to a CSS custom property declared in
 * src/index.css, so light/dark/high-contrast are runtime variable swaps.
 * `<alpha-value>` keeps opacity modifiers such as `bg-gray-900/50` working.
 */
const channel = (name) => `rgb(var(--${name}) / <alpha-value>)`;

const ramp = (name, steps) =>
  Object.fromEntries(steps.map((step) => [step, channel(`${name}-${step}`)]));

const STEPS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];

const brand = ramp('primary', STEPS);

/**
 * A ramp plus optional role keys under one color name, e.g. `bg-danger-50`
 * next to `bg-danger-soft`.
 */
const rampWith = (name, roles = {}) => ({ ...ramp(name, STEPS), ...roles });

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: brand,
        // `purple` is the brand in ~500 existing class names; aliasing it to the
        // brand ramp keeps every one of them on the logo's violet.
        purple: brand,
        gray: ramp('gray', [25, ...STEPS]),
        // Semantic, mode-aware roles (see index.css). Prefer these in new code:
        // they need no `dark:` twin.
        canvas: channel('color-canvas'),
        surface: {
          DEFAULT: channel('color-surface'),
          muted: channel('color-surface-muted'),
          hover: channel('color-surface-hover'),
          overlay: channel('color-surface-overlay'),
        },
        line: {
          DEFAULT: channel('color-line'),
          strong: channel('color-line-strong'),
        },
        fg: {
          DEFAULT: channel('color-fg'),
          muted: channel('color-fg-muted'),
          subtle: channel('color-fg-subtle'),
        },
        accent: {
          DEFAULT: channel('color-accent'),
          hover: channel('color-accent-hover'),
          fg: channel('color-accent-fg'),
          text: channel('color-accent-text'),
          soft: channel('color-accent-soft'),
          'soft-fg': channel('color-accent-soft-fg'),
        },
        danger: rampWith('danger', {
          DEFAULT: channel('color-danger'),
          hover: channel('color-danger-hover'),
          soft: channel('color-danger-soft'),
          'soft-fg': channel('color-danger-soft-fg'),
        }),
        // Status and entity ramps (see index.css). Use these instead of raw
        // Tailwind hues so a branding theme can recolor them; the ESLint
        // no-restricted-syntax rule in package.json rejects raw hue classes.
        success: ramp('success', STEPS),
        warning: ramp('warning', STEPS),
        info: ramp('info', STEPS),
        server: ramp('server', STEPS),
        agent: ramp('agent', STEPS),
        skill: ramp('skill', STEPS),
        virtual: ramp('virtual', STEPS),
        custom: ramp('custom', STEPS),
        focus: channel('color-focus'),
      },
      fontFamily: {
        sans: ['var(--font-sans)'],
        mono: ['var(--font-mono)'],
      },
      borderRadius: {
        control: 'var(--radius-control)',
        card: 'var(--radius-card)',
      },
      boxShadow: {
        sm: '0 1px 2px 0 rgb(var(--shadow-color) / 0.06), 0 1px 3px 0 rgb(var(--shadow-color) / 0.08)',
        DEFAULT: '0 1px 3px 0 rgb(var(--shadow-color) / 0.10), 0 1px 2px -1px rgb(var(--shadow-color) / 0.10)',
        md: '0 4px 8px -2px rgb(var(--shadow-color) / 0.10), 0 2px 4px -2px rgb(var(--shadow-color) / 0.06)',
        lg: '0 12px 16px -4px rgb(var(--shadow-color) / 0.10), 0 4px 6px -2px rgb(var(--shadow-color) / 0.05)',
        xl: '0 20px 24px -4px rgb(var(--shadow-color) / 0.10), 0 8px 8px -4px rgb(var(--shadow-color) / 0.04)',
      },
      ringColor: {
        DEFAULT: channel('color-focus'),
      },
      animation: {
        'fade-in': 'fadeIn 0.2s ease-in-out',
        'slide-up': 'slideUp 0.2s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}
