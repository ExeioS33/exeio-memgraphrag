/**
 * Every colour resolves through a CSS variable defined in src/index.css, where the
 * light and dark token sets live. The `<alpha-value>` slot is what keeps utilities
 * like `bg-surface-raised/95` and `ring-violet-400` working: a bare `var()` has no
 * way to take an opacity modifier.
 *
 * The light palette was sampled from the original Figma export (912x673 flattened
 * PNG): the sidebar is a neutral grey (#F3F3F3), not lavender; the only lavender
 * surfaces are the composer's inner strip (#FAF7FE) and the orb; the violet hue is
 * consistent at ~258 deg. Steps 500-700 are derived on that hue for contrast. The
 * dark palette was not inverted from it — each token was chosen against its surface
 * for a measured contrast ratio; scripts/check_contrast.py holds the thresholds.
 */

const token = (name) => `rgb(var(--c-${name}) / <alpha-value>)`

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        violet: {
          50: token('violet-50'),
          100: token('violet-100'),
          200: token('violet-200'),
          300: token('violet-300'),
          400: token('violet-400'),
          500: token('violet-500'),
          600: token('violet-600'),
          700: token('violet-700'),
        },
        surface: {
          DEFAULT: token('surface'),
          sunken: token('surface-sunken'),
          raised: token('surface-raised'),
        },
        edge: {
          DEFAULT: token('edge'),
          strong: token('edge-strong'),
        },
        ink: {
          DEFAULT: token('ink'),
          muted: token('ink-muted'),
          faint: token('ink-faint'),
          // Text on an `ink`-coloured fill: white in light, near-black in dark. Not
          // the same thing as white on a violet fill, which stays white in both.
          inverse: token('ink-inverse'),
        },
      },
      borderRadius: {
        card: '18px',
        panel: '22px',
      },
      fontFamily: {
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'sans-serif',
        ],
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 160ms ease-out',
      },
    },
  },
  plugins: [],
}
