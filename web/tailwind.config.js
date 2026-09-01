/**
 * Palette sampled from the Figma export (912x673 flattened PNG) rather than guessed.
 *
 * The draft has no layers and therefore no variables, so every flat surface below was
 * read off the actual pixels. Two corrections that visual reading got wrong: the
 * sidebar is a neutral grey (#F3F3F3), not lavender, and the only lavender surfaces
 * are the composer's inner strip (#FAF7FE) and the orb.
 *
 * The violet hue is consistent at ~258 deg across every sample (orb #CCB3FC h261,
 * greeting text #9E93C3 h254, chip text #9786BF h258, logo mark #B1A3D4 h257).
 * Steps 50-400 are sampled. Steps 500-700 are DERIVED on that same hue: the export
 * has no flat region of them — its purple only ever appears as small antialiased
 * text — so they are extrapolated for contrast, not measured.
 */
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        violet: {
          50: '#FAF7FE', // sampled - composer inner strip
          100: '#F2EEFF', // sampled - light lavender surface
          200: '#E8DFFC', // sampled - orb mid
          300: '#D0BAFC', // sampled - orb
          400: '#CCB3FC', // sampled - orb core, most saturated flat violet
          500: '#A78BFA', // derived
          600: '#8B5CF6', // derived - interactive fill / text on white
          700: '#7C3AED', // derived - hover / pressed
        },
        surface: {
          DEFAULT: '#FEFEFE', // sampled - main card, 62% of the canvas
          sunken: '#F3F3F3', // sampled - sidebar
          raised: '#FFFFFF',
        },
        edge: {
          DEFAULT: '#EDEDED', // sampled - hairline borders
          strong: '#E2E2E2',
        },
        ink: {
          DEFAULT: '#111111', // sampled - buttons and headings
          muted: '#6B6B6B', // derived - secondary text
          faint: '#9A9A9A', // derived - section labels
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
