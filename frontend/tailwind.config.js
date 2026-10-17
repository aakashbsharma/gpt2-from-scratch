/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#07080c',
          900: '#0b0d14',
          850: '#10131c',
          800: '#151926',
          700: '#1e2333',
          600: '#2a3047',
          500: '#3b4360',
        },
        accent: {
          400: '#5eead4',
          500: '#2dd4bf',
          600: '#14b8a6',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      keyframes: {
        sweep: {
          '0%': { transform: 'translateX(0%)', opacity: '0' },
          '10%': { opacity: '1' },
          '90%': { opacity: '1' },
          '100%': { transform: 'translateX(var(--sweep-end, 100%))', opacity: '0' },
        },
      },
      animation: {
        sweep: 'sweep var(--sweep-duration, 2.4s) linear infinite',
      },
    },
  },
  plugins: [],
};
