/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        parchment: {
          50: '#faf7f0',
          100: '#f5efe0',
          200: '#ebdcc0',
          300: '#dfc79b',
          400: '#d1ad71',
          500: '#c4954d',
          800: '#4a3821',
          900: '#2c2113',
        },
        sepia: {
          border: '#8c6d46',
          ink: '#38281a',
          accent: '#b24c28',
          gold: '#c99738',
        }
      },
      fontFamily: {
        serif: ['Georgia', '"Times New Roman"', 'serif'],
        sans: ['system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
        mono: ['Consolas', '"Courier New"', 'monospace'],
      }
    },
  },
  plugins: [],
}
