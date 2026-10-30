/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        panel: '#08131f',
        glow: '#2dffb6',
        accent: '#7b61ff',
        warning: '#ffb443',
      },
      boxShadow: {
        neon: '0 0 30px rgba(45, 255, 182, 0.2)',
      },
    },
  },
  plugins: [],
};
