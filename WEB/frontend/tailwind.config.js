/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
  extend: {
    colors: {
      primary: {
        DEFAULT: "#2563eb", // blue-600
        dark: "#1d4ed8",    // blue-700
        light: "#3b82f6",   // blue-500
      },
    },
  },
},
  plugins: [],
}
