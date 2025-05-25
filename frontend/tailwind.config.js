/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: 'var(--primary-color)',
        'primary-light': 'var(--primary-light)',
        'primary-dark': 'var(--primary-dark)',
        accent: 'var(--accent-yellow)',
        olive: 'var(--accent-olive)',
        background: 'var(--background-color)',
        card: 'var(--card-background)',
        text: 'var(--text-color)',
        'text-secondary': 'var(--text-secondary)',
        border: 'var(--border-color)',
      },
      borderRadius: {
        lg: '1.25rem',
        md: '1rem',
        sm: '0.5rem',
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [],
}