import { useEffect, useState } from 'react';

const STORAGE_KEY = 'ragsale-theme';

function getInitialTheme() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    // Storage can be unavailable in restricted browser sessions.
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

// Set before React's first render to avoid a flash of the wrong colors.
const initialTheme = getInitialTheme();
document.documentElement.dataset.theme = initialTheme;

export default function ThemeToggle() {
  const [theme, setTheme] = useState(initialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // The toggle still works if the browser cannot save the preference.
    }
  }, [theme]);

  return <button type="button" className="theme-toggle"
    aria-label="Dark mode" aria-pressed={theme === 'dark'}
    title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    onClick={() => setTheme(current => current === 'light' ? 'dark' : 'light')}>
    <span aria-hidden="true">{theme === 'dark' ? '☀' : '☾'}</span>
    {theme === 'dark' ? 'Light mode' : 'Dark mode'}
  </button>;
}
