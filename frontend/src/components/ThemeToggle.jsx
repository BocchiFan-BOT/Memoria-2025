// frontend/src/components/ThemeToggle.jsx
export default function ThemeToggle({ theme, onToggle }) {
  const isLight = theme === "light";
  const label = isLight ? "Cambiar a modo oscuro" : "Cambiar a modo claro";

  return (
    <button
      className="theme-toggle"
      onClick={onToggle}
      aria-pressed={isLight}
      aria-label={label}
      title={label}
    >
      {/*iconos para ver el modo*/}
      {isLight ? (
        // luna
        <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M21 12.3A8.7 8.7 0 1 1 11.7 3 7 7 0 0 0 21 12.3Z"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ) : (
        // sol
        <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" strokeWidth="1.8"/>
          <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M18.4 5.6l1.4-1.4M4.2 19.8l1.4-1.4"
            fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
        </svg>
      )}
      <span className="theme-toggle-text">{isLight ? "Modo claro" : "Modo oscuro"}</span>
    </button>
  );
}
