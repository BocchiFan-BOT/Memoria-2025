export default function CrowdBar({ value = 0, height = 300 }) {
  // Escalar para que 0.5 sea el máximo visible
  const raw = Number.isFinite(value) ? value : 0;
  const v = Math.max(0, Math.min(0.5, raw));   // valor mostrado, máximo permitido
  const pct = (raw / 0.5) * 100;               // escala visual: 0.5 = 100%

  const color =
    pct < 50 ? "rgba(46, 204, 113, 0.9)"
    : pct < 75 ? "rgba(241, 196, 15, 0.9)"
    : "rgba(231, 76, 60, 0.9)";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", minWidth: 64 }}>
      <div
        style={{
          height,
          width: 28,
          borderRadius: 8,
          border: "1px solid #ddd",
          background: "linear-gradient(180deg,#f0f0f0 0%,#fafafa 100%)",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            height: `${pct}%`,
            background: color,
            transition: "height 300ms ease",
          }}
        />
      </div>
      <div style={{ marginTop: 6, fontSize: 12, textAlign: "center" }}>
        <strong>{raw.toFixed(2)}</strong>
        <div style={{ opacity: 0.75 }}>Aglomeración</div>
      </div>
    </div>
  );
}
