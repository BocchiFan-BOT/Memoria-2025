// frontend/src/components/CrowdBar.jsx
import React from "react";

export default function CrowdBar({ value = 0, height = 300 }) {
  const v = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
  const pct = v * 100;
  const color =
    pct < 50 ? "rgba(46, 204, 113, 0.9)" : pct < 75 ? "rgba(241, 196, 15, 0.9)" : "rgba(231, 76, 60, 0.9)";

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
        role="progressbar"
        aria-label="Índice de aglomeración"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pct)}
        aria-valuetext={`${v.toFixed(2)}`}
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
        <strong>{v.toFixed(2)}</strong>
        <div style={{ opacity: 0.75 }}>Aglomeración</div>
      </div>
    </div>
  );
}
