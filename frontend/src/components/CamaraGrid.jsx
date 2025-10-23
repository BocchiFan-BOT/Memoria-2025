// frontend/src/components/CamaraGrid.jsx
import React from "react";

export default function CamaraGrid({ cameras = [] }) {
  if (!Array.isArray(cameras) || cameras.length === 0) {
    return (
      <p className="muted" style={{ textAlign: "center", marginTop: "2rem" }}>
        No hay cámaras registradas.
      </p>
    );
  }

  return (
    <section className="camera-grid">
      {cameras.map((cam) => (
        <article key={cam.id} className="camera-card">
          <header
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "8px",
            }}
          >
            <strong>{cam.name || `Cámara ${cam.id}`}</strong>
          </header>

          {/* imagen del stream */}
          <div className="cam-media">
            <img
              src={`http://localhost:8000/stream/${cam.id}`}
              alt={cam.name}
              className="camera-img"
              onError={(e) => (e.currentTarget.style.opacity = 0.3)}
            />
          </div>

          {(cam.location || cam.coordinates) && (
            <footer
              className="muted"
              style={{ marginTop: "8px", fontSize: "0.8rem" }}
            >
              {cam.location}{" "}
              {cam.coordinates && (
                <span style={{ opacity: 0.8 }}>– {cam.coordinates}</span>
              )}
            </footer>
          )}
        </article>
      ))}
    </section>
  );
}
