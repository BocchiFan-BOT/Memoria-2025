// frontend/src/components/CamaraGrid.jsx
import React, { useEffect, useState } from "react";
import { getMetrics } from "../services/api";

function CameraCard({ cam, streamFps = 8 }) {
  const [metrics, setMetrics] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let alive = true;
    let timer;
    const poll = async () => {
      try {
        const { data } = await getMetrics(cam.id);
        if (!alive) return;
        setMetrics(data || null);
        setErr(null);
      } catch (e) {
        if (!alive) return;
        setErr("sin métricas");
      } finally {
        if (alive) timer = setTimeout(poll, 2000);
      }
    };
    poll();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [cam.id]);

  return (
    <article className="camera-card">
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
          src={`http://localhost:8000/stream/${cam.id}?fps=${streamFps}`}
          alt={cam.name}
          className="camera-img"
          onError={(e) => (e.currentTarget.style.opacity = 0.3)}
        />
      </div>

      {/* métricas */}
      <div className="muted" style={{ marginTop: 6, fontSize: "0.8rem" }}>
        {metrics ? (
          <>
            <span>
              fps: <b>{(metrics.fps ?? 0).toFixed ? metrics.fps.toFixed(1) : metrics.fps}</b>
            </span>
            {" · "}
            <span>
              grab: <b>{(metrics.grab_ms ?? 0).toFixed ? metrics.grab_ms.toFixed(1) : metrics.grab_ms} ms</b>
            </span>
            {" · "}
            <span>
              infer: <b>{(metrics.infer_ms ?? 0).toFixed ? metrics.infer_ms.toFixed(1) : metrics.infer_ms} ms</b>
            </span>
            {" · "}
            <span>
              loop: <b>{(metrics.loop_ms ?? 0).toFixed ? metrics.loop_ms.toFixed(1) : metrics.loop_ms} ms</b>
            </span>
            {" · "}
            <span>
              dev: <b>{metrics.device}</b>
            </span>
          </>
        ) : err ? (
          <em>{err}</em>
        ) : (
          <em>cargando métricas…</em>
        )}
      </div>

      {(cam.location || cam.coordinates) && (
        <footer className="muted" style={{ marginTop: "8px", fontSize: "0.8rem" }}>
          {cam.location} {cam.coordinates && <span style={{ opacity: 0.8 }}>– {cam.coordinates}</span>}
        </footer>
      )}
    </article>
  );
}

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
        <CameraCard key={cam.id} cam={cam} streamFps={8} />
      ))}
    </section>
  );
}
