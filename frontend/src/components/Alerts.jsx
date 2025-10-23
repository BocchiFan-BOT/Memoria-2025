// frontend/src/components/Alerts.jsx
import React, { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";
import { getAlerts } from "../services/api";

// Toast bonito, empresarial, tipo asistente
function Toast({ alert, onClose }) {
  return (
    <div className="toast-card toast--warning" role="status" aria-live="polite">
      <div className="toast-icon" aria-hidden>
        <span className="toast-icon-shape">!</span>
      </div>
      <div className="toast-content">
        <div className="toast-title">Aglomeración detectada</div>
        <div className="toast-body">
          <div className="toast-line"><strong>{alert.cam_name}</strong></div>
          <div className="toast-line">
            Personas: <strong>{alert.count}</strong>
            {typeof alert.occupancy === "number" && (
              <>
                {" "}| Ocupación: <strong>{alert.occupancy}%</strong>
              </>
            )}
          </div>
          <div className="toast-time">{new Date(alert.t * 1000).toLocaleTimeString()}</div>
        </div>
      </div>
      <button className="toast-close" onClick={onClose} aria-label="Cerrar alerta">×</button>
    </div>
  );
}

export default function Alerts() {
  const [toasts, setToasts] = useState([]);
  const lastTsRef = useRef(undefined);
  const seenRef = useRef(new Set());
  const intervalRef = useRef(null);
  const delayRef = useRef(2000); // backoff 2s -> 4s -> 8s -> 15s
  const hiddenRef = useRef(document.hidden);

  useEffect(() => {
    let cancelled = false;

    const schedule = (ms) => {
      if (intervalRef.current) clearTimeout(intervalRef.current);
      intervalRef.current = setTimeout(tick, ms);
    };

    const onVisibility = () => {
      hiddenRef.current = document.hidden;
      if (!document.hidden) {
        delayRef.current = 2000;
        schedule(250);
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    const tick = async () => {
      try {
        if (cancelled) return;
        if (hiddenRef.current) return schedule(5000);

        const since = lastTsRef.current;
        const res = await getAlerts(since);
        const arr = Array.isArray(res.data) ? res.data : [];
        if (arr.length > 0) {
          const news = [];
          for (const a of arr) {
            const id = `${a.cam_id}-${a.t}`;
            if (!seenRef.current.has(id)) {
              seenRef.current.add(id);
              news.push(a);
            }
          }
          if (news.length > 0) {
            setToasts((prev) => [...prev, ...news]);
            lastTsRef.current = Math.max(
              ...(arr.map((x) => x.t)),
              lastTsRef.current || 0
            );
            delayRef.current = 2000; // reset backoff
            return schedule(delayRef.current);
          }
        }
        // sin novedades -> backoff
        delayRef.current = Math.min(delayRef.current * 2, 15000);
        schedule(delayRef.current);
      } catch {
        delayRef.current = Math.min(delayRef.current * 2, 15000);
        schedule(delayRef.current);
      }
    };

    schedule(0);
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVisibility);
      if (intervalRef.current) clearTimeout(intervalRef.current);
    };
  }, []);

  const closeToast = (idx) => {
    setToasts((prev) => prev.filter((_, i) => i !== idx));
  };

  const body = (
    <div className="toast-area" aria-live="polite" aria-relevant="additions removals">
      {toasts.map((t, i) => (
        <Toast key={`${t.cam_id}-${t.t}`} alert={t} onClose={() => closeToast(i)} />
      ))}
    </div>
  );
  return ReactDOM.createPortal(body, document.body);
}
