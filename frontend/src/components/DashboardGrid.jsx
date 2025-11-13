// frontend/src/components/DashboardGrid.jsx
import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import Chart from "chart.js/auto";
import CrowdBar from "./CrowdBar";

const handleDownloadReport = async (camId, camName) => {
  try {
    const resp = await fetch(`http://localhost:8000/report/${camId}`);
    if (!resp.ok) {
      console.error("Error al descargar reporte", resp.status);
      return;
    }

    const blob = await resp.blob();
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    const safeName = (camName || camId || "camara").replace(/[^a-zA-Z0-9_-]/g, "_");
    a.download = `reporte_${safeName}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    console.error("Error descargando reporte:", err);
  }
};

export default function DashboardGrid({ cameras = [] }) {
  const [histories, setHistories] = useState({});

  useEffect(() => {
    if (!cameras || cameras.length === 0) return;
    let active = true;

    const fetchAll = () => {
      cameras.forEach((cam) => {
        fetch(`http://localhost:8000/history/${cam.id}`)
          .then((r) => r.json())
          .then((data) => {
            if (!active) return;
            setHistories((prev) => ({
              ...prev,
              [cam.id]: Array.isArray(data) ? data : [],
            }));
          })
          .catch(() => {});
      });
    };

    fetchAll();
    const timer = setInterval(fetchAll, 5000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [cameras]);

  if (!cameras || cameras.length === 0) {
    return (
      <p className="muted" style={{ textAlign: "center", marginTop: "2rem" }}>
        No hay dashboards (agrega cámaras primero)
      </p>
    );
  }

  return (
    <section className="dash-grid">
      {cameras.map((cam) => {
        const h = histories[cam.id] || [];
        const labels = h.map((x) =>
          x.t ? new Date(x.t * 1000).toLocaleTimeString() : "-"
        );
        const counts = h.map((x) => x.count ?? 0);
        const lastCrowd = h.length ? Number(h[h.length - 1].crowd_index || 0) : 0;

        const data = {
          labels,
          datasets: [
            {
              label: "Personas",
              data: counts,
              borderColor: "#5b7cfa",
              backgroundColor: "rgba(91,124,250,0.15)",
              fill: true,
              tension: 0.25,
              pointRadius: 0,
            },
          ],
        };

        const options = {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { ticks: { maxRotation: 0, autoSkip: true } },
            y: { beginAtZero: true, precision: 0 },
          },
          plugins: {
            legend: { display: false },
            tooltip: { intersect: false, mode: "index" },
          },
        };

        return (
          <article key={cam.id} className="dash-card">
            <header className="dash-card-header">
              <h3>{cam.name || `Cámara ${cam.id}`}</h3>
              <span className="dash-card-sub">ID: {cam.id}</span>
            </header>

            <div className="dash-card-content">
              <div className="chart-wrapper">
                <Line data={data} options={options} />
              </div>
              <div className="pile-wrapper">
                <CrowdBar value={lastCrowd} height={180} />
              </div>
            </div>

            <div className="dash-card-footer">
              <button
                type="button"
                className="btn-report-small"
                onClick={() => handleDownloadReport(cam.id, cam.name)}
              >
                Descargar reporte
              </button>
            </div>
          </article>
        );
      })}
    </section>
  );
}
