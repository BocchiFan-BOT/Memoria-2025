// frontend/src/components/DashboardGrid.jsx
import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import Chart from "chart.js/auto";
import CrowdBar from "./CrowdBar"; // <-- agrega esto

export default function DashboardGrid({ cameras }) {
  const [histories, setHistories] = useState({});

  useEffect(() => {
    if (!cameras || cameras.length === 0) return;
    let active = true;

    const fetchAll = () => {
      cameras.forEach((cam) => {
        fetch(`http://localhost:8000/history/${cam.id}`)
          .then((r) => r.json())
          .then((data) => { if (active) setHistories((h) => ({ ...h, [cam.id]: Array.isArray(data) ? data : [] })); })
          .catch(() => {});
      });
    };

    fetchAll();
    const timer = setInterval(fetchAll, 5000); // <-- opcional
    return () => { active = false; clearInterval(timer); };
  }, [cameras]);

  if (!cameras || cameras.length === 0)
    return <div className="empty">No hay dashboards (agrega cámaras primero)</div>;

  return (
    <div className="dash-grid">
      {cameras.map((cam) => {
        const h = histories[cam.id] || [];
        const labels = h.map((x) => new Date(x.t * 1000).toLocaleTimeString());
        const counts = h.map((x) => x.count);
        const lastCrowd = h.length ? Number(h[h.length - 1].crowd_index || 0) : 0; // <-- usa crowd_index

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
          plugins: { legend: { display: false }, tooltip: { intersect: false, mode: "index" } },
        };

        return (
          <div key={cam.id} className="dash-card" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <h3 style={{ margin: 0 }}>{cam.name}</h3>

            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <div style={{ width: "100%", height: 240 }}>
                <Line data={data} options={options} />
              </div>

              {/* Pila por cámara */}
              <CrowdBar value={lastCrowd} height={220} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
