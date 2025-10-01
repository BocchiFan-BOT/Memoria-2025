// frontend/src/components/DashboardGrid.jsx
import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import Chart from 'chart.js/auto';

export default function DashboardGrid({ cameras }) {
  // Para cada cámara pedimos /history/{id} al backend
  const [histories, setHistories] = useState({});

  useEffect(() => {
    cameras.forEach(cam => {
      fetch(`http://localhost:8000/history/${cam.id}`)
        .then(r => r.json())
        .then(data => setHistories(h => ({ ...h, [cam.id]: data })))
        .catch(()=>{});
    });
  }, [cameras]);

  if (!cameras || cameras.length === 0) return <div className="empty">No hay dashboards (agrega cámaras primero)</div>;

  return (
    <div className="dash-grid">
      {cameras.map(cam => {
        const h = histories[cam.id] || [];
        const labels = h.map(x => new Date(x.t*1000).toLocaleTimeString());
        const data = h.map(x => x.count);
        return (
          <div key={cam.id} className="dash-card">
            <h3>{cam.name}</h3>
            <Line data={{
              labels: labels,
              datasets: [{ label: "Personas", data: data, fill: false }]
            }} />
          </div>
        );
      })}
    </div>
  );
}

