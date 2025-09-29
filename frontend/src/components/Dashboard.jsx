import { useEffect, useState } from "react";
import { getCount, getHistory, getReport } from "../services/api";
import { LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";

export default function Dashboard() {
  const [count, setCount] = useState(0);
  const [history, setHistory] = useState([]);
  const [nowStr, setNowStr] = useState(() =>
    new Intl.DateTimeFormat("es-CL", {
      dateStyle: "full",
      timeStyle: "medium",
    }).format(new Date())
  );

  useEffect(() => {
    const clock = setInterval(() => {
      setNowStr(
        new Intl.DateTimeFormat("es-CL", {
          dateStyle: "full",
          timeStyle: "medium",
        }).format(new Date())
      );
    }, 1000);

    const i1 = setInterval(async () => {
      try {
        const res = await getCount();
        setCount(res.data.count);
      } catch {}
    }, 500);

    const i2 = setInterval(async () => {
      try {
        const res = await getHistory();
        const data = res.data.map((item) => ({
          time: new Date(item.t * 1000).toLocaleTimeString(),
          count: item.count,
        }));
        setHistory(data);
      } catch {}
    }, 2000);

    return () => {
      clearInterval(clock);
      clearInterval(i1);
      clearInterval(i2);
    };
  }, []);

  const downloadReport = async () => {
    const res = await getReport();
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const a = document.createElement("a");
    a.href = url;
    a.download = "reporte.csv";
    a.click();
  };

  return (
    <div className="dashboard">
      {/* Header con título y fecha/hora */}
      <div className="dash-header">
        <h2>Monitoreo de aglomeraciones</h2>
        <div className="dash-time" aria-live="polite">{nowStr}</div>
      </div>

      {/* Video centrado */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: "1.5rem" }}>
        <img
          src="http://localhost:8000/stream"
          alt="stream"
          style={{ width: "600px", border: "2px solid black", borderRadius: "8px" }}
        />
      </div>

      {/* Conteo actual */}
      <p style={{ textAlign: "center" }}>
        <strong>Conteo actual:</strong> {count} personas
      </p>

      {/* Gráfico dentro de una “card” */}
      <div className="chart-card">
        <LineChart width={600} height={300} data={history}>
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="count" stroke="#8884d8" />
        </LineChart>
      </div>

      {/* Botón de descarga*/}
      <button className="btn-download" onClick={downloadReport} aria-label="Descargar reporte CSV">
        <span className="btn-icon" aria-hidden>⬇</span>
        Descargar reporte CSV
      </button>
    </div>
  );
}