import { useEffect, useState } from "react";
import { getCount, getHistory, getReport } from "../services/api";
import { LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";

export default function Dashboard() {
  const [count, setCount] = useState(0);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    // actualizar conteo cada 0.5s
    const i1 = setInterval(async () => {
      try {
        const res = await getCount();
        setCount(res.data.count);
      } catch {}
    }, 500);

    // actualizar gráfico cada 2s
    const i2 = setInterval(async () => {
      try {
        const res = await getHistory();
        const data = res.data.map(item => ({
          time: new Date(item.t * 1000).toLocaleTimeString(),
          count: item.count
        }));
        setHistory(data);
      } catch {}
    }, 2000);

    return () => {
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
      <h2>Monitor de Personas</h2>
      <div>
        <img
          src="http://localhost:8000/stream"
          alt="stream"
          style={{ width: "600px", border: "2px solid black" }}
        />
        <p><strong>Conteo actual:</strong> {count} personas</p>
      </div>

      <LineChart width={600} height={300} data={history}>
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="count" stroke="#8884d8" />
      </LineChart>

      <button onClick={downloadReport}>📄 Descargar reporte CSV</button>
    </div>
  );
}

