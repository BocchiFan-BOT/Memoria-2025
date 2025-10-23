// frontend/src/App.jsx
import React, { useState, useEffect } from "react";
import Login from "./components/Login";
import CameraManager from "./components/CamaraManager";
import CameraGrid from "./components/CamaraGrid";
import DashboardGrid from "./components/DashboardGrid";
import Alerts from "./components/Alerts";
import "./index.css";

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [cameras, setCameras] = useState([]);
  const [tab, setTab] = useState("dashboards");

  useEffect(() => {
    fetch("http://localhost:8000/cameras")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setCameras(data);
      })
      .catch(() => {
        console.warn("No se pudo obtener la lista de cámaras al iniciar");
      });
  }, []);

  const handleLoginSuccess = () => setIsAuthenticated(true);

  if (!isAuthenticated) {
    return <Login onSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="app-root">
      <Alerts />

      {/* === Sidebar izquierda (icono + texto) === */}
      <aside className="sidebar wide">

        <nav className="side-nav">
          <button
            className={`nav-row ${tab === "monitoreo" ? "active" : ""}`}
            onClick={() => setTab("monitoreo")}
            title="Monitoreo Cámaras"
          >
            {/* Icono: cámara */}
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
              <rect
                x="3"
                y="6"
                width="14"
                height="12"
                rx="2"
                ry="2"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              />
              <path
                d="M21 9l-4 2v2l4 2V9z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinejoin="round"
              />
            </svg>
            <span>Monitoreo Cámaras</span>
          </button>
          
          <button
            className={`nav-row ${tab === "dashboards" ? "active" : ""}`}
            onClick={() => setTab("dashboards")}
            title="Dashboard"
          >
            {/* Icono: reloj/actividad */}
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
              <circle
                cx="12"
                cy="12"
                r="9"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              />
              <path
                d="M12 7v5l3 2"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
            <span>Dashboard</span>
          </button>


          <button
            className={`nav-row ${tab === "manager" ? "active" : ""}`}
            onClick={() => setTab("manager")}
            title="Gestionar Cámaras"
          >
            {/* Icono: plus cuadrado */}
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
              <rect
                x="3"
                y="3"
                width="18"
                height="18"
                rx="3"
                ry="3"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              />
              <path
                d="M12 8v8M8 12h8"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
            <span>Gestionar</span>
          </button>
        </nav>
      </aside>

      {/* === Contenido === */}
      <div className="content">
        {/* Topbar compacto */}
        <header className="topbar">
          <h1>Detector de aglomeraciones</h1>
        </header>

        <main className="app-main">
          {tab === "monitoreo" && <CameraGrid cameras={cameras} />}
          {tab === "dashboards" && <DashboardGrid cameras={cameras} />}
          {tab === "manager" && (
            <CameraManager
              cameras={cameras}
              setCameras={(newList) => setCameras(newList)}
            />
          )}
        </main>

        <footer className="app-footer">
          <span>© 2025 Monitor de Afluencia - UTFSM</span>
        </footer>
      </div>
    </div>
  );
}
