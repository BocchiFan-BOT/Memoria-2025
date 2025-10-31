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
  const [tab, setTab] = useState("home"); // parte en la pantalla principal 

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

  const handleLoginSuccess = () => {
    setIsAuthenticated(true);
    setTab("home"); //  manda a la pantalla de bienvenida altiro dps del login
  };

  if (!isAuthenticated) {
    return <Login onSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="app-root">
      <Alerts />

     
      <aside className="sidebar wide">
        <nav className="side-nav">
          {/* HOME / INICIO */}
          <button
            className={`nav-row ${tab === "home" ? "active" : ""}`}
            onClick={() => setTab("home")}
            title="Inicio"
          >
            {/* Icono: casita */}
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M4 10.5 12 4l8 6.5V20a1 1 0 0 1-1 1h-4.5v-5h-5v5H5a1 1 0 0 1-1-1v-9.5Z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinejoin="round"
              />
            </svg>
            <span>Inicio</span>
          </button>

          {/* MONITOREO */}
          <button
            className={`nav-row ${tab === "monitoreo" ? "active" : ""}`}
            onClick={() => setTab("monitoreo")}
            title="Monitoreo de cámaras"
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
            <span>Cámaras</span>
          </button>

          {/* DASHBOARD */}
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
            <span>Reportes</span>
          </button>

          
          <button
            className={`nav-row ${tab === "manager" ? "active" : ""}`}
            onClick={() => setTab("manager")}
            title="Gestionar cámaras"
          >
            
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

  
      <div className="content">
       
        <header className="topbar">
          <h1>Monitoreo de afluencia</h1>
        </header>

        <main className="app-main">
          {tab === "home" && (
            <HomePanel
              cameras={cameras}
              goDashboard={() => setTab("dashboards")}
              goMonitoreo={() => setTab("monitoreo")}
              goManager={() => setTab("manager")}
            />
          )}

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


function HomePanel({ cameras, goDashboard, goMonitoreo, goManager }) {
  return (
    <div className="home-panel">
      <div className="home-head">
        <div>
          <p className="home-kicker">Página principal</p>
          <h2>Bienvenid@, Admin 👋</h2>
          <p className="home-sub">
            Monitorea cámaras, revisa el dashboard o administra puntos de captura.
          </p>
        </div>


        <div className="home-right">
          <img
            src="/logo_monitor.png"
            alt="Logo monitor inteligente"
            className="home-logo"
          />
          <div className="home-badge">Acceso autorizado</div>
        </div>
      </div>

      <div className="home-actions">
        <button className="home-card" onClick={goDashboard}>
          <h3 className="home-card-title">Ir al dashboard</h3>
          <p className="home-card-text">Gráficos, históricos y conteo en tiempo real.</p>
        </button>

        <button className="home-card" onClick={goMonitoreo}>
          <h3 className="home-card-title">Monitoreo de cámaras</h3>
          <p className="home-card-text">
            {cameras.length
              ? `${cameras.length} cámaras registradas`
              : "No hay cámaras registradas"}
          </p>
        </button>

        <button className="home-card" onClick={goManager}>
          <h3 className="home-card-title">Gestionar</h3>
          <p className="home-card-text">Agregar, editar o eliminar cámaras.</p>
        </button>
      </div>
    </div>
  );
}
