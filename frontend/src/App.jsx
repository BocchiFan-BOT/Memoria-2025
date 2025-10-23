// frontend/src/App.jsx
import React, { useState, useEffect } from "react";
import Login from "./components/Login";
import CameraManager from "./components/CamaraManager";
import CameraGrid from "./components/CamaraGrid";
import DashboardGrid from "./components/DashboardGrid";
import "./index.css";
import Alerts from "./components/Alerts";

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [cameras, setCameras] = useState([]); 
  const [tab, setTab] = useState("monitoreo");

  useEffect(() => {
    // Obtener lista de cámaras del backend
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
  };

  if (!isAuthenticated) {
    return <Login onSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="app-root">
      <Alerts />
      <header className="app-header">
        <h1>Monitor de Afluencia - Memoria</h1>
        <nav className="app-nav">
          <button
            className={tab === "monitoreo" ? "active" : ""}
            onClick={() => setTab("monitoreo")}
          >
            Cámaras
          </button>
          <button
            className={tab === "dashboards" ? "active" : ""}
            onClick={() => setTab("dashboards")}
          >
            Dashboards
          </button>
          <button
            className={tab === "manager" ? "active" : ""}
            onClick={() => setTab("manager")}
          >
            Agregar / Eliminar
          </button>
        </nav>
      </header>

      <main className="app-main">
        {tab === "monitoreo" && <CameraGrid cameras={cameras} />}
        {tab === "dashboards" && <DashboardGrid cameras={cameras} />}
        {tab === "manager" && (
          <CameraManager
            cameras={cameras}
            setCameras={(newList) => setCameras(newList)} // ahora no hace fetch aquí
          />
        )}
      </main>

      <footer className="app-footer">
        <span>© 2025 Monitor de Afluencia - UTFSM</span>
      </footer>
    </div>
  );
}

export default App;





