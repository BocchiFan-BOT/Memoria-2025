// frontend/src/App.jsx
import React, { useState, useEffect } from "react";
import Login from "./components/Login";
import CameraManager from "./components/CamaraManager";
import CameraGrid from "./components/CamaraGrid";
import DashboardGrid from "./components/DashboardGrid";
import "./index.css";

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [cameras, setCameras] = useState([]); // lista local, sincronizable con backend
  const [tab, setTab] = useState("monitoreo");

  useEffect(() => {
    // al iniciar, pedir lista de cámaras al backend (si existe)
    fetch("http://localhost:8000/cameras")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setCameras(data);
      })
      .catch(() => {
        // backend quizá no expuesto aún; mantener vacío.
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
      <header className="app-header">
        <h1>Monitor de Afluencia - Memoria</h1>
        <nav className="app-nav">
          <button className={tab==="monitoreo" ? "active":""} onClick={()=>setTab("monitoreo")}>Cámaras</button>
          <button className={tab==="dashboards" ? "active":""} onClick={()=>setTab("dashboards")}>Dashboards</button>
          <button className={tab==="manager" ? "active":""} onClick={()=>setTab("manager")}>Agregar / Eliminar</button>
        </nav>
      </header>

      <main className="app-main">
        {tab === "monitoreo" && <CameraGrid cameras={cameras} />}
        {tab === "dashboards" && <DashboardGrid cameras={cameras} />}
        {tab === "manager" && (
          <CameraManager
            cameras={cameras}
            setCameras={(newList, persist=true) => {
              setCameras(newList);
              // intentar persistir al backend
              if (persist) {
                fetch("http://localhost:8000/cameras", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(newList),
                }).catch(()=>{/* no crítico */});
              }
            }}
          />
        )}
      </main>

      <footer className="app-footer">
        <span>Usuario: admin | Contraseña: admin</span>
      </footer>
    </div>
  );
}

export default App;




