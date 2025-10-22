// src/components/Login.jsx
import { useState } from "react";
import { login, setAuthToken } from "../services/api";

export default function Login({ onSuccess }) {
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const res = await login(user, pass);        // POST /auth/login
      const { access_token } = res.data || {};
      if (!access_token) throw new Error("Sin token en la respuesta");
      setAuthToken(access_token);                 // guarda token y setea Authorization
      onSuccess?.();                              // navega al dashboard
    } catch (err) {
      const msg = err?.response?.data?.detail || "Acceso denegado. Verifica usuario/contraseña.";
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="login-shell">
      <div className="login-card" role="region" aria-label="Formulario de inicio de sesión">
        <div className="login-header" style={{ display: "flex", alignItems: "center" }}>
          <h1 className="login-title" style={{ flexGrow: 1 }}>Monitoreo de aglomeraciones</h1>
        </div>
        <p className="login-subtitle">Accede para visualizar en tiempo real</p>

        <form className="login-form" onSubmit={handleSubmit}>
          <label htmlFor="user">Usuario</label>
          <input
            id="user"
            className="login-input"
            type="text"
            value={user}
            onChange={(e) => setUser(e.target.value)}
            required
            autoComplete="username"
          />

          <label htmlFor="pass">Contraseña</label>
          <input
            id="pass"
            className="login-input"
            type="password"
            value={pass}
            onChange={(e) => setPass(e.target.value)}
            required
            autoComplete="current-password"
          />

          <div className="login-actions">
            <button className="login-btn" type="submit" disabled={submitting}>
              {submitting ? "Ingresando..." : "Ingresar"}
            </button>
            {error && <span className="login-error">{error}</span>}
          </div>
        </form>

        <p className="login-foot">Acceso restringido</p>
      </div>
    </div>
  );
}
