import { useState } from "react";
import { login } from "../services/api";

export default function Login({ onSuccess }) {
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    try {
      await login(user, pass);
      onSuccess();
    } catch {
      setError("Acceso denegado. Verifica usuario/contraseña.");
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
            placeholder="Ingresa tu usuario"
            value={user}
            onChange={(e) => setUser(e.target.value)}
            required
            autoFocus
            autoComplete="username"
          />

          <label htmlFor="pass">Contraseña</label>
          <input
            id="pass"
            className="login-input"
            type="password"
            placeholder="••••••••"
            value={pass}
            onChange={(e) => setPass(e.target.value)}
            required
            autoComplete="current-password"
          />

          <div className="login-actions">
            <button className="login-btn" type="submit">Ingresar</button>
            {error && <span className="login-error">{error}</span>}
          </div>
        </form>

        <p className="login-foot">Acceso restringido</p>
      </div>
    </div>
  );
}
