import axios from "axios";
const API = axios.create({ baseURL: "http://localhost:8000" });

// --- LOGIN JSON a /auth/login ---
export const login = (u, p) =>
  API.post("/auth/login", { username: u, password: p }, {
    headers: { "Content-Type": "application/json" }
  });

// --- Helpers para guardar/usar el token ---
export function setAuthToken(token) {
  if (token) {
    localStorage.setItem("auth_token", token);
    API.defaults.headers.common["Authorization"] = `Bearer ${token}`;
  } else {
    localStorage.removeItem("auth_token");
    delete API.defaults.headers.common["Authorization"];
  }
}

// Inicializa Authorization si ya había token guardado
const saved = localStorage.getItem("auth_token");
if (saved) {
  API.defaults.headers.common["Authorization"] = `Bearer ${saved}`;
}

// (dejamos tus otros endpoints igual por ahora)
export const getCount = () => API.get("/count");
export const getHistory = () => API.get("/history");
export const getReport = () => API.get("/report", { responseType: "blob" });

// Alertas recientes (opcional ?since=epoch)
export const getAlerts = (since) =>
  API.get("/alerts", { params: since ? { since } : {} });
