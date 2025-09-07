import axios from "axios";
const API = axios.create({ baseURL: "http://localhost:8000" });

export const login = (u, p) => {
  const data = new FormData();
  data.append("username", u);
  data.append("password", p);
  return API.post("/login", data);
};

export const getCount = () => API.get("/count");
export const getHistory = () => API.get("/history");
export const getReport = () => API.get("/report", { responseType: "blob" });

