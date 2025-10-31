// frontend/src/components/CamaraManager.jsx
import React, { useState } from "react";

export default function CameraManager({ cameras, setCameras }) {
  const [form, setForm] = useState({
    name: "",
    url: "",
    location: "",
    coordinates: "",
  });

  const onChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const add = async () => {
    if (!form.name || !form.url) {
      alert("Nombre y URL son obligatorios");
      return;
    }
    const id = Date.now().toString();
    const newCam = { id, ...form };

    // 1. Actualizar frontend
    const updated = [...cameras, newCam];
    setCameras(updated, false);

    // 2. Enviar al backend
    try {
      const token = localStorage.getItem("auth_token");
      await fetch("http://localhost:8000/cameras/add", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(newCam),
      });
    } catch (err) {
      console.error("Error registrando cámara en backend:", err);
    }

    setForm({ name: "", url: "", location: "", coordinates: "" });
  };

  const remove = async (id) => {
    const updated = cameras.filter((c) => c.id !== id);
    setCameras(updated, false);

    try {
      const token = localStorage.getItem("auth_token");
      await fetch(`http://localhost:8000/cameras/${id}`, {
        method: "DELETE",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
    } catch (err) {
      console.error("Error eliminando cámara en backend:", err);
    }
  };

  return (
    <div className="manager">
      <h2 className="manager-title">Agregar nueva cámara</h2>

      {/* formulario */}
      <div className="manager-form">
        <label>
          Nombre
          <input
            name="name"
            placeholder="Ej: Entrada principal"
            value={form.name}
            onChange={onChange}
          />
        </label>
        <label>
          URL (rtsp/rtmp/http/mjpg)
          <input
            name="url"
            placeholder="rtsp:// / http:// ..."
            value={form.url}
            onChange={onChange}
          />
        </label>
        <label>
          Dirección
          <input
            name="location"
            placeholder="Calle, edificio, zona"
            value={form.location}
            onChange={onChange}
          />
        </label>
        <label>
          Coordenadas GPS
          <input
            name="coordinates"
            placeholder="-20.22222, -70.11111"
            value={form.coordinates}
            onChange={onChange}
          />
        </label>

        <button type="button" className="btn-add" onClick={add}>
          + Agregar cámara
        </button>
      </div>

      {/* lista */}
      <h3 className="manager-subtitle">Cámaras configuradas</h3>
      <ul className="camera-list">
        {cameras.map((cam) => (
          <li key={cam.id} className="camera-item">
            <div className="camera-item-info">
              <strong>{cam.name}</strong>
              <span className="camera-item-meta">
                {cam.location}{" "}
                {cam.coordinates ? `– ${cam.coordinates}` : ""}
              </span>
              <span className="camera-item-url">{cam.url}</span>
            </div>
            <button
              className="btn-delete"
              onClick={() => remove(cam.id)}
              aria-label={`Eliminar cámara ${cam.name}`}
            >
              Eliminar
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
