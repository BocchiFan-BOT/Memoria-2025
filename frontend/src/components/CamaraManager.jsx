// frontend/src/components/CamaraManager.jsx
import React, { useState } from "react";

export default function CameraManager({ cameras, setCameras }) {
  const [form, setForm] = useState({ name: "", url: "", location: "", coordinates: "" });

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

    // 2. Enviar al backend para que cree el VideoProcessor
    try {
      await fetch("http://localhost:8000/cameras/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
      await fetch(`http://localhost:8000/cameras/remove/${id}`, {
        method: "DELETE",
      });
    } catch (err) {
      console.error("Error eliminando cámara en backend:", err);
    }
  };

  return (
    <div className="manager">
      <h2>Agregar nueva cámara</h2>
      <div className="form">
        <input name="name" placeholder="Nombre" value={form.name} onChange={onChange} />
        <input name="url" placeholder="URL (rtsp/rtmp/http/mjpg)" value={form.url} onChange={onChange} />
        <input name="location" placeholder="Dirección" value={form.location} onChange={onChange} />
        <input name="coordinates" placeholder="Coordenadas GPS" value={form.coordinates} onChange={onChange} />
        <div className="buttons">
          <button onClick={add} className="btn btn-primary">Agregar</button>
        </div>
      </div>

      <h3>Cámaras configuradas</h3>
      <ul className="camera-list">
        {cameras.map((cam) => (
          <li key={cam.id} className="camera-item">
            <div>
              <strong>{cam.name}</strong> <br />
              <span>{cam.location} {cam.coordinates ? `- ${cam.coordinates}` : ""}</span>
              <div className="small">{cam.url}</div>
            </div>
            <div>
              <button className="btn btn-danger" onClick={() => remove(cam.id)}>Eliminar</button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}


