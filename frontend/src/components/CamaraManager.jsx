// frontend/src/components/CamaraManager.jsx
import React, { useState } from "react";

export default function CameraManager({ cameras, setCameras }) {
  const [form, setForm] = useState({
    name: "",
    url: "",
    location: "",
    coordinates: "",
    alert_count_threshold: "",
    alert_occ_threshold: "",
  });

  const [editingCam, setEditingCam] = useState(null);
  const [editForm, setEditForm] = useState({
    name: "",
    url: "",
    location: "",
    coordinates: "",
    alert_count_threshold: "",
    alert_occ_threshold: "",
  });

  const onChange = (e) =>
    setForm({ ...form, [e.target.name]: e.target.value });

  const onEditChange = (e) =>
    setEditForm({ ...editForm, [e.target.name]: e.target.value });

  const add = async () => {
    if (!form.name || !form.url) {
      alert("Nombre y URL son obligatorios");
      return;
    }

    const id = Date.now().toString();
    const newCam = { id, ...form };

    // Limpia vacíos para no enviar strings vacíos
    if (newCam.alert_count_threshold === "") delete newCam.alert_count_threshold;
    if (newCam.alert_occ_threshold === "") delete newCam.alert_occ_threshold;

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

    setForm({
      name: "",
      url: "",
      location: "",
      coordinates: "",
      alert_count_threshold: "",
      alert_occ_threshold: "",
    });
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

  const startEdit = (cam) => {
    setEditingCam(cam);
    setEditForm({
      name: cam.name || "",
      url: cam.url || "",
      location: cam.location || "",
      coordinates: cam.coordinates || "",
      alert_count_threshold:
        cam.alert_count_threshold !== undefined
          ? cam.alert_count_threshold
          : "",
      alert_occ_threshold:
        cam.alert_occ_threshold !== undefined
          ? cam.alert_occ_threshold
          : "",
    });
  };

  const cancelEdit = () => {
    setEditingCam(null);
  };

  const saveEdit = async () => {
    if (!editForm.name || !editForm.url) {
      alert("Nombre y URL son obligatorios");
      return;
    }

    // Construir payload para backend
    const payload = {
      id: editingCam.id,
      name: editForm.name,
      url: editForm.url,
      location: editForm.location,
      coordinates: editForm.coordinates,
      alert_count_threshold: editForm.alert_count_threshold,
      alert_occ_threshold: editForm.alert_occ_threshold,
    };

    if (payload.alert_count_threshold === "")
      delete payload.alert_count_threshold;
    if (payload.alert_occ_threshold === "")
      delete payload.alert_occ_threshold;

    // 1. Actualizar frontend
    const updatedList = cameras.map((c) =>
      c.id === editingCam.id ? { ...c, ...payload } : c
    );
    setCameras(updatedList, false);

    // 2. Enviar al backend (asumo PUT /cameras/{id})
    try {
      const token = localStorage.getItem("auth_token");
      await fetch(`http://localhost:8000/cameras/${editingCam.id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      console.error("Error actualizando cámara en backend:", err);
    }

    setEditingCam(null);
  };

  return (
    <div className="manager">
      <h2 className="manager-title">Agregar nueva cámara</h2>

      {/* formulario alta */}
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

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 12,
          }}
        >
          <label>
            Umbral conteo (personas)
            <input
              name="alert_count_threshold"
              type="number"
              min={0}
              placeholder="Ej: 50 (vacío usa global)"
              value={form.alert_count_threshold}
              onChange={onChange}
            />
          </label>
          <label>
            Umbral ocupación (%)
            <input
              name="alert_occ_threshold"
              type="number"
              min={0}
              max={100}
              step={0.1}
              placeholder="Ej: 10 (vacío usa global)"
              value={form.alert_occ_threshold}
              onChange={onChange}
            />
          </label>
        </div>

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
              {(cam.alert_count_threshold !== undefined ||
                cam.alert_occ_threshold !== undefined) && (
                <span className="camera-item-meta">
                  {cam.alert_count_threshold !== undefined &&
                    ` | Umbral de Conteo: ${cam.alert_count_threshold}`}
                  {cam.alert_occ_threshold !== undefined &&
                    ` | Umbral de Aglomeración: ${cam.alert_occ_threshold}%`}
                </span>
              )}
            </div>
            <div className="camera-item-actions">
              <button
                className="btn-edit"
                onClick={() => startEdit(cam)}
                aria-label={`Editar cámara ${cam.name}`}
              >
                Editar
              </button>
              <button
                className="btn-delete"
                onClick={() => remove(cam.id)}
                aria-label={`Eliminar cámara ${cam.name}`}
              >
                Eliminar
              </button>
            </div>
          </li>
        ))}
      </ul>

      {/* Modal edición */}
      {editingCam && (
        <div
          className="modal-backdrop"
          onClick={cancelEdit}
        >
          <div
            className="modal"
            onClick={(e) => e.stopPropagation()}
          >
            <header className="modal-header">
              <h3>Editar cámara</h3>
              <button
                className="modal-close"
                onClick={cancelEdit}
              >
                ✕
              </button>
            </header>

            <div className="modal-body">
              <label>
                Nombre
                <input
                  name="name"
                  value={editForm.name}
                  onChange={onEditChange}
                />
              </label>
              <label>
                URL (rtsp/rtmp/http/mjpg)
                <input
                  name="url"
                  value={editForm.url}
                  onChange={onEditChange}
                />
              </label>
              <label>
                Dirección
                <input
                  name="location"
                  value={editForm.location}
                  onChange={onEditChange}
                />
              </label>
              <label>
                Coordenadas GPS
                <input
                  name="coordinates"
                  value={editForm.coordinates}
                  onChange={onEditChange}
                />
              </label>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 12,
                }}
              >
                <label>
                  Umbral conteo (personas)
                  <input
                    name="alert_count_threshold"
                    type="number"
                    min={0}
                    value={editForm.alert_count_threshold}
                    onChange={onEditChange}
                  />
                </label>
                <label>
                  Umbral ocupación (%)
                  <input
                    name="alert_occ_threshold"
                    type="number"
                    min={0}
                    max={100}
                    step={0.1}
                    value={editForm.alert_occ_threshold}
                    onChange={onEditChange}
                  />
                </label>
              </div>
            </div>

            <footer className="modal-footer">
              <button
                className="btn-secondary"
                type="button"
                onClick={cancelEdit}
              >
                Cancelar
              </button>
              <button
                className="btn-primary"
                type="button"
                onClick={saveEdit}
              >
                Guardar cambios
              </button>
            </footer>
          </div>
        </div>
      )}
    </div>
  );
}
