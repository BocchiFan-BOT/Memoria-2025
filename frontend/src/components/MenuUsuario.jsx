import { useEffect, useRef, useState } from "react";

const LS_NAME_KEY = "ui.profileName";
const LS_AVATAR_KEY = "ui.profileAvatar";

export default function UserMenu({ onLogout }) {
  const [open, setOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [name, setName] = useState("Admin");
  const [avatar, setAvatar] = useState("/usuario.png"); // dataURL

  const menuRef = useRef(null);

  // Cargar desde localStorage
  useEffect(() => {
    const n = localStorage.getItem(LS_NAME_KEY);
    const a = localStorage.getItem(LS_AVATAR_KEY);
    if (n) setName(n);
    if (a) setAvatar(a);
  }, []);

  // Cerrar al hacer clic fuera
  useEffect(() => {
    function handleClickOutside(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Helpers
  const initials = name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase())
    .join("") || "A";

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setAvatar(reader.result);
    };
    reader.readAsDataURL(file);
  }

  function handleSaveProfile() {
    const safeName = name?.trim() || "Admin";
    setName(safeName);
    localStorage.setItem(LS_NAME_KEY, safeName);
    if (avatar) localStorage.setItem(LS_AVATAR_KEY, avatar);
    else localStorage.removeItem(LS_AVATAR_KEY);
    setEditOpen(false);
  }

  function handleClearAvatar() {
    setAvatar(null);
    localStorage.removeItem(LS_AVATAR_KEY);
  }

  return (
    <div className="user-menu" ref={menuRef}>
      <button
        className="user-btn"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Abrir menú de usuario"
      >
        <div className="user-avatar" aria-hidden="true">
          {avatar ? (
            <img src={avatar} alt="" />
          ) : (
            <span className="user-initials">{initials}</span>
          )}
        </div>
        <span className="user-name">{name}</span>
        <svg
          className={`user-caret ${open ? "open" : ""}`}
          width="16"
          height="16"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            d="M7 10l5 5 5-5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {open && (
        <div className="user-dropdown" role="menu">
          <button
            className="user-dd-item"
            role="menuitem"
            onClick={() => {
              setEditOpen(true);
              setOpen(false);
            }}
          >
            Editar perfil
          </button>
          <div className="user-dd-sep" />
          <button
            className="user-dd-item danger"
            role="menuitem"
            onClick={onLogout}
          >
            Cerrar sesión
          </button>
        </div>
      )}

      {editOpen && (
        <div
          className="user-modal"
          role="dialog"
          aria-modal="true"
          aria-labelledby="userModalTitle"
        >
          <div className="user-modal-card">
            <div className="user-modal-header">
              <h3 id="userModalTitle">Editar perfil</h3>
              <button
                className="user-modal-close"
                onClick={() => setEditOpen(false)}
                aria-label="Cerrar"
              >
                ×
              </button>
            </div>

            <div className="user-modal-body">
              <label className="user-field">
                <span>Nombre / apodo</span>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Escribe tu nombre"
                />
              </label>

              <div className="user-avatar-edit">
                <div className="user-avatar preview">
                  {avatar ? (
                    <img src={avatar} alt="Avatar actual" />
                  ) : (
                    <span className="user-initials">{initials}</span>
                  )}
                </div>

                <div className="user-avatar-actions">
                  <label className="user-upload-btn">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      style={{ display: "none" }}
                    />
                    Subir foto
                  </label>
                  {avatar && (
                    <button
                      className="user-clear-btn"
                      onClick={handleClearAvatar}
                    >
                      Quitar foto
                    </button>
                  )}
                </div>
              </div>
            </div>

            <div className="user-modal-footer">
              <button className="user-save-btn" onClick={handleSaveProfile}>
                Guardar
              </button>
              <button
                className="user-cancel-btn"
                onClick={() => setEditOpen(false)}
              >
                Cancelar
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
