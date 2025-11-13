import threading
from typing import Optional, Dict, Any

from .video_processor import VideoProcessor

# ORM / CRUD
from .database.crud import (
    get_session,
    get_camara_by_public_id,
    get_camara_by_url,
    create_camara,
    update_camara,
    delete_camara,
    list_camaras as crud_list_camaras,
)
from .database import schemas

_lock = threading.Lock()

# Procesadores activos por public_id
_processors: Dict[str, VideoProcessor] = {}

# Umbrales en memoria (se sincronizan con lo que hay en BD)
_thresholds: Dict[str, Dict[str, float]] = {}


def list_cameras():
    """
    Devuelve la lista de cámaras desde la BD con el mismo formato
    que espera el frontend: {id, name, url, location, coordinates}
    """
    with _lock:
        db = get_session()
        try:
            cams = crud_list_camaras(db, limit=1000)
            out = []
            for c in cams:
                coords = None
                if c.latitude is not None and c.longitude is not None:
                    coords = f"{float(c.latitude):.6f}, {float(c.longitude):.6f}"

                item: Dict[str, Any] = {
                    "id": c.public_id,   # OJO: el frontend usa este id
                    "name": c.name,
                    "url": c.url,
                    "location": c.location,
                    "coordinates": coords,
                }

                # Umbrales guardados en BD
                if getattr(c, "alert_count_threshold", None) is not None:
                    item["alert_count_threshold"] = float(c.alert_count_threshold)
                if getattr(c, "alert_occ_threshold", None) is not None:
                    item["alert_occ_threshold"] = float(c.alert_occ_threshold)

                # Si hay overrides en memoria, pisan los valores
                th = _thresholds.get(c.public_id)
                if th:
                    if "alert_count_threshold" in th:
                        item["alert_count_threshold"] = th["alert_count_threshold"]
                    if "alert_occ_threshold" in th:
                        item["alert_occ_threshold"] = th["alert_occ_threshold"]

                out.append(item)
            return out
        finally:
            db.close()


def add_camera(cam: dict, update_id: Optional[str] = None):
    """
    Inserta o actualiza una cámara en BD.

    - Si update_id existe → UPDATE sobre esa cámara (no cambia el public_id).
    - Si update_id es None → INSERT o UPSERT usando cam["id"] como public_id.
    También inicia el VideoProcessor si no existe.
    """
    # public_id siempre viene como cam.id en el frontend
    public_id = str(cam.get("id") or cam.get("public_id") or "").strip()
    if not public_id:
        raise ValueError("cam.id (public_id) es obligatorio")

    with _lock:
        db = get_session()
        try:
            # =========================
            #        MODO UPDATE
            # =========================
            if update_id is not None:
                existing = get_camara_by_public_id(db, update_id)
                if not existing:
                    raise Exception("Cámara no encontrada para update")

                changes = schemas.CamaraUpdate(
                    # NO cambiamos public_id aquí
                    name=cam.get("name"),
                    url=cam.get("url"),
                    location=cam.get("location"),
                    latitude=None,
                    longitude=None,
                    alert_count_threshold=cam.get("alert_count_threshold"),
                    alert_occ_threshold=cam.get("alert_occ_threshold"),
                )

                updated = update_camara(db, existing, changes)

                # Sincroniza umbrales en memoria con lo que quedó en BD
                ct = getattr(updated, "alert_count_threshold", None)
                ot = getattr(updated, "alert_occ_threshold", None)

                if update_id not in _thresholds:
                    _thresholds[update_id] = {}

                if ct is not None:
                    _thresholds[update_id]["alert_count_threshold"] = float(ct)
                if ot is not None:
                    _thresholds[update_id]["alert_occ_threshold"] = float(ot)

                vp = _processors.get(update_id)
                if vp:
                    vp.set_alert_thresholds(
                        count_threshold=ct,
                        occ_threshold=ot,
                    )

                return {"updated": True}

            # =========================
            #   MODO INSERT / UPSERT
            # =========================
            existing = get_camara_by_public_id(db, public_id) or get_camara_by_url(
                db, str(cam.get("url", ""))
            )

            if existing:
                # UPSERT: si ya existe, se actualiza
                changes = schemas.CamaraUpdate(
                    name=cam.get("name"),
                    url=cam.get("url"),
                    location=cam.get("location"),
                    latitude=None,
                    longitude=None,
                    alert_count_threshold=cam.get("alert_count_threshold"),
                    alert_occ_threshold=cam.get("alert_occ_threshold"),
                )
                cam_row = update_camara(db, existing, changes)
                created = False
            else:
                data = schemas.CamaraCreate(
                    public_id=public_id,
                    name=cam["name"],
                    url=cam["url"],
                    location=cam.get("location"),
                    latitude=None,
                    longitude=None,
                    alert_count_threshold=cam.get("alert_count_threshold"),
                    alert_occ_threshold=cam.get("alert_occ_threshold"),
                )
                cam_row = create_camara(db, data)
                created = True

            # Levantar procesador si no existe aún
            if public_id not in _processors:
                vp = VideoProcessor(
                    str(cam_row.url),
                    cam_id=cam_row.public_id,
                    cam_name=cam_row.name,
                )

                # Umbrales opcionales desde BD
                th: Dict[str, float] = {}
                if getattr(cam_row, "alert_count_threshold", None) is not None:
                    th["alert_count_threshold"] = float(cam_row.alert_count_threshold)
                if getattr(cam_row, "alert_occ_threshold", None) is not None:
                    th["alert_occ_threshold"] = float(cam_row.alert_occ_threshold)

                if th:
                    _thresholds[public_id] = {
                        **_thresholds.get(public_id, {}),
                        **th,
                    }
                    vp.set_alert_thresholds(
                        count_threshold=th.get("alert_count_threshold"),
                        occ_threshold=th.get("alert_occ_threshold"),
                    )

                _processors[public_id] = vp

            return {"inserted": created, "id": public_id}
        finally:
            db.close()


def remove_camera(cam_id: str):
    """
    Elimina cámara de BD y detiene el VideoProcessor asociado.
    cam_id aquí es public_id.
    """
    with _lock:
        db = get_session()
        try:
            cam = get_camara_by_public_id(db, cam_id)
            if cam:
                delete_camara(db, cam)

            vp = _processors.pop(cam_id, None)
            if vp:
                vp.stop()

            _thresholds.pop(cam_id, None)
        finally:
            db.close()


def get_camera(cam_id: str) -> Optional[Dict[str, Any]]:
    """
    Devuelve info de una cámara específica desde BD.
    """
    with _lock:
        db = get_session()
        try:
            cam = get_camara_by_public_id(db, cam_id)
            if not cam:
                return None

            coords = None
            if cam.latitude is not None and cam.longitude is not None:
                coords = f"{float(cam.latitude):.6f}, {float(cam.longitude):.6f}"

            out: Dict[str, Any] = {
                "id": cam.public_id,
                "name": cam.name,
                "url": cam.url,
                "location": cam.location,
                "coordinates": coords,
                "status": cam.status,
                "is_online": cam.is_online,
            }

            if getattr(cam, "alert_count_threshold", None) is not None:
                out["alert_count_threshold"] = float(cam.alert_count_threshold)
            if getattr(cam, "alert_occ_threshold", None) is not None:
                out["alert_occ_threshold"] = float(cam.alert_occ_threshold)

            th = _thresholds.get(cam.public_id)
            if th:
                if "alert_count_threshold" in th:
                    out["alert_count_threshold"] = th["alert_count_threshold"]
                if "alert_occ_threshold" in th:
                    out["alert_occ_threshold"] = th["alert_occ_threshold"]

            return out
        finally:
            db.close()


def get_processor(cam_id: str, create_if_missing: bool = True) -> Optional[VideoProcessor]:
    """
    Devuelve (o crea) el VideoProcessor asociado a una cámara.
    cam_id es el public_id.
    """
    with _lock:
        vp = _processors.get(cam_id)
        if vp or not create_if_missing:
            return vp

        db = get_session()
        try:
            cam = get_camara_by_public_id(db, cam_id)
            if not cam:
                return None

            vp = VideoProcessor(str(cam.url), cam_id=cam.public_id, cam_name=cam.name)

            th: Dict[str, float] = {}
            if getattr(cam, "alert_count_threshold", None) is not None:
                th["alert_count_threshold"] = float(cam.alert_count_threshold)
            if getattr(cam, "alert_occ_threshold", None) is not None:
                th["alert_occ_threshold"] = float(cam.alert_occ_threshold)

            if th:
                _thresholds[cam.public_id] = {
                    **_thresholds.get(cam.public_id, {}),
                    **th,
                }
                vp.set_alert_thresholds(
                    count_threshold=th.get("alert_count_threshold"),
                    occ_threshold=th.get("alert_occ_threshold"),
                )

            _processors[cam.public_id] = vp
            return vp
        finally:
            db.close()


def get_alerts(since: float | None = None, cam_id: str | None = None):
    """
    Agrega alertas desde todos los VideoProcessor activos.
    - since: timestamp epoch opcional
    - cam_id: filtra por cámara
    """
    with _lock:
        alerts = []
        for cid, vp in _processors.items():
            if cam_id and cam_id != cid:
                continue
            alerts.extend(vp.get_alerts(since_ts=since))
        alerts.sort(key=lambda a: a.get("t", 0.0))
        return alerts


def get_metrics_all():
    """
    Devuelve métricas de rendimiento de todos los procesadores activos.
    """
    with _lock:
        return [vp.get_metrics() for vp in _processors.values()]


def get_metrics_one(cam_id: str):
    """
    Devuelve métricas de una cámara específica, si su procesador está activo.
    """
    with _lock:
        vp = _processors.get(cam_id)
        return vp.get_metrics() if vp else None
