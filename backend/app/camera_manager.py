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

# Procesadores activos: public_id → VideoProcessor
_processors: Dict[str, VideoProcessor] = {}


def list_cameras():
    """
    Devuelve la lista de cámaras desde la BD con el formato para el frontend.
    """
    db = get_session()
    try:
        cams = crud_list_camaras(db, limit=1000)
        out = []
        for c in cams:
            coords = None
            if c.latitude is not None and c.longitude is not None:
                coords = f"{float(c.latitude):.6f}, {float(c.longitude):.6f}"
            out.append({
                "id": c.public_id,
                "name": c.name,
                "url": c.url,
                "location": c.location,
                "coordinates": coords,
                "alert_count_threshold": c.alert_count_threshold,
                "alert_occ_threshold": c.alert_occ_threshold,
            })
        return out
    finally:
        db.close()


def _start_processor_for(camara_row):
    """
    Crea y registra un VideoProcessor para la cámara.
    """
    public_id = camara_row.public_id

    # Si ya existe, lo detenemos
    if public_id in _processors:
        try:
            _processors[public_id].stop()
        except Exception:
            pass

    vp = VideoProcessor(
        str(camara_row.url),
        cam_id=public_id,
        cam_name=camara_row.name,
    )

    # Asignar umbrales desde base de datos SIEMPRE
    vp.set_alert_thresholds(
        count_threshold=camara_row.alert_count_threshold,
        occ_threshold=camara_row.alert_occ_threshold,
        cooldown_sec=None
    )

    _processors[public_id] = vp
    return vp


def get_processor(cam_id: str, create_if_missing=True) -> Optional[VideoProcessor]:
    """
    Obtiene el VideoProcessor de una cámara.
    """
    with _lock:
        if cam_id in _processors:
            return _processors[cam_id]

        if not create_if_missing:
            return None

        db = get_session()
        try:
            cam = get_camara_by_public_id(db, cam_id)
            if not cam:
                return None
            return _start_processor_for(cam)
        finally:
            db.close()


def add_camera(data: Dict[str, Any]):
    """
    Crea o actualiza cámara en BD y sincroniza VideoProcessor.
    """
    with _lock:
        db = get_session()
        try:
            public_id = str(data.get("id") or data.get("public_id"))

            existing = get_camara_by_public_id(db, public_id) or get_camara_by_url(db, data["url"])

            if existing:
                # UPDATE
                changes = schemas.CamaraUpdate(
                    name=data.get("name"),
                    url=data.get("url"),
                    location=data.get("location"),
                    status=data.get("status"),
                    is_online=data.get("is_online"),
                    alert_count_threshold=data.get("alert_count_threshold"),
                    alert_occ_threshold=data.get("alert_occ_threshold"),
                )
                cam = update_camara(db, existing, changes)

                # Sincronizar VP
                vp = get_processor(public_id, create_if_missing=True)
                vp.set_alert_thresholds(
                    count_threshold=cam.alert_count_threshold,
                    occ_threshold=cam.alert_occ_threshold,
                    cooldown_sec=None
                )

            else:
                # INSERT
                new = schemas.CamaraCreate(
                    public_id=public_id,
                    name=data["name"],
                    url=data["url"],
                    location=data.get("location"),
                    latitude=data.get("latitude"),
                    longitude=data.get("longitude"),
                    alert_count_threshold=data.get("alert_count_threshold"),
                    alert_occ_threshold=data.get("alert_occ_threshold"),
                )
                cam = create_camara(db, new)

                _start_processor_for(cam)

            return public_id

        finally:
            db.close()


def remove_camera(public_id: str):
    """
    Elimina cámara y detiene su VideoProcessor.
    """
    with _lock:
        vp = _processors.pop(public_id, None)
        if vp:
            try:
                vp.stop()
            except Exception:
                pass

        db = get_session()
        try:
            cam = get_camara_by_public_id(db, public_id)
            if cam:
                delete_camara(db, cam)
        finally:
            db.close()


def get_alerts(since: float | None = None, cam_id: str | None = None):
    """
    Devuelve alertas agregadas desde TODOS los VideoProcessors.
    """
    out = []
    with _lock:
        for cid, vp in _processors.items():
            if cam_id and cid != cam_id:
                continue
            arr = vp.get_alerts(since_ts=since)
            out.extend(arr)
    out.sort(key=lambda x: float(x.get("t", 0)))
    return out


def get_metrics_all():
    with _lock:
        return [vp.get_metrics() for vp in _processors.values()]


def get_metrics_one(cam_id: str):
    with _lock:
        vp = _processors.get(cam_id)
        return vp.get_metrics() if vp else None
