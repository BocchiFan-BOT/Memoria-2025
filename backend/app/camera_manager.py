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


_processors: Dict[str, VideoProcessor] = {}  
# Umbrales en memoria por cámara (no persistentes)
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
                item = {
                    "id": c.public_id,
                    "name": c.name,
                    "url": c.url,
                    "location": c.location,
                    "coordinates": coords
                }
                # Adjunta, si existen, umbrales configurados en memoria
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


def add_camera(cam: dict):
    """
    Inserta o actualiza una cámara en la BD a partir del dict
    {id, name, url, location, coordinates?}. También inicia el VideoProcessor
    si no existe aún para ese public_id.
    """
    with _lock:
        db = get_session()
        try:
            # Validamos/normalizamos usando el esquema pensado para tu JSON
            j = schemas.CamaraFromJSON.model_validate(cam)
            lat, lon = j.lat_lon()

            existing = get_camara_by_public_id(db, j.id) or get_camara_by_url(db, str(j.url))
            if existing:
                changes = schemas.CamaraUpdate(
                    public_id=j.id,
                    name=j.name,
                    url=j.url,
                    location=j.location,
                    latitude=lat,
                    longitude=lon,
                )
                update_camara(db, existing, changes)
            else:
                data = schemas.CamaraCreate(
                    public_id=j.id,
                    name=j.name,
                    url=j.url,
                    location=j.location,
                    latitude=lat,
                    longitude=lon,
                )
                create_camara(db, data)

            # Levantar procesador si no existe aún
            try:
                if j.id not in _processors:
                    vp = VideoProcessor(str(j.url), cam_id=j.id, cam_name=j.name)
                    _processors[j.id] = vp
                else:
                    vp = _processors[j.id]

                # Umbrales opcionales por cámara desde el payload
                ct = cam.get("alert_count_threshold")
                ot = cam.get("alert_occ_threshold")
                # Guarda en memoria
                tmp: Dict[str, float] = {}
                if ct is not None:
                    try:
                        tmp["alert_count_threshold"] = float(ct)
                    except Exception:
                        pass
                if ot is not None:
                    try:
                        tmp["alert_occ_threshold"] = float(ot)
                    except Exception:
                        pass
                if tmp:
                    _thresholds[j.id] = {**_thresholds.get(j.id, {}), **tmp}
                    # Aplica en el procesador
                    vp.set_alert_thresholds(
                        count_threshold=tmp.get("alert_count_threshold"),
                        occ_threshold=tmp.get("alert_occ_threshold"),
                    )
            except Exception:
                # Si no se puede abrir la fuente, igual dejamos la cámara en BD
                pass
        finally:
            db.close()


def remove_camera(cam_id: str):
    """
    Elimina la cámara en BD por public_id y detiene su VideoProcessor si corre.
    """
    with _lock:
        db = get_session()
        try:
            existing = get_camara_by_public_id(db, cam_id)
            if existing:
                delete_camara(db, existing)
        finally:
            db.close()

        # Detener y limpiar procesador
        if cam_id in _processors:
            try:
                _processors[cam_id].stop()
            except Exception:
                pass
            _processors.pop(cam_id, None)


def get_camera(cam_id: str) -> Optional[dict]:
    """
    Obtiene una cámara desde BD por public_id.
    """
    db = get_session()
    try:
        c = get_camara_by_public_id(db, cam_id)
        if not c:
            return None
        coords = None
        if c.latitude is not None and c.longitude is not None:
            coords = f"{float(c.latitude):.6f}, {float(c.longitude):.6f}"
        return {
            "id": c.public_id,
            "name": c.name,
            "url": c.url,
            "location": c.location,
            "coordinates": coords
        }
    finally:
        db.close()


def get_processor(cam_id: str, create_if_missing: bool = True) -> Optional[VideoProcessor]:
    """
    Devuelve el VideoProcessor del public_id. Si no existe y create_if_missing=True,
    lo crea leyendo la URL desde la BD.
    """
    with _lock:
        if cam_id in _processors:
            return _processors[cam_id]

        if not create_if_missing:
            return None

        db = get_session()
        try:
            c = get_camara_by_public_id(db, cam_id)
            if not c:
                return None
            try:
                vp = VideoProcessor(c.url, cam_id=c.public_id, cam_name=c.name)
                # Si existen umbrales en memoria, aplicarlos
                th = _thresholds.get(cam_id)
                if th:
                    vp.set_alert_thresholds(
                        count_threshold=th.get("alert_count_threshold"),
                        occ_threshold=th.get("alert_occ_threshold"),
                    )
                _processors[cam_id] = vp
                return vp
            except Exception:
                return None
        finally:
            db.close()


def get_alerts(since: float | None = None, cam_id: Optional[str] = None):
    """
    Agrega alertas recientes de todos los VideoProcessor.
    since: epoch seconds para filtrar (opcional)
    cam_id: filtrar por cámara específica (opcional)
    """
    out = []
    with _lock:
        for cid, vp in _processors.items():
            if cam_id and cid != cam_id:
                continue
            try:
                out.extend(vp.get_alerts(since))
            except Exception:
                pass
    # ordenar por tiempo asc
    out.sort(key=lambda a: a.get("t", 0))
    return out


def get_metrics_all():
    """
    Devuelve métricas de rendimiento de todos los VideoProcessor activos.
    """
    out = []
    with _lock:
        for cid, vp in _processors.items():
            try:
                out.append(vp.get_metrics())
            except Exception:
                pass
    return out


def get_metrics_one(cam_id: str) -> Optional[dict]:
    with _lock:
        vp = _processors.get(cam_id)
        if not vp:
            return None
        try:
            return vp.get_metrics()
        except Exception:
            return None
