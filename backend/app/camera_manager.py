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
                out.append({
                    "id": c.public_id,
                    "name": c.name,
                    "url": c.url,
                    "location": c.location,
                    "coordinates": coords
                })
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
                    _processors[j.id] = VideoProcessor(str(j.url))
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
                vp = VideoProcessor(c.url)
                _processors[cam_id] = vp
                return vp
            except Exception:
                return None
        finally:
            db.close()
