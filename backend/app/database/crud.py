from __future__ import annotations
from typing import Optional, List
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from .database import SessionLocal
from .models import Camara, Historial
from . import schemas


def get_session() -> Session:
    return SessionLocal()


# -------------------------------------------------
# CRUD CÁMARAS
# -------------------------------------------------
def create_camara(db: Session, data: schemas.CamaraCreate) -> Camara:
    """
    Crea una cámara nueva en la BD.
    """
    cam = Camara(
        public_id=data.public_id,
        name=data.name,
        url=str(data.url),
        location=data.location,
        latitude=data.latitude,
        longitude=data.longitude,
        alert_count_threshold=data.alert_count_threshold,
        alert_occ_threshold=data.alert_occ_threshold,
        status="ACTIVE",
        is_online=False,
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


def get_camara_by_id(db: Session, camara_id: int) -> Optional[Camara]:
    return db.get(Camara, camara_id)


def get_camara_by_public_id(db: Session, public_id: str) -> Optional[Camara]:
    stmt = select(Camara).where(Camara.public_id == public_id)
    return db.execute(stmt).scalars().first()


def get_camara_by_url(db: Session, url: str) -> Optional[Camara]:
    stmt = select(Camara).where(Camara.url == url)
    return db.execute(stmt).scalars().first()


def list_camaras(
    db: Session,
    *,
    q: Optional[str] = None,
    status: Optional[str] = None,
    is_online: Optional[bool] = None,
    offset: int = 0,
    limit: int = 50,
) -> List[Camara]:
    """
    Lista cámaras con filtros opcionales por texto, estado y online.
    """
    stmt = select(Camara)
    if q:
        # busca por nombre o location
        like = f"%{q}%"
        stmt = stmt.where(
            (Camara.name.ilike(like)) | (Camara.location.ilike(like))
        )
    if status:
        stmt = stmt.where(Camara.status == status)
    if is_online is not None:
        stmt = stmt.where(Camara.is_online == is_online)

    stmt = stmt.offset(offset).limit(limit)
    return db.execute(stmt).scalars().all()


def update_camara(db: Session, cam: Camara, changes: schemas.CamaraUpdate) -> Camara:
    """
    Aplica cambios parciales a una cámara existente.
    OJO: NO cambia el id interno (PK), sólo los campos lógicos.
    """
    if changes.public_id is not None:
        cam.public_id = changes.public_id
    if changes.name is not None:
        cam.name = changes.name
    if changes.url is not None:
        cam.url = str(changes.url)
    if changes.location is not None:
        cam.location = changes.location
    if changes.latitude is not None:
        cam.latitude = changes.latitude
    if changes.longitude is not None:
        cam.longitude = changes.longitude
    if changes.status is not None:
        cam.status = changes.status
    if changes.is_online is not None:
        cam.is_online = changes.is_online

    # umbrales de alerta
    if changes.alert_count_threshold is not None:
        cam.alert_count_threshold = changes.alert_count_threshold
    if changes.alert_occ_threshold is not None:
        cam.alert_occ_threshold = changes.alert_occ_threshold

    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


def delete_camara(db: Session, cam: Camara) -> None:
    db.delete(cam)
    db.commit()


# -------------------------------------------------
# ESTADO DE CÁMARAS (heartbeat)
# -------------------------------------------------
def update_camara_status(
    db: Session,
    camara: Camara,
    is_online: bool,
    last_error: str | None,
    last_heartbeat: datetime,
) -> None:
    """
    Actualiza campos de estado sin hacer commit (para usar dentro
    de una transacción mayor, por ejemplo en el heartbeat).
    """
    camara.is_online = 1 if is_online else 0
    camara.last_error = last_error
    camara.last_heartbeat = last_heartbeat
    db.add(camara)
    db.flush()


# -------------------------------------------------
# HISTORIAL (tabla historial)
# -------------------------------------------------
def create_historial_entry(
    db: Session,
    camara_id: int,
    conteo: int,
    indice_aglomeracion: float,
    fecha: datetime | None = None,
) -> Historial:
    """
    Inserta un registro en la tabla historial.
    Si fecha no se entrega, usa hora chile.
    """
    entry = Historial(
        fecha= fecha or datetime.now(ZoneInfo("America/Santiago")),
        camara_id=camara_id,
        conteo=conteo,
        indice_aglomeracion=indice_aglomeracion,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def get_historial_by_camara(
    db: Session,
    camara_id: int,
    limit: int = 200,
    since: datetime | None = None,
) -> List[Historial]:
    """
    Devuelve historial para una cámara, ordenado por fecha DESC.
    """
    stmt = select(Historial).where(Historial.camara_id == camara_id)

    if since is not None:
        stmt = stmt.where(Historial.fecha >= since)

    stmt = stmt.order_by(desc(Historial.fecha)).limit(limit)
    return db.execute(stmt).scalars().all()
