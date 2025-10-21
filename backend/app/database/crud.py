from __future__ import annotations
from typing import Iterable, Optional, Tuple, List
from pathlib import Path
import json
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .database import SessionLocal, engine, Base
from .models import Camara
from . import schemas


def get_session() -> Session:
    return SessionLocal()


#crud 
def create_camara(db: Session, data: schemas.CamaraCreate) -> Camara:
    #crea camara nueva
    cam = Camara(
        public_id=data.public_id,
        name=data.name,
        url=str(data.url),
        location=data.location,
        latitude=data.latitude,
        longitude=data.longitude,
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
    stmt = select(Camara)
    if q:
        #busca por nombre o location
        like = f"%{q}%"
        stmt = stmt.where((Camara.name.ilike(like)) | (Camara.location.ilike(like)))
    if status:
        stmt = stmt.where(Camara.status == status)
    if is_online is not None:
        stmt = stmt.where(Camara.is_online == is_online)
    stmt = stmt.offset(offset).limit(limit)
    return db.execute(stmt).scalars().all()


def update_camara(db: Session, cam: Camara, changes: schemas.CamaraUpdate) -> Camara:
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

    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


def delete_camara(db: Session, cam: Camara) -> None:
    db.delete(cam)
    db.commit()


#inserta o actualiza una camara segun objeto de json
def upsert_camara_from_json(db: Session, j: schemas.CamaraFromJSON) -> Tuple[Camara, str]:
    lat, lon = j.lat_lon()
    existing = get_camara_by_public_id(db, j.id) or get_camara_by_url(db, str(j.url))

    if existing:
        changed = schemas.CamaraUpdate(
            public_id=j.id,
            name=j.name,
            url=j.url,
            location=j.location,
            latitude=lat,
            longitude=lon,
        )
        cam = update_camara(db, existing, changed)
        return cam, "updated"

    #crea nueva
    new_data = schemas.CamaraCreate(
        public_id=j.id,
        name=j.name,
        url=j.url,
        location=j.location,
        latitude=lat,
        longitude=lon,
    )
    cam = create_camara(db, new_data)
    return cam, "inserted"


def load_camaras_json(file_path: Path) -> Iterable[schemas.CamaraFromJSON]:
    #lee un archivo JSON como lista de camaras y valida cada item
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    for item in raw:
        yield schemas.CamaraFromJSON.model_validate(item)


def sync_camaras_from_file(db: Session, file_path: Path) -> Tuple[int, int]:

    #sincroniza todas las cámaras del json contra la bd
   
    inserted = updated = 0
    for cam_json in load_camaras_json(file_path):
        _, action = upsert_camara_from_json(db, cam_json)
        if action == "inserted":
            inserted += 1
        else:
            updated += 1
    return inserted, updated


#para ejecutar -m app.database.crud
if __name__ == "__main__":

    #asegura si las tablas existen 
    Base.metadata.create_all(bind=engine)

    json_path = Path(__file__).with_name("cameras.json")
    if not json_path.exists():
        print(f"No se encontró {json_path}")
        raise SystemExit(1)

    db = get_session()
    try:
        ins, upd = sync_camaras_from_file(db, json_path)
        print(f"Sincronización completada. Insertadas: {ins} · Actualizadas: {upd}")
    except IntegrityError as e:
        db.rollback()
        print("Error:", e)
    except Exception as e:
        db.rollback()
        print("Error:", e)
    finally:
        db.close()
        
#actualiza el estado de las camaras
def update_camara_status(db, camara, is_online: bool, last_error: str | None, last_heartbeat):
    camara.is_online = 1 if is_online else 0
    camara.last_error = last_error
    camara.last_heartbeat = last_heartbeat
    db.add(camara)
    db.flush()
