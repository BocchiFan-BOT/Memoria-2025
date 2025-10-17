import threading
import time
import datetime
import cv2
import logging
from sqlalchemy.orm import Session
from app.database.database import SessionLocal
from app.database import crud

logger = logging.getLogger("Heartbeat")

def check_camera_status(cam, db: Session):
    #abre stream para ver si responde
    url = cam.url
    try:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            crud.update_camara_status(
                db=db,
                camara=cam,
                is_online=True,
                last_error=None,
                last_heartbeat=datetime.datetime.utcnow(),
            )
            logger.info(f"[OK] Cámara en línea: {cam.name}")
        else:
            crud.update_camara_status(
                db=db,
                camara=cam,
                is_online=False,
                last_error="No se pudo abrir el stream",
                last_heartbeat=None,
            )
            logger.warning(f"Cámara caída: {cam.name}")
        cap.release()
    except Exception as e:
        crud.update_camara_status(
            db=db,
            camara=cam,
            is_online=False,
            last_error=str(e),
            last_heartbeat=None,
        )
        logger.error(f"Error de cámara {cam.name}: {e}")

def heartbeat_loop(interval: int = 30):
    #revisa periodicamente el estado de las camaras
    while True:
        db = SessionLocal()
        try:
            cameras = crud.list_camaras(db)
            for cam in cameras:
                check_camera_status(cam, db)
            db.commit()
        except Exception as e:
            logger.exception(f"Error: {e}")
        finally:
            db.close()
        time.sleep(interval)

def start_heartbeat_thread():
    #revisa cada 30 seg
    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    logger.info("Revisión iniciada")
