from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from sqlalchemy.exc import IntegrityError
import time
from pathlib import Path
from sqlalchemy import text 
from app.config import CORS_ORIGINS
from app.background_task import start_heartbeat_thread
from app.database.database import SessionLocal
from app.database import crud, schemas
from app.auth import router as auth_router, admin_required
from .camera_manager import list_cameras, add_camera, remove_camera, get_processor

app = FastAPI(title="Monitor Inteligente API", version="0.1.0")

#revisa camaras cada 30 seg
start_heartbeat_thread()


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

#devuelve token
app.include_router(auth_router, prefix="/auth", tags=["auth"])


#camaras (lectura)
@app.get("/cameras")
def get_cameras():
    return list_cameras()

#camaras 
@app.post("/cameras")
async def post_cameras(payload: list, user=Depends(admin_required)):
    """
    Reemplaza el set completo de cámaras con el payload recibido.
    (Destructivo) Elimina todas y vuelve a agregar.
    """
    # borrar todas las existentes (por su public_id)
    for cam in list(list_cameras()):
        remove_camera(cam["id"])
    # agregar las nuevas
    for cam in payload:
        add_camera(cam)
    return {"status": "ok", "count": len(payload)}

@app.post("/cameras/add")
async def add_one_camera(cam: dict, user=Depends(admin_required)):
    add_camera(cam)
    return {"status": "ok"}

@app.delete("/cameras/{cam_id}")
def delete_camera(cam_id: str, user=Depends(admin_required)):
    remove_camera(cam_id)
    return {"status": "ok"}

#streaming
@app.get("/stream/{cam_id}")
def stream(cam_id: str):
    vp = get_processor(cam_id)
    if vp is None:
        raise HTTPException(status_code=404, detail="Cámara no disponible")

    def gen():
        while True:
            frame = vp.get_frame()
            if not frame:
                time.sleep(0.1)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/count/{cam_id}")
def count(cam_id: str):
    vp = get_processor(cam_id, create_if_missing=False)
    if vp is None:
        return {"count": 0}
    return {"count": vp.get_current_count()}

@app.get("/history/{cam_id}")
def history(cam_id: str):
    vp = get_processor(cam_id, create_if_missing=False)
    if vp is None:
        return []
    return JSONResponse(content=vp.get_history())

@app.get("/report/{cam_id}")
def report(cam_id: str):
    vp = get_processor(cam_id, create_if_missing=False)
    if vp is None:
        raise HTTPException(status_code=404)
    csv_path = f"app/static/report_{cam_id}.csv"
    vp.export_csv(csv_path)
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=f"reporte_{cam_id}.csv"
    )

#sync
@app.post("/camaras/sync-file")
def sync_camaras_from_file(user=Depends(admin_required)):
    """
    Upsert desde backend/app/database/camaras.json sin borrar otras cámaras.
    Devuelve cuántas insertó y cuántas actualizó.
    """
    json_path = Path(__file__).parent  / "database" / "cameras.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="No se encontró cameras.json en app/database")

    db = SessionLocal()
    try:
        inserted, updated = crud.sync_camaras_from_file(db, json_path)
        db.commit()
        return {"inserted": inserted, "updated": updated}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"Integridad (duplicados): {getattr(e, 'orig', e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/camaras/sync")
def sync_camaras_from_payload(payload: list[dict], user=Depends(admin_required)):
    """
    Upsert de una lista de cámaras enviada por el frontend, sin borrar las demás.
    Devuelve insertadas/actualizadas.
    """
    db = SessionLocal()
    inserted = updated = 0
    try:
        for item in payload:
            j = schemas.CamaraFromJSON.model_validate(item)  # valida/normaliza
            _, action = crud.upsert_camara_from_json(db, j)
            if action == "inserted":
                inserted += 1
            else:
                updated += 1
        db.commit()
        return {"inserted": inserted, "updated": updated}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"Integridad (duplicados): {getattr(e, 'orig', e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()



@app.get("/health")
def health_check():
    """
    Verifica conexión con la base de datos y devuelve estado general.
    """
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))  # ← cambia esta línea
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": str(e)}
    finally:
        db.close()


