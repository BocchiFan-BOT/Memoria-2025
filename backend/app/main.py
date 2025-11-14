from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import csv
from pathlib import Path

from app.config import CORS_ORIGINS
from app.background_task import start_heartbeat_thread
from app.database.database import SessionLocal
from app.database import crud
from app.auth import router as auth_router, admin_required

from app.database.schemas import CamaraUpdate

from .camera_manager import (
    list_cameras,
    add_camera,
    remove_camera,
    get_processor,
    get_alerts,
    get_metrics_all,
    get_metrics_one,
)

app = FastAPI(title="Monitor Inteligente API", version="0.1.0")

# Heartbeat
start_heartbeat_thread()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Login
app.include_router(auth_router, prefix="/auth", tags=["auth"])


# =====================================================
# CÁMARAS (CRUD BÁSICO)
# =====================================================
@app.get("/cameras")
def get_cameras():
    return list_cameras()


@app.post("/cameras/add")
async def add_one_camera(cam: dict, user=Depends(admin_required)):
    add_camera(cam)
    return {"status": "ok"}



@app.put("/cameras/{cam_id}")
async def update_camera(cam_id: str, payload: CamaraUpdate, user=Depends(admin_required)):
    db = SessionLocal()
    try:
        cam = crud.get_camara_by_public_id(db, cam_id)
        if not cam:
            raise HTTPException(404, "Cámara no encontrada")

        updated = crud.update_camara(db, cam, payload)
        from .camera_manager import get_processor
        vp = get_processor(cam_id, create_if_missing=True)
        vp.set_alert_thresholds(
            count_threshold=updated.alert_count_threshold,
            occ_threshold=updated.alert_occ_threshold,
        )

        return {"status": "ok"}

    finally:
        db.close()


@app.delete("/cameras/{cam_id}")
def delete_camera(cam_id: str, user=Depends(admin_required)):
    remove_camera(cam_id)
    return {"status": "ok"}


# =====================================================
# STREAM + COUNT
# =====================================================
@app.get("/stream/{cam_id}")
def stream(cam_id: str, fps: int = 8):
    vp = get_processor(cam_id)
    if vp is None:
        raise HTTPException(status_code=404, detail="Cámara no disponible")

    def gen():
        target_dt = 1.0 / max(1, min(60, fps))
        last_sent = 0.0
        while True:
            now = time.time()
            if now - last_sent < target_dt:
                time.sleep(0.005)
                continue

            frame = vp.get_frame()
            if not frame:
                time.sleep(0.05)
                continue

            last_sent = time.time()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


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
    return vp.get_history()

@app.get("/report/{cam_id}")
def report(cam_id: str):
    db = SessionLocal()
    try:
        cam = crud.get_camara_by_public_id(db, cam_id)
        if not cam:
            raise HTTPException(404, "Cámara no encontrada")

        rows = crud.get_historial_by_camara(db, camara_id=cam.id, limit=10000)

        out_path = Path("app/static") / f"reporte_{cam_id}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fecha", "conteo", "indice_aglomeracion"])
            for h in rows:
                writer.writerow([
                    h.fecha.isoformat(),
                    int(h.conteo),
                    float(h.indice_aglomeracion)
                ])

        return FileResponse(
            str(out_path),
            media_type="text/csv",
            filename=f"reporte_{cam_id}.csv"
        )
    finally:
        db.close()


# =====================================================
# ALERTAS + MÉTRICAS
# =====================================================
@app.get("/alerts")
def alerts(since: float | None = None, cam_id: str | None = None):
    return get_alerts(since=since, cam_id=cam_id)


@app.get("/metrics")
def metrics():
    return get_metrics_all()


@app.get("/metrics/{cam_id}")
def metrics_one(cam_id: str):
    m = get_metrics_one(cam_id)
    if not m:
        raise HTTPException(404, "Cámara no disponible")
    return m