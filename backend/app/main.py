# backend/app/main.py
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import time, json, os

from .camera_manager import list_cameras, add_camera, remove_camera, get_camera, get_processor

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- LOGIN FIJO ---
USUARIO = "admin"
PASSWORD = "admin"

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username == USUARIO and password == PASSWORD:
        return {"status": "ok"}
    raise HTTPException(status_code=401, detail="Credenciales inválidas")

# --- Cameras CRUD ---
@app.get("/cameras")
def get_cameras():
    return list_cameras()

@app.post("/cameras")
async def post_cameras(payload: list):
    # payload es lista completa desde frontend; reemplaza en disco
    from .camera_manager import _cameras, _processors
    for cam in list(list_cameras()):
        remove_camera(cam['id'])
    for cam in payload:
        add_camera(cam)
    return {"status": "ok", "count": len(payload)}

@app.post("/cameras/add")
async def add_one_camera(cam: dict):
    add_camera(cam)
    return {"status": "ok"}

@app.delete("/cameras/{cam_id}")
def delete_camera(cam_id: str):
    remove_camera(cam_id)
    return {"status": "ok"}

# --- Streaming ---
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
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

# --- Conteo ---
@app.get("/count/{cam_id}")
def count(cam_id: str):
    vp = get_processor(cam_id, create_if_missing=False)
    if vp is None:
        return {"count": 0}
    return {"count": vp.get_current_count()}

# --- Historial ---
@app.get("/history/{cam_id}")
def history(cam_id: str):
    vp = get_processor(cam_id, create_if_missing=False)
    if vp is None:
        return []
    return JSONResponse(content=vp.get_history())

# --- Reporte ---
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


