from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import threading, time

from .video_processor import VideoProcessor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Credenciales fijas (login)
USUARIO = "admin"
PASSWORD = "admin"

# ===========================
# Cámara IP Pública fija
# ===========================
CAMERA_PUBLIC_SOURCE = "http://89.104.109.62:8081/mjpg/video.mjpg"
vp = VideoProcessor(source=CAMERA_PUBLIC_SOURCE)

# ===========================
# Endpoints
# ===========================

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """Verifica credenciales fijas"""
    if username == USUARIO and password == PASSWORD:
        return {"status": "ok"}
    raise HTTPException(status_code=401, detail="Acceso denegado: credenciales inválidas")

@app.get("/camera")
def get_camera():
    """Devuelve la URL de la cámara actual"""
    return {"source": CAMERA_PUBLIC_SOURCE}

@app.get("/stream")
def stream():
    """Devuelve el stream procesado con YOLOv8"""
    def generator():
        while True:
            frame = vp.get_frame()
            if not frame:
                time.sleep(0.1)
                continue
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
    return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/count")
def count():
    """Devuelve el conteo actual de personas"""
    return {"count": vp.get_current_count()}

@app.get("/history")
def history():
    """Devuelve el historial de conteos en el tiempo"""
    return JSONResponse(content=vp.get_history())

@app.get("/report")
def report():
    """Genera y entrega un reporte CSV del conteo"""
    csv_path = "app/static/report.csv"
    vp.export_csv(csv_path)
    return FileResponse(csv_path, media_type="text/csv", filename="reporte.csv")

