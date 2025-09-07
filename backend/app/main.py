# backend/app/main.py
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import io, os

from .video_processor import VideoProcessor

app = FastAPI()
# CORS para React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)

# credenciales fijas
USUARIO = "admin"
PASSWORD = "admin"

# iniciar video
vp = VideoProcessor(path="app/static/sample.mp4", conf=0.5, imgsz=640)

# --- RUTAS ---
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username == USUARIO and password == PASSWORD:
        return {"status": "ok"}
    raise HTTPException(status_code=401, detail="Credenciales inválidas")

@app.get("/stream")
def stream():
    def generator():
        while True:
            frame = vp.get_frame()
            if not frame:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/count")
def count():
    return {"count": vp.get_current_count()}

@app.get("/history")
def history():
    return JSONResponse(content=vp.get_history())

@app.get("/report")
def report():
    csv_path = "app/static/report.csv"
    vp.export_csv(csv_path)
    return FileResponse(csv_path, media_type="text/csv", filename="reporte.csv")
