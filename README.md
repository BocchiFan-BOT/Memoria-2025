# Monitor inteligente detector de aglomeraciones 
### FastAPI + YOLOv8 + React

Sistema de monitoreo que detecta y cuenta personas en tiempo real usando visión por computador, integrado con una interfaz web para visualizar cámaras, conteos y reportes históricos.  


## Requisitos
- Python 3.9+
- Node 16+
- Git

## Backend (FastAPI)
```bash
cd backend
# activar venv (PowerShell)
.\venv\Scripts\Activate.ps1
# o CMD:
# backend\venv\Scripts\activate.bat

uvicorn app.main:app --reload --port 8000
```
## Frontend (React)
```bash
cd frontend
npm install
npm start
```
## Login
Usuario: admin
Contraseña: admin

## Estructura del proyecto 

├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── camera_manager.py
│   │   ├── video_processor.py
│   │   └── database/
│   │       ├── database.py
│   │       ├── models.py
│   │       ├── schemas.py
│   │       ├── crud.py
│   │       └── cameras.json
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
│
└── README.md
