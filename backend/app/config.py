import os
from pathlib import Path
from dotenv import load_dotenv


# Carga variables desde .env (preferido) y, si no existe, intenta con "env"
BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BACKEND_DIR / ".env"
ENV_ALT_PATH = BACKEND_DIR / "env"

# Cargar primero .env si existe
if ENV_PATH.exists():
	load_dotenv(ENV_PATH)

# Si no hay .env o faltan variables, intenta cargar "env"
if ENV_ALT_PATH.exists():
	load_dotenv(ENV_ALT_PATH, override=False)


DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost:3306/monitor_db")

#lista
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin")

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "60"))

# Alertas
ALERT_COUNT_THRESHOLD = int(os.getenv("ALERT_COUNT_THRESHOLD", "50"))
ALERT_OCC_THRESHOLD = float(os.getenv("ALERT_OCC_THRESHOLD", "10"))  # porcentaje 0-100
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "10"))

