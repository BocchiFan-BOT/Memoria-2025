from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from app.config import ADMIN_USER, ADMIN_PASS, JWT_SECRET, JWT_ALG, JWT_EXPIRE_MIN

router = APIRouter()

security = HTTPBearer(auto_error=True)

def create_access_token(data: dict, expires_minutes: int = JWT_EXPIRE_MIN) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido o expirado")

def admin_required(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    payload = decode_token(token)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Permisos insuficientes")
    return payload

@router.post("/login")
def login(body: dict):

    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()

    if username == ADMIN_USER and password == ADMIN_PASS:
        token = create_access_token({"sub": username, "role": "admin"})
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": JWT_EXPIRE_MIN
        }

    raise HTTPException(status_code=401, detail="Credenciales inválidas")
