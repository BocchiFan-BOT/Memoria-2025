from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session

from app.database.database import SessionLocal

def get_db() -> Generator[Session, None, None]:

    #abre la sesión, la entrega a la ruta y la cierra al finalizar
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
