from sqlalchemy import (
    Column, BigInteger, String, Numeric, Enum, Boolean, Integer, SmallInteger,
    DateTime, UniqueConstraint, Index, func
)
from datetime import datetime 
from .database import Base, engine

class Camara(Base):
    __tablename__ = "camaras"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    public_id = Column(String(32), nullable=False)
    name = Column(String(120), nullable=False)
    url = Column(String(600), nullable=False)
    location = Column(String(120), nullable=True)
    latitude = Column(Numeric(9, 6), nullable=True)
    longitude = Column(Numeric(9, 6), nullable=True)
    alert_count_threshold = Column(Integer, nullable=True)
    alert_occ_threshold = Column(Numeric(5, 2), nullable=True)
    status = Column(Enum("ACTIVE", "INACTIVE", name="status_enum"), nullable=False, default="ACTIVE")
    is_online = Column(Boolean, nullable=False, default=False)
    last_heartbeat = Column(DateTime, nullable=True)
    last_error = Column(String(500), nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("public_id", name="ux_camaras_public_id"),
        UniqueConstraint("url", name="ux_camaras_url"),
        Index("ix_camaras_name_location", "name", "location"),
        Index("ix_camaras_is_online", "is_online"),
        Index("ix_camaras_status", "status"),
    )

    def __repr__(self):
        return f"<Camara id={self.id} name={self.name!r} url={self.url!r}>"
    
class Historial(Base):
    __tablename__ = "historial"

    fecha = Column(DateTime, primary_key=True, nullable=False, default=datetime.utcnow)
    camara_id = Column(Integer, primary_key=True, nullable=False)
    conteo = Column(SmallInteger, nullable=False)
    indice_aglomeracion = Column(Numeric(2, 1), nullable=False)
#crea tablas en mysql
if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Tablas creadas correctamente en la bd")
