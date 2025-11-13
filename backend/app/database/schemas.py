from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, AnyUrl, Field, field_validator


class CamaraBase(BaseModel):
    public_id: str = Field(..., max_length=32)
    name: str = Field(..., max_length=120)
    url: AnyUrl  # http/https/rtsp según soporte de tu versión
    location: Optional[str] = Field(None, max_length=120)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    alert_count_threshold: Optional[int] = None
    alert_occ_threshold: Optional[float] = None

    # normaliza espacios en blanco
    @field_validator("public_id", "name", "location", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v


# crea una cámara cuando se agrega
class CamaraCreate(CamaraBase):
    # status e is_online los define el servidor por defecto
    pass


# actualiza
class CamaraUpdate(BaseModel):
    public_id: Optional[str] = Field(None, max_length=32)
    name: Optional[str] = Field(None, max_length=120)
    url: Optional[AnyUrl] = None
    location: Optional[str] = Field(None, max_length=120)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    alert_count_threshold: Optional[int] = None
    alert_occ_threshold: Optional[float] = None
    status: Optional[Literal["ACTIVE", "INACTIVE"]] = None
    is_online: Optional[bool] = None

    @field_validator("public_id", "name", "location", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v


# salida
class CamaraOut(BaseModel):
    id: int
    public_id: str
    name: str
    url: AnyUrl
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    alert_count_threshold: Optional[int] = None
    alert_occ_threshold: Optional[float] = None
    status: Literal["ACTIVE", "INACTIVE"]
    is_online: bool
    last_heartbeat: Optional[datetime] = None
    last_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ==== HISTORIAL ====

class HistorialBase(BaseModel):
    fecha: datetime
    camara_id: int
    conteo: int
    indice_aglomeracion: float


class HistorialOut(HistorialBase):
    model_config = {"from_attributes": True}
