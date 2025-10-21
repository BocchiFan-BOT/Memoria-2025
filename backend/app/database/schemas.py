from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, AnyUrl, Field, field_validator



class CamaraBase(BaseModel):
    public_id: str = Field(..., max_length=32)
    name: str = Field(..., max_length=120)
    url: AnyUrl  # valida que sea URL (http/https/rtsp si usas pydantic v2.7+)
    location: Optional[str] = Field(None, max_length=120)
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    #normaliza espacios en blanco
    @field_validator("public_id", "name", "location", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

#crea una camara cuando se agrega
class CamaraCreate(CamaraBase):
    # status e is_online los define el servidor por defecto
    pass

#actualiza
class CamaraUpdate(BaseModel):
    public_id: Optional[str] = Field(None, max_length=32)
    name: Optional[str] = Field(None, max_length=120)
    url: Optional[AnyUrl] = None
    location: Optional[str] = Field(None, max_length=120)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: Optional[Literal["ACTIVE", "INACTIVE"]] = None
    is_online: Optional[bool] = None

    @field_validator("public_id", "name", "location", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v


#salida
class CamaraOut(BaseModel):
    id: int
    public_id: str
    name: str
    url: AnyUrl
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: Literal["ACTIVE", "INACTIVE"]
    is_online: bool
    last_heartbeat: Optional[datetime] = None
    last_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


    model_config = {"from_attributes": True}


#para importar del json og
class CamaraFromJSON(BaseModel):
    """
    {
      "id": "1759349745495",
      "name": "Panama, Hospital",
      "url": "http://200.46.196.243/mjpg/video.mjpg",
      "location": "Panama",
      "coordinates": "8.993600, -79.519730"
    }
    """
    id: str
    name: str
    url: AnyUrl
    location: Optional[str] = None
    coordinates: Optional[str] = None

    #para extraer latitud y longitud segun las coordenadas
    def lat_lon(self) -> tuple[Optional[float], Optional[float]]:
        if not self.coordinates:
            return None, None
        try:
            lat_str, lon_str = self.coordinates.split(",")
            return float(lat_str.strip()), float(lon_str.strip())
        except Exception:
            return None, None
