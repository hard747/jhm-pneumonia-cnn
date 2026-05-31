from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from api.database import Base

class AuditoriaDiagnostico(Base):
    __tablename__ = "auditorias_diagnostico"

    id = Column(Integer, primary_key=True, index=True)
    nombre_imagen = Column(String(255), nullable=False)
    resultado = Column(String(50), nullable=False)
    confianza = Column(Float, nullable=False)
    fecha_analisis = Column(DateTime, default=datetime.utcnow)
