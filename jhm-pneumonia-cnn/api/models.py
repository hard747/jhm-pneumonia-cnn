from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from api.database import Base


class AuditoriaDiagnostico(Base):
    __tablename__ = "auditorias_diagnostico"

    id = Column(Integer, primary_key=True, index=True)
    nombre_imagen = Column(String(255), nullable=False)
    resultado = Column(String(50), nullable=False)
    confianza = Column(Float, nullable=False)
    latencia_ms = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    fecha_analisis = Column(DateTime, default=datetime.utcnow)


class ModelRegistry(Base):
    """Registro de versiones del modelo (MLOps - Model Registry)."""
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), nullable=False, unique=True)
    accuracy = Column(Float, nullable=True)
    is_active = Column(Boolean, default=False)
    deployed_at = Column(DateTime, default=datetime.utcnow)
    description = Column(Text, nullable=True)
    file_path = Column(String(500), nullable=True)
