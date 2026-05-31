import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 🧠 Intenta leer la URL del archivo .env. Si por alguna razón no la encuentra,
# usa por defecto ("fallback") la ruta local de Docker para que nunca falle en tu casa.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://jhm_user:jhm_password@db:5432/jhm_pneumonia_auditoria"
)

# pool_pre_ping=True verifica que la conexión con Postgres esté viva antes de hacer consultas
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()