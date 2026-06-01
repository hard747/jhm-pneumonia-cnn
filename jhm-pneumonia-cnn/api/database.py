import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# quote_plus encode caracteres especiales como # @ en la contraseña
_user     = os.getenv("DB_USER",    "postgres")
_password = quote_plus(os.getenv("DB_PASSWORD", ""))
_host     = os.getenv("DB_HOST",    "db")
_port     = os.getenv("DB_PORT",    "5432")
_name     = os.getenv("DB_NAME",    "postgres")
_ssl      = os.getenv("DB_SSLMODE", "disable")

DATABASE_URL = f"postgresql+psycopg2://{_user}:{_password}@{_host}:{_port}/{_name}?sslmode={_ssl}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
