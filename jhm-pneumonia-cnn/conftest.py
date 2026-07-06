import os

# Garantiza que las variables de entorno estén listas antes de cualquier import de api.main
# Esto evita que database.py intente conectar a PostgreSQL durante los tests
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("TESTING", "true")
