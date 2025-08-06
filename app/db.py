from sqlalchemy import create_engine, MetaData
from databases import Database
import os



# Load from environment or hardcode for now
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:Naruto@123@localhost/sanhri")

# Async database connection (for FastAPI)
database = Database(DATABASE_URL)

# SQLAlchemy metadata for models
metadata = MetaData()

# Optional: Engine for migrations (non-async)
engine = create_engine(DATABASE_URL.replace('+asyncpg', ''))
