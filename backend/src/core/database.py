# src/core/database.py
import os
from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool

load_dotenv()
DB_URI = os.getenv("DATABASE_URL")

# Network level pings to keep the connection alive on firewalls/cloud DBs
connection_kwargs = {
    "keepalives": 1,
    "keepalives_idle": 60,       # Ping after 60 seconds of complete silence
    "keepalives_interval": 10,   # Retry every 10 seconds if a ping drops
    "keepalives_count": 3        # Drop connection only after 3 failed pings
}

# The single global connection pool configuration instance
db_pool = AsyncConnectionPool(
    conninfo=DB_URI, 
    max_size=10, 
    open=False, # Handled elegantly by the FastAPI lifespan manager
    kwargs=connection_kwargs,
    
    # CRITICAL FIX: Verifies connections aren't dead before giving them to LangGraph
    check=AsyncConnectionPool.check_connection
)