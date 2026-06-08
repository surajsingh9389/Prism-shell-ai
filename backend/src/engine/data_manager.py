import asyncio
from src.services.ingestion import IngestionService
from src.services.vector_db import VectorDBService

# Create single instances to be shared across the app
ingestor = IngestionService()
vector_db = VectorDBService()

async def run_full_ingestion(file_bytes: bytes, file_name: str, session_id: str):
    """
    Run the full ingestion pipeline.
    """
    # 1. Chunker
    print("chunking stated.")
    chunks = await ingestor.ingest_and_chunk(file_bytes, file_name)
    
    # 2. Vector DB
    vector_db.upload_documents(chunks, session_id=session_id)
    
    return f"Processed {len(chunks)} chunks from {file_name}"
