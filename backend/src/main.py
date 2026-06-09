import traceback

from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Request
from src.engine.data_manager import run_full_ingestion
from fastapi.middleware.cors import CORSMiddleware
from src.engine.graph import get_runtime_graph_with_pool
from src.core.database import db_pool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from fastapi.responses import JSONResponse



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the background connection pools
    print("Connecting to PostgreSQL Connection Pool...")
    await db_pool.open() 
    
    # Open an independent autocommit connection to setup tables cleanly
    # This disables transaction locks, allowing CREATE INDEX CONCURRENTLY to work!

    async with db_pool.connection() as conn:
        await conn.set_autocommit(True) # <--- THIS SAYS: "No transaction wrappers please!"
        print("Verifying LangGraph database tracking schemas...")
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()
    
    yield  # --- FastAPI server runs here ---
    
    print("Closing PostgreSQL Connection Pool safely...")
    await db_pool.close()


app = FastAPI(title="Personal Knowledge Agent", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

origins = [
    "http://localhost:5173",  # Default Vite + React port
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE)
    allow_headers=["*"],  # Allows all headers
)


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # This physically forces the error into your VS Code / PowerShell terminal window
        print("\n" + "="*50 + " DETECTED BACKEND CRASH " + "="*50)
        traceback.print_exc()
        print("="*124 + "\n")
        
        # Returns a clear JSON message back to the frontend instead of an empty 500
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "type": type(e).__name__}
        )

  
 
@app.get('/')
async def health_check():
    return {"status": "ok", "message": "Personal Knowledge Agent is running!"}    

@app.post("/ingest", tags=["Ingestion"])
async def ingest_data(file: UploadFile = File(...), session_id: str = Form(...)):
    # Read the network file stream directly into memory bytes
    file_bytes = await file.read()
       
    print("Session_id", session_id)
        
    print("ingestion started")
    message = await run_full_ingestion(file_bytes, file.filename, session_id=session_id)
    
    return {"status": "success", "message": message}

@app.post("/ask", response_model=QueryResponse, tags=["Agent"])
async def ask_researcher(request: QueryRequest):
    """
    Invoke the LangGraph researcher agent.
    """      
    # Pass that live connection directly to compile the graph instantly
    runnable_agent = get_runtime_graph_with_pool(db_pool)
    
    config = {"configurable": {"thread_id": request.session_id, "session_id": request.session_id}}
    
    inputs = {
    "messages": [("user", request.query)],
    "iteration": 0,
    "max_iterations": 3,
    "thoughts": []
    }
    
    # Run the graph
    final_state = await runnable_agent.ainvoke(inputs, config=config)
    
    # Extract answers and track document source metadata
    answer = final_state.get("current_answer", "No answer generated.")
    retrieved_docs = final_state.get("retrieved_docs", [])
    
    # Safely extract origin file sources if any docs were used
    sources = list(set([doc["source"] for doc in retrieved_docs if "source" in doc]))
    
    return QueryResponse(
        answer=answer,
        sources=sources 
    )

