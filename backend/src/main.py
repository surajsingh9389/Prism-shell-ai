from src.engine.graph import graph

import os
import asyncio
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.engine.data_manager import run_full_ingestion
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Personal Knowledge Agent")

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

# --- STARTUP LIFESPAN EVENT ---
@app.on_event("startup")
async def startup_event():
    """
    Runs automatically on server startup. Initializes the Neon database tables 
    for LangGraph if they don't already exist.
    """
    try:
        # Securely sets up your checkpoints schema inside Neon once
        graph.checkpointer.setup()
        print("Successfully connected to Neon Postgres and verified checkpoint tables.")
    except Exception as e:
        print(f"Failed to initialize checkpointer schema on database: {e}")

  
 
@app.get('/')
async def health_check():
    return {"status": "ok", "message": "Personal Knowledge Agent is running!"}    

@app.post("/ingest", tags=["Ingestion"])
async def ingest_data(file: UploadFile = File(...), session_id: str = Form(...)):
    """
    Step 1: Save file locally, Chunk with Docling, and Upsert to Qdrant.
    """
    temp_path = f"temp_{session_id}_{file.filename}"
    try:
        # Save temp file
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        message = await run_full_ingestion(temp_path, session_id=session_id)
        
        return {"status": "success", "message": message}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask", response_model=QueryResponse, tags=["Agent"])
async def ask_researcher(request: QueryRequest):
    """
    Step 2: Invoke the LangGraph researcher agent.
    """
    try:
        # Initial state for your LangGraph
        
        config = {"configurable": {"thread_id": request.session_id}}
        
        inputs = {
        "messages": [("user", request.query)],
        "iteration": 0,
        "max_iterations": 3,
        "thoughts": []
        }
        
        # Run the graph
        final_state = await graph.ainvoke(inputs, config=config)
        
        # Extract answers and track document source metadata
        answer = final_state.get("current_answer", "No answer generated.")
        retrieved_docs = final_state.get("retrieved_docs", [])
        
        # Safely extract origin file sources if any docs were used
        sources = list(set([doc["source"] for doc in retrieved_docs if "source" in doc]))
        
        return QueryResponse(
            answer=answer,
            sources=sources 
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)

# async def main():
#     # 1. POPULATE THE BRAIN
#     print("--- Phase 1: Ingesting Data ---")
#     file_to_process = "raw_data.txt"
#     await run_full_ingestion(file_to_process)
    
#     # 2. RUN THE AGENT
#     print("\n--- Phase 2: Running Agentic Research ---")
#     inputs = {
        
#         "query": "summarize my document",
#         "iteration": 0,
#         "max_iterations": 3,
#         "thoughts": []
#     }
    
#     # Use astream for that 2026 "live" feel
#     async for event in graph.astream(inputs):
#         for node_name, state_update in event.items():
#             print(f"[{node_name}] is processing...")
#             if "current_answer" in state_update:
#                 print(f"Result: {state_update['current_answer']}")

# if __name__ == "__main__":
#     asyncio.run(main())