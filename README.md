**Prism-Shella-AI** is an agentic RAG pipeline that transforms multi-format documents and code files (`.py`, `.js`) into an interactive knowledge base with built-in chat memory to remember context across conversations. It uses Docling for layout-aware parsing and Qdrant to handle semantic search. When a user asks a question, an intelligent planner-critic loop dynamically routes the query and double-checks the response against the source files to self-correct mistakes before output. This is highly helpful because it eliminates file-conversion friction, retains conversation history, and uses autonomous guardrails to deliver accurate, verified answers.


```
                ┌──────────────────────┐
                │   User Upload (PDF)  │
                └─────────┬────────────┘
                          ↓
                ┌──────────────────────┐
                │   Docling chunking   │────────────┐
                │                      │            |   
                └─────────┬────────────┘            |  
                          ↓                         | ───────────────→ Ingestion pipeline
                ┌──────────────────────┐            |
                │       Qdrant         │            |
                │      vector-store    │            |
                └─────────┬────────────┘────────────┘


User Query ───────────────→ Planner Node
                             ↓
        ┌────────────────────┼────────────────────┐
        ↓                                         ↓
Direct Answer Path                    Retrieval Path
(no docs needed)                      (docs needed, qdrant similarity search)
        ↓                                         ↓
        └──────────────→ Generator Node ←─────────┘
                                ↓
                         Critic Node
                                ↓
                    (loop if needed)
                                ↓
                             FINAL

```
