Prism-Shella-AI is an agentic RAG pipeline that transforms multi-format documents and code files (.py, .js) into an interactive knowledge base with built-in chat memory to remember context across conversations. It implements a native Chonkie processing pipeline, using hierarchical recursive chunking followed by an overlapping context refinery stage, with Qdrant similarity search driving the retrieval. When a user asks a question, an intelligent planner-critic loop dynamically routes the query and double-checks the response against the source files to self-correct mistakes before output. This eliminates file-conversion friction, retains conversation history, and uses autonomous guardrails to deliver accurate, verified answers.

```
                ┌──────────────────────┐
                │   User Upload (PDF)  │
                └─────────┬────────────┘
                          ↓
                ┌──────────────────────┐
                │   Chonkie chunking   │────────────┐
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
