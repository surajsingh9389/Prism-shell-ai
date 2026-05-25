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