from typing import List, Optional
import os

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.documents import Document
from src.core.state import RetrievedDoc
from qdrant_client import models
# import asyncio


class VectorDBService:
    def __init__(
        self,  
        collection_name: str = "research_docs",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2" 
    ):
        self.collection_name = collection_name
        
        # It handles the embedding generation locally with high optimization
        self.embeddings = FastEmbedEmbeddings(model_name=model_name, )
        self.vectorstore = None
        
        # Determine the environment mode: "local" or "cloud" (defaults to local)
        self.mode = os.getenv("QDRANT_MODE", "local").lower()
        
        # Path configuration for local mode
        self.db_path = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_db")
        
        # Credentials configuration for cloud mode
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Automatically connect to the selected environment
        self.initialize_store()
        
        
    def _get_connection_kwargs(self) -> dict:
        """Helper to generate appropriate arguments based on the selected mode."""
        if self.mode == "cloud":
            if not self.qdrant_url or not self.qdrant_api_key:
                raise ValueError("Cloud mode active but QDRANT_URL or QDRANT_API_KEY is missing in env.")
            return {
                "url": self.qdrant_url,
                "api_key": self.qdrant_api_key,
                "prefer_grpc": True  # Optimizes cloud streaming speeds
            }
        else:
            # Local Mode connection arguments
            return {
                "path": self.db_path
            }
            
    def _is_local_db_empty(self) -> bool:
        """Checks if the local DB storage directory contains active files."""
        return not (os.path.exists(self.db_path) and os.listdir(self.db_path))

    def initialize_store(self):
        """
        Connects to Qdrant seamlessly. Handles initialization schemas for both
        local environments and remote cloud clusters safely.
        """
        connection_args = self._get_connection_kwargs()
        
        # Determine if we should attempt loading an existing collection first
        should_load_existing = True
        if self.mode == "local" and self._is_local_db_empty():
            should_load_existing = False

        if should_load_existing:
            try:
                self.vectorstore = QdrantVectorStore.from_existing_collection(
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    retrieval_mode=RetrievalMode.DENSE,
                    **connection_args
                )
                print(f"Successfully connected to existing Qdrant collection in [{self.mode.upper()}] mode.")
                return
            except Exception as e:
                # If cloud collection does not exist yet, fallback to creation below
                if self.mode == "local":
                    raise e

        # Fallback: Initialize a brand new clean schema context
        self.vectorstore = QdrantVectorStore.from_documents(
            documents=[],  # Created empty to avoid collisions/locks
            embedding=self.embeddings,
            collection_name=self.collection_name,
            retrieval_mode=RetrievalMode.DENSE,
            **connection_args
        )
        print(f"Initialized a brand new Qdrant collection schema in [{self.mode.upper()}] mode.")
        
    def upload_documents(self, documents: List[Document], session_id: str):
        """Appends new embeddings safely to either environment choice."""
        if not self.vectorstore:
            self.initialize_store()
            
        if documents:
            # Inject session_id into every document metadata block before upserting
            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["session_id"] = session_id
            self.vectorstore.add_documents(documents=documents)
            

    async def search_docs(self, query: str, session_id: str, top_k: int = 3):
        """Simple semantic search without reranking."""
        if not self.vectorstore:
            self.initialize_store()
            
        session_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.session_id",  # LangChain nests metadata dictionaries under this path in Qdrant
                    match=models.MatchValue(value=session_id),
                )
            ]
        )
            
        docs_with_scores = await self.vectorstore.asimilarity_search_with_score(query, k=top_k, filter=session_filter) 
            
        final_output: List[RetrievedDoc] = []
        for doc, score in docs_with_scores:         
            final_output.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "session_id": doc.metadata.get("session_id"),
                "retrieval_score": float(score),    
                "rerank_score": 0.0
            })
        
        return final_output
    
    
    