from typing import List
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.documents import Document
from core.state import RetrievedDoc

class VectorDBService:
    def __init__(
        self, 
        db_path: str = "./qdrant_db", 
        collection_name: str = "research_docs",
        # Using a highly efficient, lightweight model
        model_name: str = "BAAI/bge-small-en-v1.5" 
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # It handles the embedding generation locally with high optimization
        self.embeddings = FastEmbedEmbeddings(model_name=model_name)
        self.vectorstore = None

    def initialize_store(self, documents: List[Document] = None):
        """
        Creates a new collection if documents are passed, 
        otherwise loads the existing local database.
        """
        if documents:
            # This line generates embeddings AND stores them in Qdrant
            self.vectorstore = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                path=self.db_path,
                collection_name=self.collection_name,
                retrieval_mode=RetrievalMode.DENSE,
            )
        else:
            self.vectorstore = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                path=self.db_path,
                collection_name=self.collection_name,
                retrieval_mode=RetrievalMode.DENSE,
            )

    async def search_docs(self, query: str, top_k: int = 3):
        """Simple semantic search without reranking."""
        if not self.vectorstore:
            self.initialize_store()
            
        docs = await self.vectorstore.asimilarity_search(query, k=top_k) 
            
        final_output: List[RetrievedDoc] = []
        for i, doc in enumerate(docs):         
            final_output.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "retrieval_score": 0.0,
                "rerank_score": 0.0
            })
        
        print(final_output)
        return final_output