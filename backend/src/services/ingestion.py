from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Import Chonkie's modular pipeline constructs
from chonkie import Pipeline

class IngestionService:
    def __init__(self):
        # Configured with a valid Hugging Face string. 
        # Chonkie only pulls down the vocabulary ruleset, keeping it light and stable.
        self.chonkie_pipeline = (
            Pipeline()
            .chunk_with("recursive", tokenizer="sentence-transformers/all-MiniLM-L6-v2", chunk_size=512)
            .refine_with("overlap", context_size=64)
        )
    
    async def ingest_and_chunk(self, file_path: str) -> List[Document]:
        """Parses PDFs via pypdf and applies structural chunking with overlapping context loops."""
        source_name = Path(file_path).name
        
        loader = PyPDFLoader(file_path)
        raw_pages = await loader.aload()
            
        final_cleaned_chunks = []
        chunk_counter = 0
        
        for page in raw_pages:
            page_text = page.page_content
            page_num = page.metadata.get("page", 0) + 1
            
            # 1. Run the text through the pipeline wrapper
            pipeline_doc = self.chonkie_pipeline.run(page_text)
            
            # FIX: Loop through pipeline_doc.chunks instead of trying to iterate the doc directly
            for chunk in pipeline_doc.chunks:
                clean_metadata = {
                    "source": source_name,
                    "chunk_id": chunk_counter,
                    "page": page_num,
                    "token_count": chunk.token_count
                }
                
                langchain_doc = Document(
                    page_content=chunk.text,
                    metadata=clean_metadata
                )
                final_cleaned_chunks.append(langchain_doc)
                chunk_counter += 1
                
        print(f"Ingestion complete. Created {final_cleaned_chunks} context-overlapped chunks.")
        return final_cleaned_chunks