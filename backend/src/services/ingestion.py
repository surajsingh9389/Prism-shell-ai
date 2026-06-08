from typing import List
from langchain_core.documents import Document

import io
from unstructured.partition.auto import partition

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
    
    async def ingest_and_chunk(self, file_bytes: bytes, file_name: str) -> List[Document]:
        """Pure Python multi-format parser using Unstructured."""
        
        # Wrap bytes in an in-memory stream
        file_stream = io.BytesIO(file_bytes)
        
        # Automatically detect layout elements and parse text
        elements = partition(file=file_stream, file_name=file_name)
        full_text = "\n\n".join([str(el) for el in elements])
        
        if not full_text.strip():
            return []
        
        # Run the extracted text string straight through your Chonkie pipeline wrapper
        pipeline_doc = self.chonkie_pipeline.run(full_text)
        
        print("Recursive chunking")
        print(pipeline_doc)
        
        final_cleaned_chunks = []
        for chunk_counter, chunk in enumerate(pipeline_doc.chunks):
            langchain_doc = Document(
                page_content=chunk.text,
                metadata={
                    "source": file_name,
                    "chunk_id": chunk_counter,
                    "token_count": chunk.token_count
                }
            )
            final_cleaned_chunks.append(langchain_doc)
            
        return final_cleaned_chunks