from pathlib import Path
from typing import List
import asyncio

from langchain_docling.loader import ExportType  
from langchain_core.documents import Document
from docling.chunking import HybridChunker

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain_docling import DoclingLoader


# 1. Initialize the default converter normally
# This populates all the correct backends and pipeline classes automatically
converter = DocumentConverter()

# 2. Update the PDF pipeline options safely in-place
if InputFormat.PDF in converter.format_options:
    # Target the existing pipeline_options directly
    converter.format_options[InputFormat.PDF].pipeline_options.allow_external_plugins = True

# 3. For non-PDF formats (Word, PPTX, Excel, HTML, etc.), handle their options
# Docling uses specific format options (like WordFormatOption) internally
for format_key, format_opt in converter.format_options.items():
    if format_key != InputFormat.PDF and hasattr(format_opt, "pipeline_options"):
        if format_opt.pipeline_options:
            format_opt.pipeline_options.allow_external_plugins = True

class IngestionService:
    def __init__(self):
        pass
    
    async def ingest_and_chunk(self, file_path: str) -> List[Document]:
        """Uses Docling to export tables as Markdown to preserve row relationships."""
        
        # Initialize DoclingLoader with Markdown export enabled
        loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.DOC_CHUNKS, # Tells Docling to handle the chunking
        chunker=HybridChunker(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2", # Matches embedding model
        max_tokens=512,  
        merge_peers=True, # Keeps related list items/table rows together
        document_converter=converter
        )
    )
        
        raw_docs = await loader.aload() 
        
        final_cleaned_chunks = []
        source_name = Path(file_path).name
        
        for idx, doc in enumerate(raw_docs):
            # For small tables, we don't want to split them at all.
            # We want the WHOLE table in one chunk so the LLM sees the header + row.
            
            clean_metadata = {
                "source": source_name,
                "chunk_id": idx,
                "heading": doc.metadata.get("dl_meta", {}).get("headings", ["None"])[-1],
            }
            
            # Use the page_content directly from Docling's Markdown export
            doc.metadata = clean_metadata
            final_cleaned_chunks.append(doc)
        
        return final_cleaned_chunks
