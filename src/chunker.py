from langchain_docling import DoclingLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os
import asyncio

async def ingest_and_chunk_document(file_path: str) -> List[Document]:
    """
    Loads a document using Docling, converts to Markdown, and chunks 
    while preserving header context.
    """
    
    # Load document and convert to Markdown
    # Docling is superior for preserving tables and headers
    loader = DoclingLoader(file_path=file_path)
    raw_docs = await loader.aload() # Usually returns 1 large Doc per file
    
    
    print(raw_docs)
    print("-"*50)
    print('\n\n')
    
    # Split by Markdown Headers
    # This ensures a chunk doesn't mix "Introduction" with "Conclusion"
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Keep the '#' symbols for the LLM to see structure
    )
    
    # Hold chunks from ALL raw documents
    all_semantic_chunks = []
    
    # Iterate through every document found by Docling
    for doc in raw_docs:
        # Split this specific document
        chunks = header_splitter.split_text(doc.page_content)
        # Add these chunks to our master list
        all_semantic_chunks.extend(chunks)

    
    print(all_semantic_chunks)
    print("-"*50)
    print('\n\n')
    
    # Sub-split large sections
    # If a section is 5000 words, we break it down further
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    final_chunks = text_splitter.split_documents(all_semantic_chunks)
    
    # Add the filename to metadata for your source tracking
    for chunk in final_chunks:
        chunk.metadata["file_name"] = os.path.basename(file_path)
        
    print(final_chunks)
    
    return final_chunks


# 3. Run the async loop
if __name__ == "__main__":
    asyncio.run(ingest_and_chunk_document("raw_data.txt"))