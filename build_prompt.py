def build_prompt(query: str, context: str) -> str:
    prompt = f"""
    Answer the question ONLY using the context below.
    
    Context:
    {context}
    
    Question:
    {query}
    
    If answer not found, say "I don't know".
    """
    
    return prompt.strip()