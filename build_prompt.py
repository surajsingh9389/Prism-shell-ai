def build_prompt(query: str, context: str) -> str:
    prompt = f"""
    You must answer the question using ONLY the provided context.

    STRICT RULES:
    - Do NOT use outside knowledge
    - If the answer is not explicitly in the context, say "I don't know"
    - Always cite the source in your answer
    
    Context:
    {context}
    
    Question:
    {query}
    
    If answer not found, say "I don't know".
    """
    
    return prompt.strip()