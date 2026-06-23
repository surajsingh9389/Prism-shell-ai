from langchain_core.prompts import ChatPromptTemplate

#  Generator Prompt
GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You must answer the question using ONLY the provided context. "
               "STRICT RULES: Do NOT use outside knowledge. "
               "If the answer is not explicitly in the context, say 'I don't know'. "
               "Always cite the source in your answer."
               "If answer not found, say 'I don't know'."),
    ("user", "Context:\n{context}\n\nQuestion:\n{query}")
])

# Evaluator Prompt
EVALUATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an evaluator that assesses answer quality based on context and query.\n\n"
        "EVALUATION CRITERIA:\n"
        "1. Faithfulness: Adherence to context without hallucinations.\n"
        "2. Relevance: How well the answer addresses the question.\n\n"
        "For each criterion, provide a score from 0 to 1 and a brief explanation for your rating.\n\n"
        "Return response strictly in JSON:\n"
        "{{\n"
        "  \"faithfulness_score\": float, \n"
        "  \"relevance_score\": float, \n"
        "  \"feedback\": str\n"
        "}}\n"
        "DO NOT include extra text."
    )),
    ("user", "QUERY: {query}\n\nCONTEXT: {context}\n\nANSWER: {answer}")
])

# Refiner Prompt
REFINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a query optimization expert. Rewrite the user query to improve retrieval quality. "
               "Make it specific, add technical keywords, but keep intent same. "
               "Return ONLY the improved query."),
    ("user", "Original Query: {query}\n\nFeedback from system: {feedback}")
])


PLANER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert routing assistant for an interactive knowledge base.
                Your job is to determine if the user's request requires fetching information from their uploaded documents (retrieve), or if it's a general greeting/casual query (direct_answer).
                You MUST respond ONLY with a JSON object matching this schema:
                {{
                "next_step": "retrieve" or "direct_answer",
                "reasoning": "A brief explanation of why this route was chosen"
                }}"""),
    ("user", "User Query: {latest_message}")  
])