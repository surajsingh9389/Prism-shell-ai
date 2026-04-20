from typing import List, Literal, TypedDict

# Document structure to hold retrieved information along with its source and scores for retrieval and reranking.
class Document(TypedDict):
    content: str
    source: str
    retrieval_score: float
    rerank_score: float

# State Schema for the agent, capturing all relevant information across the different stages of the process.
class AgentState(TypedDict):
    # --- Input ---
    query: str
    iteration: int
    max_iterations: int

    # --- Retrieval ---
    retrieved_docs: List[Document]

    # --- Generation ---
    current_answer: str

    # --- Evaluation ---
    faithfulness_score: float
    relevance_score: float
    feedback: str
    failure_type: Literal[
        "hallucination",
        "low_relevance",
        "incomplete",
        "good"
    ]

    # --- Observability ---
    thoughts: List[str]
    