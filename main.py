import asyncio
from typing import List, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from hybrid_search_with_reranking import get_hybrid_reranked_docs
from build_prompt import build_prompt
from generator import generateAnswer

# Document structure to hold retrieved information along with its source and scores for retrieval and reranking.
class RetrievedDoc(TypedDict):
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
    retrieved_docs: List[RetrievedDoc]

    # --- Generation ---
    current_answer: str

    # # --- Evaluation ---
    # faithfulness_score: float
    # relevance_score: float
    # feedback: str
    # failure_type: Literal[
    #     "hallucination",
    #     "low_relevance",
    #     "incomplete",
    #     "good"
    # ]

    # # --- Observability ---
    # thoughts: List[str]
    

async def retriever_node(state: AgentState):
    query = state["query"]
    # We AWAIT the result here
    processed_docs = await get_hybrid_reranked_docs(query)
    state["iteration"] += 1
    
    return {"retrieved_docs": processed_docs}

async def generator(state: AgentState) -> AgentState:
    # This function would typically use the retrieved documents to generate an answer to the query.
    query = state["query"]
    docs = state["retrieved_docs"]
    context = "\n\n".join([f"{doc['content']} (source: {doc['source']})" for doc in docs])
    prompt = build_prompt(query, context)
    answer = await generateAnswer(prompt)
    state['current_answer'] = answer
    return state

async def critic(state: AgentState) -> AgentState:
    # Placeholder for evaluation logic
    # This function would typically evaluate the generated answer for faithfulness and relevance.
    # For demonstration, we will just return the state with some dummy evaluation scores and feedback.
    state['faithfulness_score'] = 0.8
    state['relevance_score'] = 0.75
    state['feedback'] = "The answer is mostly faithful but could be more relevant to the query."
    state['failure_type'] = "low_relevance"
    return state


agent_builder = StateGraph(AgentState)

agent_builder.add_node("retriever", retriever_node)
agent_builder.add_node("generator", generator)
# agent_builder.add_node("critic", critic)

agent_builder.add_edge(START, "retriever")
agent_builder.add_edge("retriever", "generator")
# agent_builder.add_edge("generator", "critic")

graph = agent_builder.compile()

async def run_agent():
    # Use AINVOKE instead of INVOKE
    state = await graph.ainvoke({
        "query": "What is FAISS?", 
        "iteration": 0, 
        "max_iterations": 3
    })
    print(state)

# 3. Run the async loop
if __name__ == "__main__":
    asyncio.run(run_agent())