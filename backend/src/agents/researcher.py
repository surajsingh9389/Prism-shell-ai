import json
from typing import Literal, Optional

from src.core.state import AgentState
from src.core.prompts import GENERATOR_PROMPT, EVALUATOR_PROMPT, REFINER_PROMPT
from src.services.llm import LLMService
from src.engine.data_manager import vector_db
from langchain_core.runnables import RunnableConfig

llm_service = LLMService()

# --- NODES ---

async def planner(state: AgentState) -> AgentState:
    """Decides if the query needs document retrieval or a direct answer."""
    
    # Pull the newest incoming user query from the message history
    latest_message = state["messages"][-1].content if state.get("messages") else ""
    
    query = latest_message.lower()
    
    # Cache the latest query text into the state string for the retriever to use
    state["query"] = latest_message
    
    # Heuristics for 2026 RAG intent detection
    retrieval_keywords = [
        "document", "pdf", "file", "uploaded",
        "according to", "in the text", "based on"
    ]
    
    state["plan"] = "retrieve" if any(k in query for k in retrieval_keywords) else "direct_answer"
    
    state.setdefault("thoughts", []).append(f"Planner: Strategy set to {state['plan']}")
    return state


async def retriever_node(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    """Fetches and reranks documents from Qdrant."""
    query = state["query"]
    
    session_id = None
    if config and "configurable" in config:
        session_id = config["configurable"].get("session_id")
        
    if not session_id:
        raise ValueError("Security Boundary Error: Missing 'session_id' inside graph context configuration.")
    
    processed_docs = await vector_db.search_docs(query, session_id=session_id, top_k=3)
    
    return {"retrieved_docs": processed_docs}


async def generator(state: AgentState) -> AgentState:
    """Generates an answer based on the planner's decision and retrieved context."""
    query = state["query"]
    
    chat_history = state.get("messages", [])[:-1] # History excluding the very last query
    
    if state["plan"] == "direct_answer":
        system_msg = "Answer the question directly based on your general knowledge and chat history."
        user_msg = f"Conversation History:\n{chat_history}\n\nQuestion: {query}"
        answer = await llm_service.generate_text(system_msg, user_msg)
    else:
        docs = state.get("retrieved_docs", [])
        if not docs:
            state["current_answer"] = "I'm sorry, I couldn't find any relevant information in the documents."
            return state

        # Format context for the prompt
        context_str = "\n\n".join([f"{d['content']} (Source: {d['source']})" for d in docs])
        
        # Using the LangChain Prompt Templates from core/prompts.py
        prompt_value = GENERATOR_PROMPT.invoke({"context": context_str, 
        "query": f"Chat History for Context:\n{chat_history}\n\nCurrent Query: {query}"})
        
        # Extract system and human messages from the template
        system_msg = prompt_value.messages[0].content
        user_msg = prompt_value.messages[1].content
        
        answer = await llm_service.generate_text(system_msg, user_msg)

    state["current_answer"] = answer
    state["thoughts"].append("Generator: Answer synthesized.")
    return {"current_answer": answer, "messages": [("assistant", answer)]}


async def critic(state: AgentState) -> AgentState:
    """Evaluates the answer using a high-parameter model for strict verification."""
    state["iteration"] += 1
    
    docs = state.get("retrieved_docs", [])
    context = "\n\n".join([f"{d['content']}" for d in docs])
    
    prompt_value = EVALUATOR_PROMPT.invoke({
        "query": state["query"],
        "context": context,
        "answer": state["current_answer"]
    })
    
    # Extract system and human messages from the template
    system_msg = prompt_value.messages[0].content
    user_msg = prompt_value.messages[1].content

    # Call the structured evaluation service (Llama-3.3-70B)
    raw_eval = await llm_service.evaluate_structured(
        system_msg, 
        user_msg
    )
    
    eval_data = json.loads(raw_eval)
    
    state["faithfulness_score"] = eval_data.get("faithfulness_score", 0.0)
    state["relevance_score"] = eval_data.get("relevance_score", 0.0)
    state["feedback"] = eval_data.get("feedback", "")

    # Logic-based failure assignment
    if state["faithfulness_score"] < 0.7:
        state["failure_type"] = "hallucination"
    elif state["relevance_score"] < 0.7:
        state["failure_type"] = "low_relevance"
    else:
        state["failure_type"] = "good"
        
    state["thoughts"].append(f"Critic: Score {state['faithfulness_score']} | Status: {state['failure_type']}")
    return state


async def refiner(state: AgentState) -> AgentState:
    """Optimizes the query if retrieval relevance was low."""
    prompt_value = REFINER_PROMPT.invoke({
        "query": state["query"],
        "feedback": state["feedback"]
    })
    
    # Extract system and human messages from the template
    system_msg = prompt_value.messages[0].content
    user_msg = prompt_value.messages[1].content

    improved_query = await llm_service.generate_text(
        system_msg, 
        user_msg
    )

    state["refined_query"] = improved_query
    state["query"] = improved_query  # Injecting refined query for the next loop
    state["thoughts"].append(f"Refiner: New search query - {improved_query}")
    
    return state


# --- ROUTING LOGIC ---

def route_from_planner(state: AgentState) -> Literal["generator", "retriever"]:
    if state["plan"] == "direct_answer":
        return "generator"
    return "retriever"


def should_continue(state: AgentState) -> Literal["end", "refiner", "retriever"]:
    if state["iteration"] >= state["max_iterations"] or state["failure_type"] == "good":
        return "end"
    
    if state["failure_type"] == "low_relevance":
        return "refiner"
    
    # If hallucination, retry retrieval directly
    return "retriever"