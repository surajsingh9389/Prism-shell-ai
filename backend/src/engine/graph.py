from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from src.core.state import AgentState
from src.core.database import db_pool
from src.agents.researcher import (
    planner, retriever_node, generator, 
    critic, refiner, route_from_planner, should_continue
)


def create_graph_builder() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner)
    builder.add_node("retriever", retriever_node)
    builder.add_node("generator", generator)
    builder.add_node("critic", critic)
    builder.add_node("refiner", refiner)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_from_planner,
        {"generator": "generator", "retriever": "retriever"}
    )
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", "critic")
    builder.add_conditional_edges(
        "critic", 
        should_continue, 
        {"end": END, "retriever": "retriever", "refiner": "refiner"}
    )
    builder.add_edge("refiner", "retriever")
    
    return builder

def get_runtime_graph_with_pool(pool: AsyncConnectionPool):
    """
    Binds the checkpointer smoothly to the pooled connection.
    No database setup runs here, keeping database tasks clear.
    """
    checkpointer = AsyncPostgresSaver(pool)
    builder = create_graph_builder()
    return builder.compile(checkpointer=checkpointer)

