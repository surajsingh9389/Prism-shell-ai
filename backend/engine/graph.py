import asyncio
from langgraph.graph import StateGraph, START, END
from backend.core.state import AgentState
from backend.agents.researcher import (
    planner, retriever_node, generator, 
    critic, refiner, route_from_planner, should_continue
)

def compile_workflow():
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

    return builder.compile()

graph = compile_workflow()



# if __name__ == "__main__":
#     inputs = {
#         "query": "summarize my document",
#         "iteration": 0,
#         "max_iterations": 3,
#         "thoughts": []
#     }
#     asyncio.run(graph.ainvoke(inputs))