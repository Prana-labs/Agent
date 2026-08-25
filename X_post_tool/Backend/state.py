import uuid
from typing import Dict, Any, Union

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from rag import DocumentCollection
from agent import run_agent

sessions: Dict[str, Dict[str, Any]] = {}
checkpointer = InMemorySaver()


# =========================================================
# AGENT LANGGRAPH NODE
# =========================================================

def agent_node(
    state: MessagesState,
    config: RunnableConfig
):
    """
    Main LangGraph node executing the Agent:
    1. Senses query intent (no_rag, standard_rag, hybrid_rag).
    2. Runs balanced retrieval if RAG is required (single-PDF or multi-PDF).
    3. Generates conversational or structured learning output.
    """
    thread_id = config["configurable"]["thread_id"]

    if thread_id not in sessions:
        raise ValueError(
            "No PDF session found for this thread_id. Please upload a PDF first."
        )

    session = sessions[thread_id]
    pipeline: DocumentCollection = session["pipeline"]

    question = state["messages"][-1].content
    history = state["messages"]

    # Run the intelligent Agent controller
    answer = run_agent(
        question=question,
        history=history,
        doc_collection=pipeline
    )

    return {
        "messages": [AIMessage(content=answer)]
    }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

builder = StateGraph(MessagesState)

builder.add_node(
    "agent",
    agent_node
)

builder.add_edge(
    START,
    "agent"
)

# COMPILE GRAPH
graph = builder.compile(
    checkpointer=checkpointer
)


# =========================================================
# SESSION MANAGEMENT
# =========================================================

def create_session(
    pipeline: DocumentCollection,
    filename: str
) -> str:
    """
    Creates a new conversation session with a unique thread_id.
    """
    thread_id = str(uuid.uuid4())

    sessions[thread_id] = {
        "pipeline": pipeline,
        "filename": filename
    }

    return thread_id


def chat(
    thread_id: str,
    question: str
) -> str:
    """
    Sends a question to the LangGraph agent conversation.
    LangGraph maintains stateful conversation history for the thread_id.
    """
    if thread_id not in sessions:
        raise ValueError(
            "Invalid thread_id. Please upload a PDF first."
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=question
                )
            ]
        },
        config
    )

    answer = result["messages"][-1].content
    return answer


def get_history(thread_id: str):
    """
    Returns the conversation history for a thread.
    """
    if thread_id not in sessions:
        raise ValueError("Invalid thread_id.")

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = graph.get_state(config)
    return state.values.get("messages", [])


def delete_session(thread_id: str):
    """
    Removes the PDF session from memory.
    """
    if thread_id in sessions:
        del sessions[thread_id]