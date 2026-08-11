
import uuid

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from rag import retrieve
from LLMs import model

sessions = {}
checkpointer = InMemorySaver()

def rag_node(
    state: MessagesState,
    config: RunnableConfig
):
    """
    Main LangGraph node.

    1. Gets the current question
    2. Retrieves relevant PDF chunks
    3. Reranks them
    4. Combines PDF context + conversation history
    5. Sends everything to the LLM
    6. Returns the AI response
    """
    thread_id = config["configurable"]["thread_id"]

    if thread_id not in sessions:
        raise ValueError(
            "No PDF session found for this thread_id"
        )

    session = sessions[thread_id]

    vectorstore = session["vectorstore"]
    bm25 = session["bm25"]

    question = state["messages"][-1].content

    docs = retrieve(
        vectorstore,
        bm25,
        question
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    
    system_message = SystemMessage(
        content=f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using the provided
document context and the previous conversation.

Rules:

1. Use the document context as the primary source.
2. Use conversation history to understand follow-up
   questions and references.
3. Do not invent information that is not supported
   by the document.
4. If the answer cannot be found in the document,
   clearly say that you could not find it.

DOCUMENT CONTEXT:

{context}
"""
    )

    # -----------------------------------------------------
    # Conversation history
    # -----------------------------------------------------

    messages = [
        system_message
    ] + state["messages"]

    # -----------------------------------------------------
    # Call LLM
    # -----------------------------------------------------

    response = model.invoke(messages)

    # -----------------------------------------------------
    # Return new AI message
    # -----------------------------------------------------

    return {
        "messages": [response]
    }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

builder = StateGraph(MessagesState)

builder.add_node(
    "rag",
    rag_node
)

builder.add_edge(
    START,
    "rag"
)


# =========================================================
# COMPILE GRAPH
# =========================================================

graph = builder.compile(
    checkpointer=checkpointer
)


# =========================================================
# CREATE SESSION
# =========================================================

def create_session(
    vectorstore,
    bm25,
    filename
):
    """
    Creates a new conversation session.

    Returns:
        thread_id
    """

    thread_id = str(uuid.uuid4())

    sessions[thread_id] = {
        "vectorstore": vectorstore,
        "bm25": bm25,
        "filename": filename
    }

    return thread_id


# =========================================================
# CHAT
# =========================================================

def chat(
    thread_id: str,
    question: str
):
    """
    Sends a question to the LangGraph conversation.

    LangGraph automatically maintains the conversation
    history for this thread_id.
    """

    # -----------------------------------------------------
    # Validate session
    # -----------------------------------------------------

    if thread_id not in sessions:
        raise ValueError(
            "Invalid thread_id. Upload a PDF first."
        )

    # -----------------------------------------------------
    # LangGraph configuration
    # -----------------------------------------------------

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # -----------------------------------------------------
    # Invoke graph
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Get latest AI response
    # -----------------------------------------------------

    answer = result["messages"][-1].content

    return answer


# =========================================================
# GET CONVERSATION HISTORY
# =========================================================

def get_history(thread_id: str):
    """
    Returns the conversation history for a thread.
    """

    if thread_id not in sessions:
        raise ValueError(
            "Invalid thread_id."
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = graph.get_state(config)

    return state.values.get(
        "messages",
        []
    )


# =========================================================
# DELETE SESSION
# =========================================================

def delete_session(thread_id: str):
    """
    Removes the PDF session from memory.
    """

    if thread_id in sessions:
        del sessions[thread_id]