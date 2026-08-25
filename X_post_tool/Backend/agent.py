import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langsmith import traceable

from LLMs import model
from schema import LearningResponse
from rag import DocumentCollection


# =========================================================
# QUERY CLASSIFICATION ROUTER SCHEMA
# =========================================================

class QueryRoute(BaseModel):
    """
    Intelligent query classification:
    - 'no_rag': Greetings, casual chit-chat, pleasantries, assistant meta-questions.
    - 'standard_rag': General conceptual / high-level semantic queries on documents.
    - 'hybrid_rag': Technical queries, keyword/acronym lookups, multi-document comparisons ('topic of each', 'compare all').
    """
    route: Literal["no_rag", "standard_rag", "hybrid_rag"] = Field(
        ...,
        description=(
            "Use 'no_rag' for greetings ('hi', 'hey', 'how are you doing'), pleasantries, thanks, or casual conversation. "
            "Use 'standard_rag' for high-level semantic conceptual questions about the document(s). "
            "Use 'hybrid_rag' for technical questions, specific terms, acronyms, numbers, or multi-document comparison questions (e.g. 'topic of each', 'summarize each')."
        )
    )
    reasoning: str = Field(
        ...,
        description="Brief 1-sentence reasoning for the chosen route."
    )


# =========================================================
# FAST HEURISTIC GREETING CHECK (Low Latency)
# =========================================================

GREETING_PATTERNS = [
    r"^(hi|hello|hey|heyy|greetings|hola|howdy)(\s+.*)?$",
    r"^how\s+(are\s+you|you\s+doing|is\s+it\s+going|are\s+things)(\s+.*)?$",
    r"^(good\s+(morning|afternoon|evening|day))(\s+.*)?$",
    r"^(who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+your\s+name)(\s+.*)?$",
    r"^(thank\s+you|thanks|thx|bye|goodbye)(\s+.*)?$",
    r"^(help|what\s+should\s+i\s+ask)(\s+.*)?$"
]

def is_obvious_greeting(text: str) -> bool:
    cleaned = text.strip().lower()
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, cleaned):
            return True
    return False


# =========================================================
# AGENT ROUTER LOGIC
# =========================================================

@traceable(name="agent_classify_query")
def classify_query(question: str, history: Optional[List[BaseMessage]] = None) -> QueryRoute:
    """
    Classifies the user query into no_rag, standard_rag, or hybrid_rag.
    """
    # 1. Fast path for obvious greetings / chit-chat
    if is_obvious_greeting(question):
        return QueryRoute(
            route="no_rag",
            reasoning="Fast-path detected conversational greeting or pleasantry."
        )

    # 2. Multi-document / comparison heuristic keywords -> hybrid_rag
    lower_q = question.lower()
    if any(keyword in lower_q for keyword in ["each", "all", "compare", "difference", "versus", "vs", "list of", "overview of all", "every"]):
        return QueryRoute(
            route="hybrid_rag",
            reasoning="Query involves cross-document comparison or per-document enumeration."
        )

    # 3. LLM-based intelligent classification
    try:
        router_llm = model.with_structured_output(QueryRoute)
        system_prompt = SystemMessage(
            content="""You are a query routing agent for a document RAG system.
Classify the user query into:
- 'no_rag': Greetings, pleasantries, conversational chit-chat, or general questions not referencing documents.
- 'standard_rag': High-level semantic or conceptual questions about the document(s).
- 'hybrid_rag': Specific technical questions, keywords, acronyms, data points, or multi-document comparison queries.
"""
        )
        messages = [system_prompt, HumanMessage(content=question)]
        route_decision = router_llm.invoke(messages)
        if isinstance(route_decision, QueryRoute):
            return route_decision
    except Exception as e:
        print(f"Router LLM fallback to hybrid_rag due to: {e}")

    return QueryRoute(
        route="hybrid_rag",
        reasoning="Defaulted to hybrid RAG for robust coverage."
    )


# =========================================================
# AGENT EXECUTION NODES
# =========================================================

@traceable(name="execute_no_rag")
def execute_no_rag(
    question: str,
    history: List[BaseMessage],
    doc_names: List[str]
) -> str:
    """
    Handles conversational interactions naturally without querying vector stores or forcing schema.
    """
    doc_count = len(doc_names)
    doc_context_hint = ""
    if doc_count == 1:
        doc_context_hint = f"The user has 1 active document loaded: '{doc_names[0]}'."
    elif doc_count > 1:
        doc_context_hint = f"The user has {doc_count} active documents loaded: {', '.join(doc_names)}."

    system_prompt = SystemMessage(
        content=f"""You are a helpful, polite, and friendly PDF Research Assistant.
{doc_context_hint}

Respond to the user's conversational message naturally, warmly, and concisely.
If they greeted you, greet them back and let them know you are ready to answer any questions about their loaded document(s).
Do NOT output rigid document bullet points or hallucinate technical papers unless the user asked a specific document question.
"""
    )

    messages = [system_prompt] + history
    response = model.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


@traceable(name="execute_rag")
def execute_rag(
    question: str,
    history: List[BaseMessage],
    doc_collection: DocumentCollection,
    mode: str = "hybrid"
) -> str:
    """
    Executes Standard or Hybrid RAG with balanced per-document retrieval and structured learning points.
    """
    num_docs = doc_collection.document_count
    top_k_per_doc = 3 if num_docs > 1 else 6

    context = doc_collection.get_formatted_context(
        question=question,
        mode=mode,
        top_k_per_doc=top_k_per_doc
    )

    doc_list_str = ", ".join(doc_collection.doc_names) if doc_collection.doc_names else "Uploaded Document"

    system_prompt = SystemMessage(
        content=f"""You are an expert PDF learning assistant analyzing the following loaded document(s): {doc_list_str}.

Your task is to answer the user's question using the provided document context and previous conversation.

CRITICAL INSTRUCTIONS FOR MULTI-DOCUMENT ACCURACY & GROUNDEDNESS:
1. The DOCUMENT CONTEXT contains excerpts tagged with `[Document: <filename> | Page: <page_num>]`.
2. If the user asks about "each", "all", or "compare", create a distinct, numbered learning point for EACH document present in the context.
3. Explicitly cite the document name for each point.
4. Do NOT confuse or merge findings from different documents.
5. If a document's details are not in the context, explicitly state that context was insufficient for that document rather than guessing.

Always structure your response using the required LearningResponse schema:
- `initial_heading`: Engaging, clear topic title.
- `overview`: Concise 1-2 sentence introduction.
- `learning_points`: Clear, numbered learning points with subheadings and detailed bullet points.
- `key_takeaways`: Core summary takeaways.

DOCUMENT CONTEXT:
{context}
"""
    )

    messages = [system_prompt] + history

    try:
        structured_llm = model.with_structured_output(LearningResponse)
        structured_result = structured_llm.invoke(messages)

        if isinstance(structured_result, LearningResponse):
            return structured_result.to_markdown()
        elif isinstance(structured_result, dict):
            return LearningResponse(**structured_result).to_markdown()
        else:
            return str(structured_result)
    except Exception as e:
        print(f"Structured output error: {e}, falling back to standard LLM invoke")
        raw_response = model.invoke(messages)
        return raw_response.content if hasattr(raw_response, "content") else str(raw_response)


# =========================================================
# MAIN AGENT CONTROLLER
# =========================================================

@traceable(name="run_agent")
def run_agent(
    question: str,
    history: List[BaseMessage],
    doc_collection: DocumentCollection
) -> str:
    """
    Agent main entrypoint:
    1. Sense and classify user query (no_rag / standard_rag / hybrid_rag).
    2. Dispatch to the appropriate execution node.
    3. Return clean, formatted response.
    """
    decision = classify_query(question, history)
    route = decision.route

    if route == "no_rag":
        return execute_no_rag(
            question=question,
            history=history,
            doc_names=doc_collection.doc_names
        )
    elif route == "standard_rag":
        return execute_rag(
            question=question,
            history=history,
            doc_collection=doc_collection,
            mode="standard"
        )
    else:  # hybrid_rag
        return execute_rag(
            question=question,
            history=history,
            doc_collection=doc_collection,
            mode="hybrid"
        )
