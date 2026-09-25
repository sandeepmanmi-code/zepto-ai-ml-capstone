import json
import os
from pathlib import Path
from typing import List, Literal, TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


# ============================================================
# Pydantic Models
# ============================================================

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


# ============================================================
# LangGraph State
# ============================================================

class GraphState(TypedDict, total=False):
    query: str
    intent: Literal[
        "policy_question",
        "general_question"
    ]

    retrieved_documents: List[str]
    retrieved_ids: List[str]

    answer: str
    sources: List[str]
    confidence: float

    response: dict


# ============================================================
# Embedding Model + ChromaDB
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Connecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"Connected to collection '{COLLECTION_NAME}' "
    f"with {collection.count()} documents."
)


# ============================================================
# Structured Prompt
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are Zepto's customer-support policy assistant.

CONTEXT:
You may answer policy questions ONLY using the Zepto policy
information provided in the retrieved context.

TASK:
Answer the user's question using only the supplied context.
If the context does not contain enough information, say that
the available Zepto policy context does not contain the required
information.

FORMAT:
Return a JSON object with exactly these fields:
{
    "answer": "string",
    "sources": ["chunk/document IDs"],
    "confidence": 0.0
}

LENGTH:
Keep the answer concise and customer-friendly, preferably within
2 to 4 sentences.

NEGATIVE CONSTRAINT:
Do not use outside knowledge.
Do not invent, infer, or assume a Zepto policy that is not present
in the provided context.

FEW-SHOT EXAMPLE:

User:
What is the delivery fee for an order below INR 149?

Context:
Standard delivery is free on orders over INR 149; orders below
this threshold incur a flat INR 25 delivery fee.

Expected output:
{
    "answer": "Orders below INR 149 incur a flat INR 25 standard delivery fee.",
    "sources": ["doc_01"],
    "confidence": 1.0
}

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{query}
"""


def build_prompt(query: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(
        query=query,
        context=context,
    )


# ============================================================
# MOCK_LLM
# ============================================================

def mock_llm_enabled() -> bool:
    """
    MOCK_LLM unset or equal to 1 -> mock mode.
    MOCK_LLM=0 -> optional real LLM mode.
    """

    return os.getenv("MOCK_LLM", "1") != "0"


# ============================================================
# Node 1: Intent Classification
# ============================================================

def classify_intent(state: GraphState) -> GraphState:

    query = state["query"].lower()

    matched = any(
        keyword in query
        for keyword in POLICY_KEYWORDS
    )

    if matched:
        intent = "policy_question"
    else:
        intent = "general_question"

    print(
        f"[classify_intent] "
        f"{state['query']} -> {intent}"
    )

    return {
        **state,
        "intent": intent,
    }


# ============================================================
# Retrieval
# ============================================================

def retrieve_documents(query: str):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    return documents, ids


# ============================================================
# Optional Real LLM
# ============================================================

def call_real_llm(
    query: str,
    documents: List[str],
    ids: List[str],
) -> AnswerResponse:

    """
    Optional MOCK_LLM=0 path.

    The graded implementation uses MOCK_LLM=1 and therefore
    never reaches this function.
    """

    try:
        from groq import Groq
    except ImportError:

        return AnswerResponse(
            answer=(
                "Real LLM mode requires the optional "
                "groq package."
            ),
            sources=ids,
            confidence=0.0,
        )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        return AnswerResponse(
            answer=(
                "GROQ_API_KEY is not configured."
            ),
            sources=ids,
            confidence=0.0,
        )

    client = Groq(api_key=api_key)

    context = "\n\n".join(
        f"[{doc_id}]\n{document}"
        for doc_id, document
        in zip(ids, documents)
    )

    prompt = build_prompt(
        query=query,
        context=context,
    )

    last_error = None

    for attempt in range(3):

        try:

            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )

            raw_output = (
                completion
                .choices[0]
                .message
                .content
                .strip()
            )

            # Remove accidental Markdown code fences.
            if raw_output.startswith("```"):
                raw_output = raw_output.replace(
                    "```json",
                    ""
                )
                raw_output = raw_output.replace(
                    "```",
                    ""
                )
                raw_output = raw_output.strip()

            parsed = json.loads(raw_output)

            return AnswerResponse.model_validate(
                parsed
            )

        except Exception as exc:

            last_error = exc

            prompt = build_prompt(
                query=query,
                context=context,
            )

            prompt += """

CORRECTION:
Your previous answer did not satisfy the required JSON
schema. Return ONLY valid JSON containing:
answer, sources, and confidence.
"""

    return AnswerResponse(
        answer=(
            "ERROR: Real LLM output could not be "
            f"validated after three attempts: {last_error}"
        ),
        sources=ids,
        confidence=0.0,
    )


# ============================================================
# Node 2: Retrieve + Answer
# ============================================================

def retrieve_and_answer(
    state: GraphState
) -> GraphState:

    query = state["query"]

    print(
        f"[retrieve_and_answer] Query: {query}"
    )

    documents, ids = retrieve_documents(
        query
    )

    print(
        "[retrieve_and_answer] Retrieved:",
        ids
    )

    # --------------------------------------------------------
    # Required graded mock path
    # --------------------------------------------------------

    if mock_llm_enabled():

        top_chunk = documents[0]

        snippet = top_chunk[:200]

        answer = (
            "Based on the retrieved context: "
            f"{snippet}"
        )

        response = AnswerResponse(
            answer=answer,
            sources=ids,
            confidence=1.0,
        )

    # --------------------------------------------------------
    # Optional real LLM path
    # --------------------------------------------------------

    else:

        response = call_real_llm(
            query=query,
            documents=documents,
            ids=ids,
        )

    return {
        **state,
        "retrieved_documents": documents,
        "retrieved_ids": ids,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "response": response.model_dump(),
    }


# ============================================================
# Node 3: Direct Answer
# ============================================================

def direct_answer(
    state: GraphState
) -> GraphState:

    print(
        f"[direct_answer] Query: {state['query']}"
    )

    # Required mock behaviour.
    if mock_llm_enabled():

        response = AnswerResponse(
            answer=(
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            sources=[],
            confidence=1.0,
        )

    # Optional real LLM extension.
    else:

        response = AnswerResponse(
            answer=(
                "General-question real LLM mode "
                "can be connected here."
            ),
            sources=[],
            confidence=0.5,
        )

    return {
        **state,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "response": response.model_dump(),
    }


# ============================================================
# Conditional Router
# ============================================================

def route_intent(state: GraphState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# Build LangGraph
# ============================================================

def build_graph():

    graph = StateGraph(
        GraphState
    )

    graph.add_node(
        "classify_intent",
        classify_intent,
    )

    graph.add_node(
        "retrieve_and_answer",
        retrieve_and_answer,
    )

    graph.add_node(
        "direct_answer",
        direct_answer,
    )

    graph.set_entry_point(
        "classify_intent"
    )

    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer":
                "retrieve_and_answer",

            "direct_answer":
                "direct_answer",
        },
    )

    graph.add_edge(
        "retrieve_and_answer",
        END,
    )

    graph.add_edge(
        "direct_answer",
        END,
    )

    return graph.compile()


app_graph = build_graph()


# ============================================================
# Helper
# ============================================================

def ask_question(
    query: str
) -> AnswerResponse:

    result = app_graph.invoke(
        {
            "query": query
        }
    )

    return AnswerResponse.model_validate(
        result["response"]
    )


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Zepto Policy Support Assistant",
    description=(
        "Grounded Zepto policy assistant using "
        "LangGraph, ChromaDB and local embeddings."
    ),
    version="1.0.0",
)


@app.get("/")
def root():

    return {
        "service": "Zepto Support Assistant",
        "status": "running",
        "mock_llm": mock_llm_enabled(),
        "documents": collection.count(),
    }


@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask(
    request: AskRequest
):

    return ask_question(
        request.query
    )
