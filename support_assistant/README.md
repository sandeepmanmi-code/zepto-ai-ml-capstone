Zepto Support Assistant
Overview

This module implements a grounded Zepto policy support assistant using:

Sentence Transformers with all-MiniLM-L6-v2 for local embeddings

ChromaDB for vector storage and cosine-similarity retrieval

LangGraph for intent routing and orchestration

Pydantic for validated structured responses

FastAPI for the HTTP API

Docker for local containerization

The required graded path is completely offline with respect to LLM inference. MOCK_LLM defaults to 1, so no LLM provider, API key, signup, or external LLM network request is required.

Project Structure
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── chroma_db/
├── build_index.py
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md

Setup

From the support_assistant directory:

python -m venv .venv


Activate the environment.

Linux/macOS:

source .venv/bin/activate


Windows:

.venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

Build the Vector Index

Run:

python build_index.py


The script:

Reads all eight policy documents from docs/.

Uses one document as one chunk because each supplied policy document is short enough for this assignment.

Embeds every chunk with all-MiniLM-L6-v2.

Creates the persistent ChromaDB collection zepto_policies.

Stores document IDs and source metadata together with the embeddings.

The expected output is similar to:

Indexed 8 documents.
Collection: zepto_policies

Running the API

The default mode is the required mock mode:

uvicorn main:app --reload --port 7860


No MOCK_LLM variable is required because the application defaults to mock mode.

The API endpoint is:

POST /ask


with request body:

{
  "query": "What is the delivery fee for an order below INR 149?"
}

Example 1: Policy Question

Request:

curl -X POST "http://127.0.0.1:7860/ask" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What is the delivery fee for an order below INR 149?\"}"


Example response:

{
  "answer": "Based on the retrieved context: Delivery Policy: \"Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_04"
  ],
  "confidence": 1.0
}


The exact ordering of the lower-ranked retrieved documents can depend on the embedding/index implementation, but doc_01 is the relevant source for the delivery-fee question.

Example 2: General Question

Request:

curl -X POST "http://127.0.0.1:7860/ask" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What is the capital of France?\"}"


Response:

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}

Architecture

The complete RAG pipeline is:

8 Zepto Policy Documents
          |
          v
     Ingestion
   build_index.py
          |
          v
     Chunking
  one chunk/document
          |
          v
  Sentence Transformer
  all-MiniLM-L6-v2
          |
          v
      ChromaDB
  zepto_policies collection
          |
          |
       User Query
          |
          v
    classify_intent
       /       \
      /         \
policy_question general_question
     |                |
     v                v
retrieve_and_answer direct_answer
     |
     v
Top-3 cosine retrieval
     |
     v
Grounded answer
     |
     +--------+
              |
              v
     Pydantic validation
              |
              v
          FastAPI
          POST /ask

Ingestion

build_index.py loads all eight files from docs/. Each supplied document is treated as one chunk. The document ID, source filename, text, and embedding are stored in ChromaDB.

Embedding

The SentenceTransformer model all-MiniLM-L6-v2 generates local embeddings. No external embedding API is required.

Retrieval

The retrieve() function in main.py embeds the incoming query and calls the zepto_policies ChromaDB collection. It requests the three most similar chunks using cosine similarity.

Intent routing

The LangGraph graph first executes classify_intent.

In mock mode, it uses the required keyword heuristic. The keywords are:

delivery
return
refund
membership
tracking
cancel
gift card
support hours


A matching query becomes policy_question and goes to retrieve_and_answer.

All other queries become general_question and go to direct_answer.

The routing itself does not depend on the LLM toggle.

Generation

For policy questions, retrieve_and_answer retrieves the top three chunks. In mock mode, the answer is generated deterministically using:

Based on the retrieved context: <top retrieved chunk excerpt>


For general questions, direct_answer returns the deterministic response:

I can only answer questions about Zepto policies right now.

Structured output

The final response is validated with the Pydantic AskResponse model:

answer: string
sources: list[string]
confidence: float between 0 and 1


Policy responses contain the retrieved document IDs in sources. General responses have an empty sources list.

Mock mode sets confidence deterministically to 1.0.

Structured Prompt

The optional real-LLM path uses a role-context-task-format-length prompt skeleton.

The prompt explicitly contains:

Role: Zepto policy support assistant

Context: supplied Zepto policy documents

Task: answer the user's policy question

Format: structured answer, sources, confidence

Length: concise 1–4 sentence answer

Negative constraint: do not use information outside the supplied context

Few-shot example: a delivery-fee question and structured answer

The prompt is stored in main.py as PROMPT_TEMPLATE.

MOCK_LLM Behavior

MOCK_LLM is the single switch controlling generation.

Default: MOCK_LLM=1

The system:

makes no LLM API calls

uses keyword-based intent classification

performs real local embedding

performs real ChromaDB retrieval

generates deterministic answers

validates the final response through Pydantic

This is the required grading path.

Optional: MOCK_LLM=0

The intended architecture allows the generation portions to be connected to a real LLM provider.

The retrieval stage does not change. Embeddings and ChromaDB retrieval remain local.

The structured prompt is then supplied to the LLM for policy-answer generation or direct-answer generation.

A production implementation should validate the returned JSON and retry up to two additional times with a corrective instruction if the model output fails the required schema.

No API key is stored in this repository.

Docker

Build the container from /support_assistant:

docker build -t zepto-support-assistant .


Run it:

docker run --rm -p 7860:7860 zepto-support-assistant


The service is then available at:

http://127.0.0.1:7860


Test:

curl -X POST "http://127.0.0.1:7860/ask" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Can I cancel an order after it has been packed?\"}"


The Dockerfile runs build_index.py during image construction, so the ChromaDB collection is available when the FastAPI service starts.

Design Decisions
One chunk per document

The supplied policy documents are short and internally coherent. Splitting them further would add unnecessary complexity and could separate closely related policy statements. One chunk per document therefore provides simple, transparent retrieval while satisfying the assignment.

Local embeddings

all-MiniLM-L6-v2 was selected because it is lightweight, widely used for semantic similarity, and runs locally without an API key.

ChromaDB

ChromaDB provides persistent local vector storage and cosine-similarity retrieval, matching the requirements without introducing a hosted vector database.

LangGraph

LangGraph makes the intent-routing flow explicit. The graph contains the three required nodes:

classify_intent
retrieve_and_answer
direct_answer


The conditional edge after classify_intent selects the appropriate downstream node.

Mock generation

The mock mode is deliberately deterministic. This makes the graded baseline reproducible and ensures that network availability or an external LLM service cannot affect the result.

Structured response

Pydantic validates every API response. This prevents malformed responses from leaving the service even though the mock generation path itself is deterministic.