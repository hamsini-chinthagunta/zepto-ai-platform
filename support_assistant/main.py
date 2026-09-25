
import os
from pathlib import Path
from typing import TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "zepto_policy_docs"

app = FastAPI(title="Zepto Support Assistant")

# Structured prompt template: role, context, task, format, length.
# The mock mode below does not call an external LLM.
PROMPT_TEMPLATE = """
ROLE:
You are a helpful Zepto customer-support assistant.

CONTEXT:
Use only the policy excerpts retrieved for this question.
Do not answer using information that is not present in the provided context.

TASK:
Answer the customer's question using the retrieved policy context.
If the context does not contain the answer, say that you do not have
enough policy information to answer.

FORMAT:
Return a JSON object with answer, sources, and confidence.

LENGTH:
Keep the answer concise, ideally 1-3 sentences.

FEW-SHOT EXAMPLE:
Context: "Orders below INR 149 have a delivery fee of INR 30."
Question: "What is the delivery fee for orders below INR 149?"
Answer: "Orders below INR 149 have a delivery fee of INR 30."
"""


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


# Load the local embedding model and persistent ChromaDB collection.
embedding_model = SentenceTransformer(MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def split_into_chunks(text: str, max_chars: int = 700) -> list[str]:
    """Split each policy document into small, readable text chunks."""
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start:start + max_chars])
            continue

        if current and len(current) + len(paragraph) + 1 > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}".strip()

    if current:
        chunks.append(current)

    return chunks


def ingest_documents() -> None:
    """Read the 8 policy files, embed chunks, and store them in ChromaDB."""
    if collection.count() > 0:
        return

    documents = []
    ids = []
    metadatas = []

    for file_path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        for index, chunk in enumerate(split_into_chunks(text)):
            chunk_id = f"{file_path.stem}_chunk_{index:02d}"
            documents.append(chunk)
            ids.append(chunk_id)
            metadatas.append(
                {
                    "document_id": file_path.stem,
                    "chunk_id": chunk_id,
                    "filename": file_path.name,
                }
            )

    if not documents:
        raise RuntimeError(
            f"No policy documents found. Add the 8 text files to: {DOCS_DIR}"
        )

    embeddings = embedding_model.encode(
        documents,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )


ingest_documents()


def classify_intent(state: AssistantState) -> AssistantState:
    """Classify policy questions with a deterministic keyword heuristic."""
    query = state["query"].lower()
    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours",
    ]

    if any(keyword in query for keyword in policy_keywords):
        return {"intent": "policy_question"}

    return {"intent": "general_question"}


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    """Retrieve the top 3 matching chunks and produce the mock answer."""
    query_embedding = embedding_model.encode(
        [state["query"]],
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_meta = results.get("metadatas", [[]])[0]

    source_ids = [
        metadata["chunk_id"]
        for metadata in retrieved_meta
        if metadata and "chunk_id" in metadata
    ]

    if not retrieved_docs:
        return {
            "answer": "I could not find a matching policy excerpt.",
            "sources": [],
            "confidence": 0.0,
        }

    top_chunk_snippet = retrieved_docs[0][:200].strip()

    if MOCK_LLM:
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
    else:
        answer = (
            "Real LLM mode is not configured. Set MOCK_LLM=1 to use "
            "the deterministic offline response."
        )

    return {
        "answer": answer,
        "sources": source_ids,
        "confidence": 0.82,
    }


def direct_answer(state: AssistantState) -> AssistantState:
    """Respond to general questions without retrieving policy documents."""
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
    else:
        answer = (
            "Real LLM mode is not configured. Set MOCK_LLM=1 to use "
            "the deterministic offline response."
        )

    return {
        "answer": answer,
        "sources": [],
        "confidence": 0.95,
    }


def route_by_intent(state: AssistantState) -> str:
    return state["intent"]


# Exactly three LangGraph nodes:
# classify_intent -> retrieve_and_answer OR direct_answer -> END
workflow = StateGraph(AssistantState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer",
    },
)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

assistant_graph = workflow.compile()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = assistant_graph.invoke({"query": request.query.strip()})

    return AskResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
    )


@app.get("/")
def root():
    return {"message": "Zepto Support Assistant is running. POST questions to /ask."}