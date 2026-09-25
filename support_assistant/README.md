# Zepto Support Assistant

A local retrieval-augmented support assistant for answering questions about Zepto policies. It uses sentence-transformer embeddings, ChromaDB for vector retrieval, LangGraph for the conversation flow, and FastAPI to expose the `/ask` endpoint.

## Architecture

1. Policy documents are stored as text files in `docs/`.
2. Documents are split into chunks and embedded using `all-MiniLM-L6-v2`.
3. ChromaDB stores the embeddings and retrieves relevant chunks for a user query.
4. LangGraph routes the query through exactly three nodes:
   - `classify_intent`
   - `retrieve_and_answer`
   - `direct_answer`
5. The API returns a structured response containing `answer`, `sources`, and `confidence`.

## Requirements

- Python 3.11 recommended
- Internet access on first run to download the embedding model
- Dependencies listed in `requirements.txt`

## Run locally

From this directory:

```bash
pip install -r requirements.txt
uvicorn main:app --reload