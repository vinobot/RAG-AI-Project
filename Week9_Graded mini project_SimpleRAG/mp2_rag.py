"""MP2 · Mini-RAG — Complete Implementation
===========================================

Pipeline:
    corpus/*.txt  →  chunks  →  embeddings  →  Qdrant
                                                  ↓
                              question  →  retrieve  →  answer + citations
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from dotenv import load_dotenv

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR        = Path(__file__).parent / "corpus"
DATA_DIR          = Path(__file__).parent / "data"
COLLECTION_NAME   = "mp2_sherlock"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
CHAT_MODEL        = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP     = 80    # characters

load_dotenv()
openai = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
print("Environment details are loaded...!")

qdrant = QdrantClient(
    url=os.environ.get("QDRANT_URL", "http://localhost:6333")
)


# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory."""
    docs = []
    for file_path in sorted(corpus_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Extract first non-empty line as title; fallback to formatted filename
        title = lines[0] if lines else file_path.stem.replace("_", " ").title()
        
        docs.append({
            "source": file_path.name,
            "title": title,
            "text": text
        })
    print("Corpus got loaded successfully...!!")   
    return docs


# ─── Step 2: Chunk each document ────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into paragraph-aware chunks with section header detection."""
    chunks = []
    raw_paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]
    
    current_section = "General"
    current_chunk = ""
    
    for para in raw_paragraphs:
        # Heuristic section header detection: short line (<60 chars), no terminal punctuation
        if len(para) < 60 and not para.endswith((".", "?", "!", '"', "'")):
            current_section = para
            continue

        # Check if adding paragraph exceeds max chunk size
        if len(current_chunk) + len(para) > TARGET_CHUNK_SIZE and current_chunk:
            chunks.append({
                "source": doc["source"],
                "title": doc["title"],
                "section": current_section,
                "text": current_chunk.strip()
            })
            # Carry over overlapping text from end of current chunk
            overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) >= CHUNK_OVERLAP else current_chunk
            current_chunk = overlap_text + "\n\n" + para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    # Flush remaining text buffer
    if current_chunk.strip():
        chunks.append({
            "source": doc["source"],
            "title": doc["title"],
            "section": current_section,
            "text": current_chunk.strip()
        })
    
    return chunks
print("Document got chunked successfully...!!")   


# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model."""
    if not texts:
        return []
        
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    embeddings = [data.embedding for data in response.data]
    return embeddings



# ─── Step 4: Set up the Qdrant collection ───────────────────────────────

def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection."""
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
        
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        )
    )
    print("Collection got created successfully in qdrant ...!!")


# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]], batch_size: int = 50) -> None:
    """Embed every chunk and upsert into Qdrant in batches."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        
        embeddings = embed_texts(texts)
        
        points = []
        for chunk, embedding in zip(batch, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=chunk
                )
            )
            
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
    print("Embedded chunks got ingested into qdrant..!!!")    


# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query."""
    query_vector = embed_texts([query])[0]
    
    # Updated for newer qdrant-client versions
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True
    )
    
    retrieved = []
    for i, hit in enumerate(response.points, start=1):
        chunk = dict(hit.payload) if hit.payload else {}
        chunk["score"] = hit.score
        chunk["id"] = hit.id
        retrieved.append(chunk)
        
        # Print complete dictionary structure with all fields
        # print(f"\n--- Chunk {i} ---")
        # print(json.dumps(chunk, indent=2))

    return retrieved


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result."""
    start_time = time.time()
    
    # 1. Retrieve relevant chunks
    chunks = retrieve(question, k=k)
    
    # 2. Format context for prompt
    context_blocks = []
    citations = []
    
    for chunk in chunks:
        context_blocks.append(
            f"[Source: {chunk['title']} — {chunk['section']}]\n{chunk['text']}"
        )
        citations.append({
            "source": chunk["source"],
            "title": chunk["title"],
            "section": chunk["section"],
            "score": chunk.get("score", 0.0)
        })
        
    context_str = "\n\n---\n\n".join(context_blocks)
    
    # 3. Call LLM
    user_message = f"Context:\n{context_str}\n\nQuestion: {question}"
    
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1
    )
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "citations": citations,
        "latency_ms": elapsed_ms
    }


# ─── Validation harness (provided — do not modify) ──────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()