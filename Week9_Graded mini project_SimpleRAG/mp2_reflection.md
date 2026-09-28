# MP2 · Mini-RAG Reflection Document

## 1. System Overview & Architecture
The system uses a standard RAG pipeline built with Python, OpenAI APIs (`text-embedding-3-small` and `gpt-4o-mini`), and a local Qdrant vector storage container. 

1. **Ingest Phase**: Documents in `corpus/` are parsed and split into text chunks.
2. **Embedding & Storage**: Each chunk is embedded into a 1536-dimensional vector using `text-embedding-3-small` and upserted into Qdrant using Cosine similarity.
3. **Retrieval**: User queries are embedded and queried against Qdrant via `.query_points()` returning the top-$k$ ($k=3$) relevant chunks.
4. **Generation**: The retrieved context is formatted into a prompt bound to system rules that mandate using *only* provided excerpts and citing sources.

---

## 2. Chunking Strategy Trade-offs
* **Target Size**: ~500 characters with an 80-character overlap.
* **Heuristic Structure**: Paragraph breaks (`\n\n`) were preserved, and short lines without ending punctuation were extracted as section headings.
* **Trade-offs**: 
  * *Pros*: Small chunk sizes maintain high semantic density and fit comfortably within context limits while reducing noise.
  * *Cons*: Very small chunk sizes can split complex multi-sentence relationships across boundaries, requiring overlap to preserve continuity.

---

## 3. Retrieval Performance & Accuracy
* **Document Matching**: The system achieved high source-matching accuracy on both predefined and custom learner test questions.
* **Validation Fact Matching**: Simple exact string matching in validation scripts can undercount accuracy when an LLM paraphrases correct answers. Focusing `expected_facts` on targeted single-token keywords resolved verification discrepancy.

---

## 4. Latency Analysis
* Average query processing latency ranged between **800ms and 3000ms**.
* **Vector Search Latency**: Qdrant search took $<50\text{ms}$.
* **LLM Completion Latency**: OpenAI API generation constituted over 90% of end-to-end latency.

---

## 5. Potential Improvements
* **Hybrid Search**: Combine vector similarity (dense) with sparse keyword search (BM25) to improve retrieval on specific named entities or archaic terms.
* **Reranking**: Implement a re-ranker cross-encoder model to sort top-10 retrieved candidates down to the top-3 before prompting the LLM.