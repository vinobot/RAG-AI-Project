# Mini-RAG: Sherlock Holmes Question Answering System

This project is an end-to-end Retrieval-Augmented Generation (RAG) system built in Python. It retrieves relevant chunks from Sherlock Holmes short stories stored in a Qdrant vector database and uses OpenAI's models to answer questions accurately with citations.

---

## Project Structure

* `mp2_rag.py`: Main executable containing loading, chunking, embedding, vector storage, retrieval, generation, and validation logic.
* `corpus/`: Text files containing Sherlock Holmes stories.
* `data/predefined_questions.jsonl`: Validation set of predefined test questions.
* `data/learner_questions.jsonl`: Custom validation questions created for testing.
* `.env`: Environment configuration file storing API keys and endpoint URLs.
* `mp2_validation.txt`: Output log generated from running the automated validation harness.

---

## Prerequisites & Installation

1. **Clone the repository** and navigate to the project root.
2. **Set up a Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
3. Install dependencies
4. Config environment variables
5. Ensure qdrant is running locally via docker

## How to run
1. Ingest corpus into qdrant(Chunks documents, generates vector embeddings, and stores them in Qdrant)
   python mp2_rag.py ingest
2. Interactive Q&A(Ask custom queries in real time)
   python mp2_rag.py ask
3. Run validation suite(Test pipeline accuracy and output validation log)
   python mp2_rag.py validate | Out-File -FilePath mp2_validation.txt -Encoding utf-8      

