# Insurance AI Agent — RAG-Based Claim Assistant

## What is this project?

Insurance AI Agent is a full-stack AI application that helps analyze insurance policy documents and answer claim-related questions.

Users upload an insurance PDF and ask questions in natural language. The system extracts the document text, retrieves the most relevant policy clauses using RAG, and sends the relevant context to `openai/gpt-oss-120b` through Groq to generate a structured answer.

## Flow

```text
Insurance PDF
     ↓
PDF Text Extraction (PyMuPDF)
     ↓
Text Cleaning & ₹ Currency Fix
     ↓
Sentence Transformer Embeddings
     ↓
FAISS Vector Search
     ↓
Relevant Policy Clauses
     ↓
Groq — openai/gpt-oss-120b
     ↓
Structured Claim Answer
     ↓
React Frontend
```

## Technologies Used

### Backend
- Python
- FastAPI
- PyMuPDF
- Sentence Transformers — `all-MiniLM-L6-v2`
- FAISS
- Groq API
- `openai/gpt-oss-120b`

### Frontend
- React
- TypeScript
- Vite
- Tailwind CSS
- ShadCN UI

## Project Structure

```text
Insurance-AI-Agent-RAG-Based/
│
├── Backend-ai/
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   └── .env                 # API configuration
│   │
│   ├── utils/
│   │   ├── embedding.py         # Text embeddings
│   │   ├── faiss_index.py       # FAISS vector index
│   │   ├── pdf_processing.py    # PDF processing
│   │   └── search.py             # Clause retrieval
│   │
│   ├── requirements.txt
│   └── venv/
│
├── Frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

## Model

**LLM:** `openai/gpt-oss-120b` via Groq

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`

