# CampusAI-RAG-Assistant
An AI-powered college assistant built using React, TypeScript, FastAPI, and Retrieval-Augmented Generation (RAG). Uses FAISS vector search, BM25 hybrid retrieval, Sentence Transformers, and Groq LLM to answer college-related, personal profile, and general knowledge questions. Supports PDF uploads and document-based question answering.
# CampusAI – RAG College Assistant

CampusAI is an AI-powered college assistant that helps students get answers to college-related questions, understand academic information, and access general knowledge through a conversational interface.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from college documents and provide context-aware responses using a Large Language Model (LLM).

## Features

- **AI-Powered Chat:** Ask questions through a conversational interface.
- **Document-Based Question Answering:** Retrieve relevant information from college PDF documents.
- **Hybrid Search:** Combines FAISS vector similarity search with BM25 keyword search.
- **Profile-Based Responses:** Answers questions using configured profile information.
- **PDF Upload:** Upload additional PDF documents and index their contents.
- **Source Information:** Displays document sources used to generate answers.
- **Fresh Chat Experience:** Start a new conversation without restoring previous chat messages.
- **Modern User Interface:** Built with React, TypeScript, and CSS.

## Technology Stack

### Frontend
- React
- TypeScript
- Vite
- CSS

### Backend
- Python
- FastAPI
- Uvicorn

### AI & RAG
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- BM25 (`rank-bm25`)
- Groq API
- PyPDF

## Project Architecture

```text
CampusAI-RAG-Assistant/
│
├── backend/
│   ├── main.py
│   ├── rag_engine.py
│   └── profile.json
│
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── App.css
│       └── index.css
│
├── documents/
│   ├── College_Regulations_and_Student_Data.pdf
│   └── uploads/
│
├── faiss_index.index
├── .env
└── README.md
```

## How It Works

1. College PDF documents are loaded and their text is extracted.
2. Extracted content is divided into smaller chunks.
3. Sentence Transformers converts text chunks into vector embeddings.
4. FAISS performs semantic similarity search.
5. BM25 performs keyword-based retrieval.
6. Results from both retrieval methods are combined to identify relevant context.
7. The retrieved context is passed to the Groq LLM.
8. The generated answer is returned to the frontend.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Madhukunchala99/CampusAI-RAG-Assistant.git
cd CampusAI-RAG-Assistant
```

### 2. Backend Setup

Create and activate a virtual environment:

**Windows:**

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install fastapi uvicorn python-multipart pypdf sentence-transformers faiss-cpu rank-bm25 numpy python-dotenv groq
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your Groq API key.

**Important:** Never upload your `.env` file or expose your API key on GitHub.

### 4. Run the Backend

From the project root:

```bash
python -m uvicorn backend.main:app --reload
```

Backend API:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Frontend Setup

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the frontend at:

```text
http://localhost:5173
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check whether the API is running |
| GET | `/health` | Check backend health |
| POST | `/chat` | Ask questions to CampusAI |
| POST | `/upload-pdf` | Upload and index PDF documents |
| GET | `/documents` | View indexed documents |

## Future Improvements

- Conversation history management
- Image-based question answering
- Improved retrieval and reranking
- User authentication and role-based access
- Support for additional document formats
- More advanced evaluation of RAG response quality

## Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Large Language Model integration
- Vector databases and semantic search
- Hybrid information retrieval
- REST API development
- Full-stack application development
- Document processing and indexing

## Author

**Harshith Katta**

GitHub: [Madhukunchala99](https://github.com/Madhukunchala99)

---

*CampusAI – Making college information easier to access through AI.*
