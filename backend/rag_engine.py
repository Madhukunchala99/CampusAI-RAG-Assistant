
import os
import re
import gc
import json
from pathlib import Path

# ============================================================
# MEMORY OPTIMIZATION SETTINGS
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import faiss
import torch

from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = ROOT_DIR / "documents" / "College_Regulations_and_Student_Data.pdf"
UPLOAD_DIR = ROOT_DIR / "documents" / "uploads"

INDEX_PATH = ROOT_DIR / "faiss_index.index"
RECORDS_PATH = ROOT_DIR / "chunk_records.json"

PROFILE_PATH = ROOT_DIR / "backend" / "profile.json"
ENV_PATH = ROOT_DIR / ".env"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ENV_PATH)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found!")

client = Groq(api_key=api_key)

GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_SEMANTIC_SCORE = 0.35
BATCH_SIZE = 8

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

model = None
index = None
bm25 = None

chunk_records = []
profile_data = {}


# ============================================================
# STEP 2: TEXT PROCESSING
# ============================================================

def normalize_text(text):
    return " ".join(text.lower().split())


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# ============================================================
# STEP 3: LOAD STUDENT PROFILE
# ============================================================

def load_profile():
    global profile_data

    if PROFILE_PATH.exists():
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as file:
                profile_data = json.load(file)

        except (OSError, json.JSONDecodeError) as error:
            print("Profile loading error:", error)
            profile_data = {}


# ============================================================
# STEP 4: LAZY LOAD EMBEDDING MODEL
# ============================================================

def get_model():
    global model

    if model is None:
        print("Loading embedding model...")

        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

        model.max_seq_length = 256

        print("Embedding model loaded successfully!")

    return model


# ============================================================
# STEP 5: PDF EXTRACTION AND CHUNKING
# ============================================================

def create_chunks(pdf_path, source_filename):

    records = []

    reader = PdfReader(str(pdf_path))

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text() or ""
        page_text = " ".join(page_text.split())

        start = 0

        while start < len(page_text):

            end = start + CHUNK_SIZE

            chunk = page_text[start:end].strip()

            if chunk:
                records.append({
                    "text": chunk,
                    "source": source_filename,
                    "page": page_number
                })

            if end >= len(page_text):
                break

            start = end - CHUNK_OVERLAP

    return records


def get_all_records():

    records = []

    if PDF_PATH.exists():
        records.extend(
            create_chunks(PDF_PATH, PDF_PATH.name)
        )

    for pdf_path in sorted(UPLOAD_DIR.glob("*.pdf")):

        records.extend(
            create_chunks(pdf_path, pdf_path.name)
        )

    if not records:
        raise ValueError("No readable PDF documents found!")

    for chunk_id, record in enumerate(records):
        record["chunk_id"] = chunk_id

    return records


# ============================================================
# STEP 6: BUILD BM25 INDEX
# ============================================================

def build_bm25():
    global bm25

    tokenized_chunks = [
        tokenize(record["text"])
        for record in chunk_records
    ]

    bm25 = BM25Okapi(tokenized_chunks)


# ============================================================
# STEP 7: SAVE INDEX METADATA
# ============================================================

def save_records():

    with open(RECORDS_PATH, "w", encoding="utf-8") as file:

        json.dump(
            chunk_records,
            file,
            ensure_ascii=False
        )


# ============================================================
# STEP 8: LOAD SAVED INDEX
# ============================================================

def load_saved_index():

    global index, chunk_records

    if not INDEX_PATH.exists() or not RECORDS_PATH.exists():
        return False

    try:

        with open(RECORDS_PATH, "r", encoding="utf-8") as file:
            records = json.load(file)

        loaded_index = faiss.read_index(
            str(INDEX_PATH)
        )

        if not records:
            return False

        if loaded_index.ntotal != len(records):
            return False

        chunk_records = records
        index = loaded_index

        build_bm25()

        print("Existing FAISS index loaded.")
        print("Total chunks:", len(chunk_records))

        return True

    except Exception as error:

        print("Saved index loading failed:", error)

        index = None
        bm25 = None
        chunk_records = []

        return False


# ============================================================
# STEP 9: REBUILD INDEXES IN SMALL BATCHES
# ============================================================

def rebuild_indexes():

    global index, chunk_records

    print("Reading PDF documents...")

    records = get_all_records()

    embedding_model = get_model()

    new_index = None

    normalized_chunks = [
        normalize_text(record["text"])
        for record in records
    ]

    print("Generating embeddings in batches...")

    for start in range(0, len(normalized_chunks), BATCH_SIZE):

        batch = normalized_chunks[
            start:start + BATCH_SIZE
        ]

        embeddings = embedding_model.encode(
            batch,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = np.ascontiguousarray(
            embeddings,
            dtype="float32"
        )

        if new_index is None:

            new_index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )

        new_index.add(embeddings)

        del embeddings

    if new_index is None:
        raise ValueError("No embeddings generated.")

    chunk_records = records
    index = new_index

    build_bm25()

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    save_records()

    gc.collect()

    print("FAISS and BM25 indexes built successfully.")
    print("Total indexed chunks:", len(chunk_records))


# ============================================================
# STEP 10: INITIALIZE INDEXES
# ============================================================

def initialize_indexes():

    if index is not None and bm25 is not None:
        return

    if not load_saved_index():
        rebuild_indexes()


# ============================================================
# STEP 11: ADD UPLOADED PDF
# ============================================================

def add_uploaded_pdf(file_path):

    file_path = Path(file_path)

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    reader = PdfReader(str(file_path))

    if len(reader.pages) == 0:
        raise ValueError("Uploaded PDF is empty.")

    rebuild_indexes()

    return {
        "message": "PDF uploaded and indexes rebuilt successfully.",
        "filename": file_path.name,
        "total_chunks": len(chunk_records)
    }


# ============================================================
# STEP 12: LIST INDEXED DOCUMENTS
# ============================================================

def list_indexed_documents():

    # Ensure indexes and records are loaded first
    initialize_indexes()

    documents = {}

    for record in chunk_records:

        filename = record.get("source", "Unknown")
        page = record.get("page", 0)

        if filename not in documents:

            documents[filename] = {
                "filename": filename,
                "chunks": 0,
                "pages": set()
            }

        documents[filename]["chunks"] += 1

        documents[filename]["pages"].add(page)

    return [
        {
            "filename": filename,
            "chunks": details["chunks"],
            "indexed_pages": len(details["pages"])
        }
        for filename, details in documents.items()
    ]


# ============================================================
# STEP 13: QUESTION CLASSIFICATION
# ============================================================

def classify_question(question):

    prompt = (
        "Classify this question. Return only CODE or GENERAL. "
        "CODE means programming, debugging, or implementation. "
        "GENERAL means everything else.\n"
        f"Question: {question}"
    )

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=5
        )

        result = response.choices[0].message.content or ""

        return (
            "CODE"
            if "CODE" in result.upper()
            else "GENERAL"
        )

    except Exception as error:

        print("Classification error:", error)

        return "GENERAL"


def is_code_question(question):

    patterns = [
        r"\b(code|coding|programming|python|java|javascript)\b",
        r"\b(function|class|algorithm|debug|exception|syntax)\b",
        r"\b(api|fastapi|react|sql|html|css|docker)\b",
        r"\b(implement|write a script|write code|fix this code)\b"
    ]

    return any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in patterns
    )


def clean_code_response(response):

    if not response:
        return response

    match = re.search(
        r"```(?:[\w+-]+)?\s*([\s\S]*?)```",
        response
    )

    if match:
        return match.group(1).strip()

    return response.strip()


# ============================================================
# STEP 14: HYBRID RETRIEVAL - FAISS + BM25
# ============================================================

def retrieve_context(question):

    initialize_indexes()

    normalized_question = normalize_text(question)

    question_embedding = get_model().encode(
        [normalized_question],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.ascontiguousarray(
        question_embedding,
        dtype="float32"
    )

    k = min(
        TOP_K * 3,
        len(chunk_records)
    )

    semantic_scores, semantic_ids = index.search(
        question_embedding,
        k
    )

    semantic_score_map = {}

    for score, chunk_id in zip(
        semantic_scores[0],
        semantic_ids[0]
    ):

        if chunk_id >= 0:
            semantic_score_map[int(chunk_id)] = float(score)

    bm25_scores = bm25.get_scores(
        tokenize(question)
    )

    bm25_ids = np.argsort(bm25_scores)[::-1][:k]

    combined_scores = {}

    # Semantic ranking
    for rank, chunk_id in enumerate(semantic_ids[0]):

        if chunk_id >= 0:

            combined_scores[int(chunk_id)] = (
                combined_scores.get(int(chunk_id), 0)
                + 1 / (60 + rank + 1)
            )

    # Keyword ranking
    for rank, chunk_id in enumerate(bm25_ids):

        combined_scores[int(chunk_id)] = (
            combined_scores.get(int(chunk_id), 0)
            + 1 / (60 + rank + 1)
        )

    ranked_ids = sorted(
        combined_scores,
        key=combined_scores.get,
        reverse=True
    )

    retrieved = []

    for chunk_id in ranked_ids:

        semantic_score = semantic_score_map.get(
            chunk_id,
            0.0
        )

        if (
            semantic_score >= MIN_SEMANTIC_SCORE
            or not retrieved
        ):

            record = chunk_records[chunk_id]

            retrieved.append({
                **record,
                "semantic_score": semantic_score
            })

        if len(retrieved) >= TOP_K:
            break

    return retrieved


# ============================================================
# STEP 15: STUDENT PROFILE CONTEXT
# ============================================================

def get_profile_context(question):

    if not profile_data:
        return ""

    question_lower = question.lower()

    matches = []

    for key, value in profile_data.items():

        name = ""

        if isinstance(value, dict):
            name = str(
                value.get("name", "")
            ).lower()

        if (
            key.lower() in question_lower
            or (name and name in question_lower)
        ):

            matches.append(
                json.dumps(
                    value,
                    ensure_ascii=False
                )
            )

    return "\n".join(matches)


# ============================================================
# STEP 16: GENERATE ANSWER USING GROQ
# ============================================================

def ask_question(question):

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    question_type = (
        "CODE"
        if is_code_question(question)
        else classify_question(question)
    )

    retrieved_chunks = retrieve_context(question)

    context = "\n\n".join(
        (
            f"Source: {item['source']} | Page: {item['page']}\n"
            f"{item['text']}"
        )
        for item in retrieved_chunks
    )

    profile_context = get_profile_context(question)

    if profile_context:

        context += (
            "\n\nRelevant student profile information:\n"
            + profile_context
        )

    prompt = f"""
You are CampusAI, an intelligent College Assistant.

You have access to:
1. Retrieved college document context.
2. General knowledge as an AI language model.
3. Relevant student profile information, when available.

RULE 1:
For college-specific questions about regulations, attendance,
student records, examinations, and college policies, use the
provided document context as the source of official information.

RULE 2:
If college-specific information is missing from the retrieved context,
clearly say that it is not available in the uploaded documents.
You may then provide a clearly labelled general explanation.

RULE 3:
For general educational and technical questions, use your general
knowledge and explain clearly.

RULE 4:
Never present general knowledge as an official college rule.

RULE 5:
Use student profile information only when relevant to the question.
Do not invent personal details or make insulting claims about people.

RULE 6:
For programming questions, provide accurate, beginner-friendly
answers and complete code when requested.

Retrieved Document and Profile Context:
{context if context else "No relevant document context found."}

User Question:
{question}

Answer:
"""

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1500
        )

        answer = (
            response.choices[0].message.content or ""
        )

        if question_type == "CODE":
            answer = clean_code_response(answer)

        return {
            "answer": answer,
            "sources": [
                {
                    "filename": item["source"],
                    "page": item["page"],
                    "score": round(
                        item["semantic_score"],
                        4
                    )
                }
                for item in retrieved_chunks
            ]
        }

    except Exception as error:

        print("Groq API request failed:", error)

        raise


# ============================================================
# STEP 17: INITIALIZE PROFILE
# ============================================================

load_profile()

print("CampusAI RAG engine initialized successfully.")
print("Embedding model and indexes will load when first needed.")
