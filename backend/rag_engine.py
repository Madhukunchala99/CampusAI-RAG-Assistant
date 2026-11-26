
import os
import re
import json
from pathlib import Path

import numpy as np
import faiss

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
ENV_PATH = ROOT_DIR / ".env"
PROFILE_PATH = ROOT_DIR / "backend" / "profile.json"

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
STRONG_SEMANTIC_SCORE = 0.50


# ============================================================
# STEP 2: LOAD PROFILE
# ============================================================

with open(PROFILE_PATH, "r", encoding="utf-8") as file:
    PROFILE_DATA = json.load(file)

print("Personal profile loaded successfully!")


# ============================================================
# STEP 3: TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    return " ".join(text.lower().split())


def tokenize(text):
    return re.findall(r"\b[\w.%-]+\b", normalize_text(text))


# ============================================================
# STEP 4: PDF CHUNKING
# ============================================================

def create_chunks(pdf_path: Path, source_filename: str):

    pdf = PdfReader(str(pdf_path))

    chunk_records = []

    for page_number, page in enumerate(pdf.pages, start=1):

        page_text = page.extract_text()

        if not page_text:
            continue

        page_text = page_text.strip()

        start = 0

        while start < len(page_text):

            end = start + CHUNK_SIZE

            chunk_text = page_text[start:end].strip()

            if chunk_text:

                chunk_records.append({
                    "chunk_id": len(chunk_records),
                    "filename": source_filename,
                    "page": page_number,
                    "text": chunk_text
                })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunk_records


# ============================================================
# STEP 5: LOAD MODEL AND BUILD INDEXES
# ============================================================

print("Loading RAG resources...")

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_records = []
bm25 = None
index = None


def rebuild_indexes():

    global chunk_records
    global bm25
    global index

    print("Rebuilding document indexes...")

    all_records = []

    if PDF_PATH.exists():

        records = create_chunks(
            PDF_PATH,
            PDF_PATH.name
        )

        all_records.extend(records)

    for pdf_path in sorted(UPLOAD_DIR.glob("*.pdf")):

        records = create_chunks(
            pdf_path,
            pdf_path.name
        )

        all_records.extend(records)

    if not all_records:
        raise ValueError("No readable PDF documents found!")

    for chunk_id, record in enumerate(all_records):
        record["chunk_id"] = chunk_id

    chunk_records = all_records

    normalized_chunks = [
        normalize_text(record["text"])
        for record in chunk_records
    ]

    # BM25 INDEX

    tokenized_chunks = [
        tokenize(chunk)
        for chunk in normalized_chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    print("BM25 index created successfully!")

    # FAISS INDEX

    embeddings = model.encode(
        normalized_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    embeddings = np.ascontiguousarray(embeddings)

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    print("FAISS index created successfully!")
    print("Total indexed chunks:", len(chunk_records))

    indexed_files = {
        record["filename"]
        for record in chunk_records
    }

    print("Total indexed PDFs:", len(indexed_files))


# ============================================================
# STEP 6: ADD UPLOADED PDF
# ============================================================

def add_uploaded_pdf(pdf_path: Path):

    if not pdf_path.exists():
        raise FileNotFoundError("Uploaded PDF does not exist.")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF documents are supported.")

    test_pdf = PdfReader(str(pdf_path))

    if len(test_pdf.pages) == 0:
        raise ValueError("The uploaded PDF is empty.")

    if not any(page.extract_text() for page in test_pdf.pages):
        raise ValueError(
            "No selectable text found in this PDF. "
            "Scanned PDFs require OCR."
        )

    rebuild_indexes()

    return {
        "filename": pdf_path.name,
        "total_chunks": len(chunk_records),
        "status": "indexed"
    }


# ============================================================
# STEP 7: LIST INDEXED DOCUMENTS
# ============================================================

def list_indexed_documents():

    documents = {}

    for record in chunk_records:

        filename = record["filename"]

        if filename not in documents:

            documents[filename] = {
                "filename": filename,
                "pages": set()
            }

        documents[filename]["pages"].add(
            record["page"]
        )

    return [
        {
            "filename": item["filename"],
            "indexed_pages": len(item["pages"])
        }
        for item in documents.values()
    ]


# BUILD INDEXES WHEN BACKEND STARTS

rebuild_indexes()


# ============================================================
# STEP 8: QUESTION CLASSIFICATION
# ============================================================

def classify_question(question: str):

    prompt = f"""
Classify the question into exactly one category.

COLLEGE:
College regulations, attendance, examinations,
student records, fees and college documents.

PROFILE:
Questions about Madhu, Sneha, their personal details,
education, careers, personality or relationship.

GENERAL:
General knowledge, AI, programming, science,
technology and educational questions.

MIXED:
Questions requiring information from multiple sources.

Return only one word:
COLLEGE, PROFILE, GENERAL, or MIXED.

Question: {question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    category = (
        response.choices[0].message.content or ""
    ).strip().upper()

    allowed = [
        "COLLEGE",
        "PROFILE",
        "GENERAL",
        "MIXED"
    ]

    if category not in allowed:
        return "GENERAL"

    return category


# ============================================================
# STEP 9: CODE QUESTION DETECTION
# ============================================================

def is_code_question(question: str):

    q = question.lower().strip()

    code_patterns = [

        r"\bwrite\s+(me\s+)?(some\s+)?code\b",
        r"\bgive\s+(me\s+)?(the\s+)?code\b",
        r"\bgenerate\s+(the\s+)?code\b",
        r"\bprovide\s+(the\s+)?code\b",
        r"\bshow\s+(me\s+)?(the\s+)?code\b",

        r"\b(write|create|generate|give|provide)\s+(me\s+)?(a\s+)?(python|java|javascript|typescript|sql|html|css|c\+\+|c#)\s+code\b",

        r"\b(write|create|generate)\s+(a|an)\s+(python|java|javascript|typescript)\s+(program|script|function)\b",

        r"\bcode\s+to\s+\w+",
        r"\bprogram\s+to\s+\w+",
        r"\bfunction\s+to\s+\w+",
        r"\bscript\s+to\s+\w+",

        r"\busing\s+(python|java|javascript|typescript|sql)\b",

        r"\bcreate\s+(a|an)\s+(function|class|program|script|component|api)\b",

        r"\bimplement\b",
        r"\bcode\s+only\b",
        r"\bonly\s+code\b",
        r"\bfull\s+code\b",
        r"\bcomplete\s+code\b",

        r"\bfix\s+(this\s+)?code\b",
        r"\bdebug\s+(this\s+)?code\b",
        r"\bmodify\s+(this\s+)?code\b",
        r"\bupdate\s+(this\s+)?code\b",

        r"\bwrite\s+(a\s+)?sql\s+query\b",
        r"\bgive\s+(me\s+)?(a\s+)?sql\s+query\b",

        r"\b(build|create|write)\s+(a|an)\s+(website|application|api|component)\b"
    ]

    return any(
        re.search(pattern, q)
        for pattern in code_patterns
    )


# ============================================================
# STEP 10: CLEAN CODE RESPONSE
# ============================================================

def clean_code_response(answer: str):

    answer = answer.strip()

    if not answer:
        return answer

    # Extract the first Markdown code block if present.

    code_block = re.search(
        r"```(?:python|py|java|javascript|js|typescript|ts|sql|html|css|c\+\+|c#)?\s*\n(.*?)```",
        answer,
        flags=re.IGNORECASE | re.DOTALL
    )

    if code_block:
        return code_block.group(1).strip()

    # Remove conversational introductions when no
    # Markdown code block is present.

    lines = answer.splitlines()

    code_starters = (
        "import ",
        "from ",
        "def ",
        "class ",
        "if ",
        "for ",
        "while ",
        "try:",
        "with ",
        "const ",
        "let ",
        "var ",
        "function ",
        "public ",
        "private ",
        "SELECT ",
        "#include",
        "print(",
        "return ",
        "async ",
        "export "
    )

    start_index = None

    for i, line in enumerate(lines):

        stripped = line.strip()

        if stripped.startswith(code_starters):
            start_index = i
            break

    if start_index is not None:
        answer = "\n".join(lines[start_index:]).strip()

    return answer


# ============================================================
# STEP 11: HYBRID RETRIEVAL
# ============================================================

def retrieve_context(question: str):

    query = normalize_text(question)

    # SEMANTIC SEARCH

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    query_embedding = np.ascontiguousarray(
        query_embedding
    )

    semantic_k = min(5, len(chunk_records))

    semantic_scores, semantic_indices = index.search(
        query_embedding,
        semantic_k
    )

    semantic_score_map = {}

    for idx, score in zip(
        semantic_indices[0],
        semantic_scores[0]
    ):

        if idx != -1:
            semantic_score_map[int(idx)] = float(score)

    # KEYWORD SEARCH

    query_tokens = tokenize(query)

    keyword_scores = bm25.get_scores(query_tokens)

    keyword_results = []

    for idx, score in enumerate(keyword_scores):

        if score > 0:
            keyword_results.append((idx, float(score)))

    keyword_results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    keyword_results = keyword_results[:5]

    keyword_score_map = {
        idx: score
        for idx, score in keyword_results
    }

    # RECIPROCAL RANK FUSION

    fused_scores = {}

    semantic_results = sorted(
        semantic_score_map.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for rank, (chunk_id, score) in enumerate(
        semantic_results,
        start=1
    ):

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0)
            + 1 / (60 + rank)
        )

    for rank, (chunk_id, score) in enumerate(
        keyword_results,
        start=1
    ):

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0)
            + 1 / (60 + rank)
        )

    # RANK RESULTS

    ranked_chunks = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # FILTER IRRELEVANT RESULTS

    relevant_chunks = []

    for chunk_id, rrf_score in ranked_chunks:

        semantic_score = semantic_score_map.get(
            chunk_id, 0.0
        )

        keyword_score = keyword_score_map.get(
            chunk_id, 0.0
        )

        has_keyword_match = keyword_score > 0

        is_relevant = (
            semantic_score >= MIN_SEMANTIC_SCORE
            and (
                has_keyword_match
                or semantic_score >= STRONG_SEMANTIC_SCORE
            )
        )

        print(
            f"Chunk {chunk_id}: "
            f"Semantic={semantic_score:.4f}, "
            f"BM25={keyword_score:.4f}, "
            f"RRF={rrf_score:.6f}, "
            f"Relevant={is_relevant}"
        )

        if is_relevant:

            relevant_chunks.append({
                "chunk_id": chunk_id,
                "rrf_score": rrf_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score
            })

    # SELECT TOP RELEVANT CHUNKS

    selected_chunks = relevant_chunks[:TOP_K]

    retrieved_chunks = []
    source_details = []

    for result in selected_chunks:

        chunk_id = result["chunk_id"]

        record = chunk_records[chunk_id]

        retrieved_chunks.append(record["text"])

        source_details.append({
            "chunk_id": record["chunk_id"],
            "filename": record["filename"],
            "page": record["page"],
            "retrieval_score": round(
                float(result["rrf_score"]), 6
            ),
            "semantic_score": round(
                float(result["semantic_score"]), 4
            ),
            "keyword_score": round(
                float(result["keyword_score"]), 4
            ),
            "text": record["text"]
        })

    # RETRIEVAL STATUS

    if source_details:
        retrieval_status = "matched"
    else:
        retrieval_status = "no_relevant_context"

    print("Semantic results:", list(semantic_score_map.keys()))

    print("Keyword results:", [
        idx for idx, score in keyword_results
    ])

    print("Selected chunks:", [
        item["chunk_id"] for item in source_details
    ])

    print("Source files:", [
        item["filename"] for item in source_details
    ])

    print("Source pages:", [
        item["page"] for item in source_details
    ])

    print("Retrieval status:", retrieval_status)

    return (
        retrieved_chunks,
        source_details,
        retrieval_status
    )


# ============================================================
# STEP 12: GENERATE ANSWER
# ============================================================

def ask_question(question: str):

    # DETECT CODE MODE

    code_mode = is_code_question(question)

    print("\n========== NEW QUESTION ==========")
    print("Question:", question)
    print("Code-only mode:", code_mode)

    # CLASSIFY QUESTION

    category = classify_question(question)

    print("Question category:", category)

    retrieved_chunks = []
    source_details = []
    retrieval_status = "not_applicable"

    # RETRIEVE COLLEGE INFORMATION

    if category in ["COLLEGE", "MIXED"]:

        (
            retrieved_chunks,
            source_details,
            retrieval_status
        ) = retrieve_context(question)

    # PREPARE COLLEGE CONTEXT

    context_parts = []

    for source in source_details:

        context_parts.append(
            f"""
[Source File: {source["filename"]}]
[PDF Page: {source["page"]}]
[Chunk ID: {source["chunk_id"]}]

{source["text"]}
"""
        )

    context = "\n\n".join(context_parts)

    # PREPARE PROFILE CONTEXT

    if category in ["PROFILE", "MIXED"]:

        profile_context = json.dumps(
            PROFILE_DATA,
            indent=2,
            ensure_ascii=False
        )

    else:
        profile_context = ""

    # RESPONSE INSTRUCTIONS

    if code_mode:

        system_prompt = """
You are CampusAI, a programming assistant.

STRICT CODE-ONLY MODE.

Return exactly one solution.

Rules:

1. Return source code only.
2. Generate simple beginner-friendly code.
3. Do not write an introduction.
4. Do not explain the code.
5. Do not add headings.
6. Do not use Markdown fences.
7. Do not add conclusions.
8. Do not generate alternative solutions.
9. Do not add unnecessary imports.
10. Do not add command-line argument handling unless requested.
11. Do not add unnecessary error handling.
12. Do not add unnecessary classes.
13. Do not over-engineer simple tasks.
14. Include a simple function call or example when useful.
15. Follow the exact programming language requested.

For a simple programming question, prefer the shortest
clear and correct implementation.

Your entire response must be source code only.
"""

        response_instructions = """
CODE-ONLY MODE:

Return only one simple, correct code solution.

No introductions.
No explanations.
No headings.
No Markdown fences.
No alternatives.
No conclusions.

Keep the implementation beginner-friendly.
"""

        temperature = 0.1

    else:

        system_prompt = """
You are CampusAI, a friendly, intelligent,
natural and conversational AI assistant.

Answer questions clearly and accurately.

Follow the source restrictions and response instructions
provided in the user's message.
"""

        response_instructions = """
NORMAL RESPONSE MODE:

Answer naturally and conversationally.

Use clear explanations and examples when useful.

Answer directly and keep simple answers short.

Use more detail when requested.
"""

        temperature = 0.4

    # BUILD PROMPT

    prompt = f"""
QUESTION CATEGORY:
{category}

COLLEGE DOCUMENT CONTEXT:
{context}

PERSONAL PROFILE:
{profile_context}

USER QUESTION:
{question}

IMPORTANT INSTRUCTIONS:

1. COLLEGE QUESTIONS:

Use college document context for official college facts.

Never invent rules or percentages.

If no relevant college document context is provided,
do not guess or use general knowledge for official
college rules.

Clearly say that the information could not be found
in the uploaded document.

2. PERSONAL QUESTIONS:

Use the personal profile as the source of truth.

Understand different wording and indirect references.

Combine information when appropriate.

3. FUNNY QUESTIONS:

Use the funny_responses and humor_guidelines
sections in the personal profile.

Understand the user's intention even when the exact
question is not listed in the triggers.

Use natural, playful, affectionate humor.

You can generate new jokes inspired by the profile.

Do not simply repeat the same predefined response.

4. RELATIONSHIP QUESTIONS:

Madhu and Sneha are romantic partners who hope
to get married in the future.

Answer warmly and naturally.

Never claim they are already married or engaged.

5. GENERAL QUESTIONS:

Use your general knowledge.

Explain concepts clearly and use examples when useful.

6. MIXED QUESTIONS:

Combine relevant sources without confusing
official facts, personal facts and general knowledge.

If college context is missing, do not invent
college-related information.

7. SOURCE ACCURACY:

Use the supplied PDF context for college facts.

Use the exact source filename and page number
when identifying document sources.

Never invent PDF page numbers or source filenames.

If the context does not support the answer,
say the information is not available in the document.

8. RESPONSE FORMAT:

{response_instructions}

Now answer the user's question.
"""

    # CALL GROQ

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature
    )

    answer = response.choices[0].message.content or ""

    # CLEAN CODE RESPONSE

    if code_mode:
        answer = clean_code_response(answer)

    print("Final code mode:", code_mode)

    # RETURN RESPONSE

    return {
        "answer": answer.strip(),
        "sources": retrieved_chunks,
        "source_details": source_details,
        "retrieval_status": retrieval_status
    }
