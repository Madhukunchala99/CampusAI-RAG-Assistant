
import re
import base64
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag_engine import (
    ask_question,
    add_uploaded_pdf,
    list_indexed_documents,
    UPLOAD_DIR,
    rebuild_indexes,
    client
)


# APP CONFIGURATION

app = FastAPI(
    title="CampusAI - Multimodal RAG Assistant API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# CONFIGURATION

MAX_PDF_SIZE = 20 * 1024 * 1024
MAX_IMAGE_SIZE = 10 * 1024 * 1024

VISION_MODEL = "qwen/qwen3.8-27b"

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


# REQUEST MODEL

class ChatRequest(BaseModel):
    question: str


# HOME

@app.get("/")
def home():
    return {
        "message": "CampusAI API is running!"
    }


# HEALTH CHECK

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# NORMAL RAG CHAT

@app.post("/chat")
def chat(request: ChatRequest):

    try:

        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        result = ask_question(question)

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "source_details": result.get("source_details", []),
            "retrieval_status": result.get(
                "retrieval_status",
                "not_applicable"
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        print("CHAT ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating the answer."
        )


# PDF UPLOAD

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    saved_path = None

    try:

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Please select a PDF file."
            )

        original_name = Path(file.filename).name

        if not original_name.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )

        contents = await file.read(MAX_PDF_SIZE + 1)

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        if len(contents) > MAX_PDF_SIZE:
            raise HTTPException(
                status_code=413,
                detail="PDF size cannot exceed 20 MB."
            )

        if not contents.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid PDF."
            )

        safe_name = re.sub(
            r"[^A-Za-z0-9._-]",
            "_",
            original_name
        ).strip("._")

        if not safe_name:
            safe_name = "uploaded_document.pdf"

        destination = UPLOAD_DIR / safe_name

        counter = 1

        while destination.exists():

            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix

            destination = UPLOAD_DIR / (
                f"{stem}_{counter}{suffix}"
            )

            counter += 1

        destination.write_bytes(contents)

        saved_path = destination

        result = add_uploaded_pdf(saved_path)

        documents = list_indexed_documents()

        document_info = next(
            (
                doc for doc in documents
                if doc["filename"] == saved_path.name
            ),
            None
        )

        return {
            "message": "PDF uploaded and indexed successfully!",
            "filename": result["filename"],
            "total_indexed_chunks": result["total_chunks"],
            "indexed_pages": (
                document_info["indexed_pages"]
                if document_info else 0
            ),
            "status": "indexed"
        }

    except HTTPException:
        raise

    except Exception as e:

        print("PDF UPLOAD ERROR:", str(e))

        if saved_path and saved_path.exists():

            try:
                saved_path.unlink()
                rebuild_indexes()

            except Exception as cleanup_error:
                print("INDEX CLEANUP ERROR:", str(cleanup_error))

        raise HTTPException(
            status_code=500,
            detail="PDF upload or indexing failed. Check backend logs."
        )

    finally:
        await file.close()


# LIST INDEXED DOCUMENTS

@app.get("/documents")
def get_documents():

    try:

        documents = list_indexed_documents()

        return {
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as e:

        print("DOCUMENT LIST ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve indexed documents."
        )


# IMAGE ANALYSIS

@app.post("/chat-image")
async def chat_image(
    question: str = Form(...),
    image: UploadFile = File(...)
):

    try:

        question = question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Please enter a question about the image."
            )

        if not image.filename:
            raise HTTPException(
                status_code=400,
                detail="Please select an image."
            )

        # CHECK IMAGE TYPE

        image_type = image.content_type

        if image_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, PNG and WebP images are supported."
            )

        # READ IMAGE INTO MEMORY

        image_bytes = await image.read(MAX_IMAGE_SIZE + 1)

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The image is empty."
            )

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image size cannot exceed 10 MB."
            )

        # ENCODE IMAGE AS BASE64

        encoded_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        image_data_url = (
            f"data:{image_type};base64,{encoded_image}"
        )

        # SEND IMAGE + QUESTION TO VISION MODEL

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are CampusAI, a helpful AI assistant. "
                                "Analyze the attached image carefully "
                                "and answer the user's question. "
                                "If text is visible, read it accurately. "
                                "Do not invent information that cannot "
                                "be seen in the image.\n\n"
                                f"User question: {question}"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_completion_tokens=2048
        )

        answer = (
            response.choices[0].message.content or ""
        )

        return {
            "question": question,
            "answer": answer,
            "image_name": Path(image.filename).name,
            "analysis_type": "image",
            "stored_permanently": False
        }

    except HTTPException:
        raise

    except Exception as e:

        print("IMAGE ANALYSIS ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Image analysis failed. Check backend logs."
        )

    finally:
        await image.close()
  