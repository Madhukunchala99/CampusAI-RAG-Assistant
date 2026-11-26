
# ==================================================
# STEP 1: IMPORT LIBRARIES AND INITIALIZE GROQ
# ==================================================

import os
import time
import numpy as np
import faiss

from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
def normalize_text(text):
    text = text.lower()
    text = " ".join(text.split())
    return text


# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found!")

# Initialize Groq client
client = Groq(api_key=api_key)

print("Groq client initialized successfully!")


# ==================================================
# STEP 2: LOAD PDF AND EXTRACT TEXT
# ==================================================

pdf = PdfReader(
    "documents/College_Regulations_and_Student_Data.pdf"
)

all_text = ""

for page in pdf.pages:
    text = page.extract_text()

    if text:
        all_text += text + "\n"

print("\nPDF TEXT EXTRACTION SUCCESSFUL!")
print("Total characters:", len(all_text))


# ==================================================
# STEP 3: SPLIT TEXT INTO CHUNKS
# ==================================================

chunk_size = 500
chunk_overlap = 50

chunks = []

start = 0

while start < len(all_text):

    end = start + chunk_size

    chunk = all_text[start:end]

    chunks.append(chunk)

    start += chunk_size - chunk_overlap


print("\nTOTAL CHUNKS:", len(chunks))

for i, chunk in enumerate(chunks):

    print(f"\nCHUNK {i + 1}:")
    print(chunk)


# ==================================================
# STEP 4: GENERATE EMBEDDINGS
# ==================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

normalized_chunks = [normalize_text(chunk) for chunk in chunks]

embeddings = model.encode(normalized_chunks)

embeddings = np.array(
    embeddings
).astype("float32")

print("\nEMBEDDINGS GENERATED SUCCESSFULLY")

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)


# ==================================================
# STEP 5: STORE EMBEDDINGS USING FAISS
# ==================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nFAISS INDEX CREATED SUCCESSFULLY")

print("Total vectors:", index.ntotal)
print("Vector dimension:", dimension)

# Save FAISS index
faiss.write_index(index, "faiss_index.index")

print("FAISS INDEX SAVED SUCCESSFULLY")



# ==================================================
# STEP 6: SEMANTIC SEARCH / RETRIEVAL
# ==================================================

question = input("\nAsk your college assistant: ")

normalized_question = normalize_text(question)

question_embedding = model.encode([normalized_question])

question_embedding = np.array(
    question_embedding
).astype("float32")

k = min(3, len(chunks))

distances, indices = index.search(
    question_embedding, k
)

print("\nRETRIEVED CHUNKS")

retrieved_chunks = []

for i, idx in enumerate(indices[0]):

    if idx == -1:
        continue

    retrieved_chunks.append(chunks[int(idx)])

    print(f"\nRESULT {i + 1}")
    print("Distance:", distances[0][i])
    print("Content:", chunks[int(idx)])

context = "\n".join(retrieved_chunks)


# ==================================================
# STEP 7: PREPARE HYBRID RAG PROMPT
# ==================================================

prompt = f"""
You are an intelligent College Assistant.

You have access to:
1. Retrieved college document context.
2. Your general knowledge as an AI language model.

Follow these rules carefully:

RULE 1:
If the question is about college regulations,
attendance, student records, examination policies,
or college-specific information, answer using ONLY
the provided document context.


RULE 2:
If the question is about college-specific information
and the answer is missing from the retrieved context:

1. Clearly mention that the information is not
   available in the uploaded document.

2. If relevant, provide a general explanation
   using your pretrained knowledge.

3. Clearly label the general explanation as
   general information, not an official college rule.

4. Never present general information as an
   official college-specific answer.


RULE 3:
If the question is a general educational or
technical question, you may use your general
knowledge to answer it.

RULE 4:
Do not confuse general knowledge with official
college information.

RULE 5:
If you use general knowledge, clearly mention
that the answer is based on general knowledge
rather than the uploaded document.

Retrieved Document Context:
{context}

User Question:
{question}

Answer:
"""

print("\nHYBRID PROMPT CREATED SUCCESSFULLY!")



# ==================================================
# STEP 8: GENERATE ANSWER USING GROQ
# ==================================================

try:

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    print("\nHYBRID RAG GENERATED ANSWER")
    print(answer)

except Exception as e:

    print("Groq API request failed:")
    print(e)

