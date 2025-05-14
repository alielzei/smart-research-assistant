import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import uuid
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from tqdm import tqdm

from pathlib import Path

load_dotenv(override=True)

# Load config
DATA_DIR = Path("data/papers")
EMBEDDING_DIR = Path("embeddings")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
POSTGRES_URL = os.getenv("POSTGRES_URL")

print(POSTGRES_URL)

# Initialize embedding model
model = SentenceTransformer(MODEL_NAME)

# Prepare FAISS index
embedding_dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)
doc_ids = []

# Prepare PostgreSQL connection
conn = psycopg2.connect(POSTGRES_URL)
cur = conn.cursor()

# Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    title TEXT,
    author TEXT,
    filename TEXT,
    chunk TEXT
);
""")
conn.commit()

def extract_text_and_meta(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    metadata = doc.metadata
    return full_text, metadata

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_pdf(pdf_path):
    text, meta = extract_text_and_meta(pdf_path)
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)

    doc_id = str(uuid.uuid4())
    filename = os.path.basename(pdf_path)
    title = meta.get("title", filename)
    author = meta.get("author", "Unknown")

    # Add to FAISS
    index.add(embeddings)
    doc_ids.extend([doc_id] * len(embeddings))

    # Add metadata to Postgres
    rows = [(doc_id, title, author, filename, chunk) for chunk in chunks]
    execute_values(cur, """
        INSERT INTO documents (id, title, author, filename, chunk)
        VALUES %s
    """, rows)

    conn.commit()
    print(f"Ingested {filename} with {len(chunks)} chunks.")

def save_faiss_index():
    EMBEDDING_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(EMBEDDING_DIR / "faiss.index"))
    with open(EMBEDDING_DIR / "doc_ids.pkl", "wb") as f:
        pickle.dump(doc_ids, f)

def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    for pdf_path in tqdm(pdf_files):
        ingest_pdf(pdf_path)

    save_faiss_index()
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
