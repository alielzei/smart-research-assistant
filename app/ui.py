import streamlit as st
import os
import faiss
import pickle
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import tool
from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np

load_dotenv()

# Load configs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")
EMBEDDING_DIR = "embeddings"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load models and index
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(os.path.join(EMBEDDING_DIR, "faiss.index"))
with open(os.path.join(EMBEDDING_DIR, "doc_ids.pkl"), "rb") as f:
    doc_ids = pickle.load(f)

# Connect to Postgres
conn = psycopg2.connect(POSTGRES_URL)
cur = conn.cursor()


# Define tools
@tool
def semantic_search(query: str) -> str:
    """Searches document chunks using semantic similarity and returns relevant snippets."""
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding), k=5)
    results = []
    for idx in I[0]:
        cur.execute(
            "SELECT chunk FROM documents WHERE id = %s LIMIT 1", (doc_ids[idx],)
        )
        row = cur.fetchone()
        if row:
            results.append(row[0])
    return "\n---\n".join(results)


@tool
def get_metadata_by_title(title: str) -> str:
    """Retrieves document metadata (title, author, filename) from Postgres by title."""
    cur.execute(
        """
        SELECT DISTINCT title, author, filename FROM documents
        WHERE LOWER(title) LIKE LOWER(%s)
        LIMIT 5
    """,
        (f"%{title}%",),
    )
    rows = cur.fetchall()
    if not rows:
        return "No metadata found."
    return "\n".join([f"Title: {r[0]}, Author: {r[1]}, File: {r[2]}" for r in rows])


# Initialize agent
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
tools = [
    Tool.from_function(
        semantic_search,
        name="semantic_search",
        description="Searches document chunks using semantic similarity and returns relevant snippets.",
    ),
    Tool.from_function(
        get_metadata_by_title,
        name="get_metadata_by_title",
        description="Retrieves document metadata (title, author, filename) from Postgres by title.",
    ),
]
agent = initialize_agent(
    tools=tools, llm=llm, agent="zero-shot-react-description", verbose=False
)

# --- Streamlit UI ---
st.set_page_config(page_title="Smart Research Assistant", layout="centered")
st.title("🤖 Smart Research Assistant")
st.markdown("Ask anything about your documents (PDFs you've ingested).")

query = st.text_input(
    "Enter your question:",
    placeholder="e.g. What are the key ideas in 'Attention is All You Need'?",
)

if query:
    with st.spinner("Thinking..."):
        try:
            response = agent.run(query)
            st.success("Here's what I found:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
