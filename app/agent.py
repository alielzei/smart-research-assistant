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

# Configs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")
EMBEDDING_DIR = "embeddings"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load embedding model and FAISS
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(os.path.join(EMBEDDING_DIR, "faiss.index"))
with open(os.path.join(EMBEDDING_DIR, "doc_ids.pkl"), "rb") as f:
    doc_ids = pickle.load(f)

# Connect to Postgres
conn = psycopg2.connect(POSTGRES_URL)
cur = conn.cursor()

# --- Define LangChain Tools ---

@tool
def semantic_search(query: str) -> str:
    """Searches document chunks using semantic similarity and returns relevant snippets."""
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding), k=5)
    results = []
    for idx in I[0]:
        cur.execute("SELECT chunk FROM documents WHERE id = %s LIMIT 1", (doc_ids[idx],))
        row = cur.fetchone()
        if row:
            results.append(row[0])
    return "\n---\n".join(results)

@tool
def get_metadata_by_title(title: str) -> str:
    """Retrieves document metadata (title, author, filename) from Postgres by title."""
    cur.execute("""
        SELECT DISTINCT title, author, filename FROM documents
        WHERE LOWER(title) LIKE LOWER(%s)
        LIMIT 5
    """, (f"%{title}%",))
    rows = cur.fetchall()
    if not rows:
        return "No metadata found."
    return "\n".join([f"Title: {r[0]}, Author: {r[1]}, File: {r[2]}" for r in rows])

# --- Setup Agent ---
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

tools = [
    Tool.from_function(
        semantic_search,
        name="semantic_search",
        description="Searches document chunks using semantic similarity and returns relevant snippets."
    ),
    Tool.from_function(
        get_metadata_by_title,
        name="get_metadata_by_title",
        description="Retrieves document metadata (title, author, filename) from Postgres by title."
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

def main():
    print("🤖 Smart Research Assistant ready. Ask me something!")
    while True:
        try:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                break
            response = agent.run(query)
            print(f"\nAssistant:\n{response}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
