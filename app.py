# app.py
import os
import json
import re
import chromadb
from dotenv import load_dotenv
import streamlit as st

# LangChain / Ollama
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# -------- CONFIG --------
FACTS_CSV = "facts.csv"     # CSV is kept in SAME folder as app.py
DB_DIR = "vectorstore"      # Chroma DB folder
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma2:2b"
TOP_K = 3
# ------------------------

st.set_page_config(page_title="Fact Checker (RAG)", layout="wide")
st.title("LLM-Powered Fact Checker — Ollama + Chroma (New API)")

# Ensure DB folder exists
os.makedirs(DB_DIR, exist_ok=True)

# ----- Build vectorstore (NEW CHROMA API) -----
def build_vectorstore_from_csv(csv_path):
    import pandas as pd

    if not os.path.exists(csv_path):
        st.error(f"facts.csv not found at: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    source_col = df.columns[0]   # auto-pick first column
    texts = df[source_col].astype(str).tolist()

    # Split long facts into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text("\n".join(texts))

    # Embeddings
    embed = OllamaEmbeddings(model=EMBED_MODEL)
    embeddings = embed.embed_documents(chunks)

    # Create Chroma persistent DB
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection("facts")

    # Insert vectors
    collection.upsert(
        ids=[f"id_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    st.success("Vectorstore created from CSV successfully!")
    return collection


# Load collection or build fresh
def load_or_build_vectorstore():
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        return client.get_collection("facts")
    except Exception:
        return build_vectorstore_from_csv(FACTS_CSV)


# Startup: load or build DB
with st.spinner("Loading vector database..."):
    vectorstore = load_or_build_vectorstore()

if vectorstore is None:
    st.stop()

# LLM (Ollama)
llm = Ollama(model=LLM_MODEL)

# UI Input
st.write("Enter a short news statement or claim to verify.")
input_text = st.text_input("Enter claim:", "")

if st.button("Check Claim") and input_text.strip():

    # -------- RETRIEVE --------
    embed = OllamaEmbeddings(model=EMBED_MODEL)
    query_embed = embed.embed_query(input_text)

    results = vectorstore.query(
        query_embeddings=[query_embed],
        n_results=TOP_K
    )

    docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(docs)

    st.subheader("Retrieved Evidence")
    if docs:
        for i, d in enumerate(docs):
            st.write(f"**{i+1}.** {d}")
    else:
        st.write("_No evidence found._")

    # -------- PROMPT --------
    prompt = (
        "You are a fact-checking assistant.\n\n"
        "Using ONLY the evidence below, evaluate the user's question and return EXACTLY one JSON object "
        "with keys: verdict, evidence, reasoning.\n\n"
        "Rules:\n"
        "- verdict must be one of: \"Likely True\", \"Likely False\", \"Unverifiable\"\n"
        "- evidence is an array of strings from the retrieved evidence\n"
        "- reasoning must be 1-2 sentences\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question: {input_text}\n\n"
        "Return ONLY the JSON object."
    )

    # -------- LLM CALL --------
    response = llm.invoke(prompt)
    raw_output = response.content if hasattr(response, "content") else str(response)

    st.subheader("Raw LLM Output")
    st.code(raw_output)

    # -------- JSON PARSE --------
    parsed = None
    try:
        parsed = json.loads(raw_output)
    except:
        match = re.search(r"\{.*\}", raw_output, flags=re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except:
                parsed = None

    if parsed:
        st.subheader("Final Verdict")
        st.json(parsed)
    else:
        st.error("Could not parse JSON from LLM output.")
