# LLM-Powered-Fact-Checker-with-Custom-Embedding-Based-Retrieval

LLM-Powered Fact Checker (RAG System)

This project is a small fact-checking tool that takes a short news claim and checks whether it’s true, false, or unverifiable. It does this by comparing the claim with a set of trusted factual statements that you provide in a CSV file.

The system uses three main steps:

Find similar facts using text embeddings

Retrieve the top evidence from a vector database

Ask a local LLM (Gemma 2B via Ollama) to reason and give a verdict

Everything runs locally — no API keys, no external calls.

What This Project Does

You enter any short claim (news headline, WhatsApp forward, social post).

The app searches your facts.csv file for similar statements.

The local LLM evaluates the claim using the retrieved evidence.

You get a JSON result with a verdict:

Likely True

Likely False

Unverifiable

You also see the exact evidence used and a short explanation.

📁 Project Structure
project-folder/
│
├── app.py          # Streamlit app (main working RAG system)
├── facts.csv       # Your factual data (placed directly in the project folder)
├── vectorstore/    # Auto-created Chroma DB folder
└── README.md

⚙️ How to Set Up
1. Install requirements

Recommended to create a virtual environment:

python -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

2. Pull Ollama models

Make sure Ollama is installed, then run:

ollama pull nomic-embed-text
ollama pull gemma2:2b

3. Add your facts file

Create or copy a facts.csv file in the same folder as app.py.

Example:

statement
Government launched a solar subsidy scheme in 2024.
RBI increased repo rate by 25 basis points in 2023.
India announced a digital crop insurance mission in 2025.


The system automatically reads the first column.

Run the App

Use:

streamlit run app.py


Your browser will open at:

http://localhost:8501

How to Use

Enter a claim (example:
“India announced free electricity for all farmers from July 2025.”)

The app retrieves the most relevant statements from your CSV.

It sends the claim + evidence to the LLM.

You instantly get:

{
  "verdict": "Unverifiable",
  "evidence": [],
  "reasoning": "No evidence was found in the fact database to support the claim."
}

What This Shows

This project demonstrates:

How to build a Retrieval-Augmented Generation (RAG) pipeline

How to use embeddings + vector search

How to use a local LLM for reasoning

Clean, modular ML/LLM engineering workflow

Practical fact-checking logic
