# StudyNoteGeneratorUsingRAG
Easy-to-run Streamlit app that demonstrates a practical RAG pipeline for study/education use-cases.
Converts uploaded PDFs (or pasted text) into concise study materials using a Retrieval-Augmented Generation (RAG) pipeline: 
PDF → chunk → embeddings → retrieval → LLM (Groq Llama-3). Produces: precise bullet notes, 1-line summary, keywords, mermaid mind-map and exam answer guidance based on syllabus weightage.

Tech Stack:
- Python, Streamlit
- Groq Llama-3 (RAG reasoning, notes generation)
- sentence-transformers + scikit-learn (embeddings + retrieval)
- pdfplumber (PDF text extraction)
- NumPy, pandas (processing)
