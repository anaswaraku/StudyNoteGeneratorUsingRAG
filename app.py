# app_rag_groq.py
import os
import json
import streamlit as st
import pdfplumber
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Dict, Tuple

# ---------- CONFIG ----------
MODEL_NAME = "llama-3.1-8b-instant"  # change if desired
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast embedding model
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200
TOP_K = 5  # retrieved passages per topic

# ---------- Utils ----------
def extract_text_from_pdf(file_bytes) -> str:
    """Return the extracted text of a PDF file (bytes)."""
    text_parts = []
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
    return "\n\n".join(text_parts)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Chunk text into overlapping character chunks; return list of dicts with id & text."""
    chunks = []
    start = 0
    i = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"id": f"p{i:05d}", "text": chunk_text, "start": start, "end": end})
            i += 1
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build_faiss_index(passages: List[Dict], embedder: SentenceTransformer):
    texts = [p["text"] for p in passages]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_top_k(query: str, embedder: SentenceTransformer, index: faiss.IndexFlatL2, passages: List[Dict], k=TOP_K) -> List[Dict]:
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for i in I[0]:
        if i < len(passages):
            results.append(passages[i])
    return results

def weight_to_guidance(weight: float, total_marks: int = 100) -> Dict:
    marks = round(total_marks * weight / 100)
    # heuristic words_per_mark
    words_per_mark = 10 if marks >= 20 else 8
    recommended_words = max(50, marks * words_per_mark)
    if weight >= 25:
        structure = [
            {"section": "Introduction", "words": int(0.12*recommended_words)},
            {"section": "Main Body", "words": int(0.76*recommended_words)},
            {"section": "Conclusion", "words": int(0.12*recommended_words)},
        ]
    elif weight >= 15:
        structure = [
            {"section": "Introduction", "words": int(0.15*recommended_words)},
            {"section": "Main Body", "words": int(0.70*recommended_words)},
            {"section": "Conclusion", "words": int(0.15*recommended_words)},
        ]
    else:
        structure = [{"section": "Short answer / bullets", "words": recommended_words}]
    return {"suggested_marks": marks, "recommended_words": recommended_words, "structure": structure}

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Study Notes RAG (Groq + FAISS)", layout="wide")
st.title("📚 Study Notes Generator — RAG pipeline (Groq + FAISS)")

# API key handling
API_KEY = "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"
if not API_KEY:
    st.warning("GROQ_API_KEY not found in environment. Paste it for this session (will not be saved).")
    key_in = st.text_input("Groq API Key (paste)", type="password")
    if key_in:
        API_KEY = key_in
if not API_KEY:
    st.stop()

client = Groq(api_key=API_KEY)

# Left column: upload + syllabus
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1) Upload / Input")
    uploaded_file = st.file_uploader("Upload a PDF file (or skip to paste text)", type=["pdf"])
    pasted_text = st.text_area("Or paste reference text (optional)", height=200)

    st.markdown("---")
    st.header("2) Syllabus (topics + weightage)")
    st.markdown("Enter topics, one per line, with optional weight as percentage. Example:\n`Demand & Supply,25`")
    syllabus_raw = st.text_area("Syllabus (topic,weight%)", height=180, placeholder="Demand & Supply,25\nElasticity,20\nMarket Structures,25\nConsumer Theory,30")
    total_marks = st.number_input("Total marks for the paper (used for guidance)", min_value=10, value=100, step=10)

    st.markdown("---")
    st.markdown("Options:")
    retr_k = st.slider("Top-K passages to retrieve per topic", min_value=1, max_value=10, value=TOP_K)
    num_bullets = st.slider("Bullets per topic", 3, 10, 6)
    gen_mindmap = st.checkbox("Generate Mermaid mindmap?", value=True)

with col2:
    st.header("Output")
    output_area = st.empty()

# ---------- Main processing ----------
if st.button("Build RAG and Generate Notes"):
    # 1. Prepare reference text
    with st.spinner("Extracting text..."):
        if uploaded_file:
            try:
                reference_text = extract_text_from_pdf(uploaded_file)
            except Exception as e:
                st.error(f"PDF extraction error: {e}")
                reference_text = ""
        else:
            reference_text = pasted_text or ""

        if not reference_text.strip():
            st.error("No reference text found. Upload a PDF or paste text.")
            st.stop()

    # 2. Chunk
    with st.spinner("Chunking text..."):
        passages = chunk_text(reference_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        st.write(f"Created {len(passages)} passages (chunks). Showing first 3:")
        for p in passages[:3]:
            st.write(p["id"], p["text"][:250].replace("\n"," "), "...")

    # 3. Embedding model & FAISS index
    with st.spinner("Building embeddings (this may download a model first time)..."):
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        index, embeddings = build_faiss_index(passages, embedder)
        st.success("Vector index built.")

    # 4. Parse syllabus
    syllabus_lines = [line.strip() for line in syllabus_raw.splitlines() if line.strip()]
    syllabus = []
    for ln in syllabus_lines:
        if "," in ln:
            parts = ln.split(",")
            title = parts[0].strip()
            try:
                weight = float(parts[1].strip())
            except:
                weight = 0.0
        else:
            title = ln
            weight = 0.0
        syllabus.append({"title": title, "weight": weight})

    # 5. For each topic: RAG -> Groq
    results_all = {}
    for topic in syllabus:
        topic_title = topic["title"]
        topic_weight = topic["weight"]
        with st.spinner(f"Retrieving for topic: {topic_title}"):
            retrieved = retrieve_top_k(topic_title, embedder, index, passages, k=retr_k)
        # Build retrieval context text
        ctx = ""
        for r in retrieved:
            ctx += f"===PASSAGE_START {r['id']}===\n{r['text']}\n===PASSAGE_END===\n\n"

        # Compose prompt
        system_msg = (
            "You are an exam-prep assistant. Output strict JSON with keys: summary, bullets, keywords, mermaid_mindmap (optional), answer_guidance. "
            "Bullets must be short (8-18 words). Add source passage ids in square brackets at the end of factual bullets, like [p00001]."
        )
        guidance = weight_to_guidance(topic_weight, total_marks)
        user_prompt = f"""
TASK: For the topic: "{topic_title}" (weight {topic_weight}% of {total_marks} marks).
CONTEXT PASSAGES:
{ctx}

Produce a JSON with keys:
- summary: 1 sentence (max 25 words)
- bullets: an array with exactly {num_bullets} concise bullets (8-18 words each). End factual bullets with their source id in brackets.
- keywords: 5 keywords (array)
- mermaid_mindmap: a mermaid graph string that captures main nodes (only when requested)
- answer_guidance: include suggested_marks (int), recommended_words (approx int), structure (array with sections)

Use the following guidance mapping: {json.dumps(guidance)}

Return ONLY valid JSON.
"""

        # Call Groq chat completion
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=600,
                temperature=0.0,
            )
            assistant_text = resp.choices[0].message.content
        except Exception as e:
            st.error(f"Groq API error for topic {topic_title}: {e}")
            assistant_text = None

        # Parse JSON
        parsed = None
        if assistant_text:
            try:
                parsed = json.loads(assistant_text)
            except Exception:
                # attempt to extract JSON substring
                try:
                    start = assistant_text.index("{")
                    end = assistant_text.rindex("}") + 1
                    parsed = json.loads(assistant_text[start:end])
                except Exception:
                    st.warning(f"Could not parse JSON for topic {topic_title}. Showing raw output below.")
                    parsed = {"raw": assistant_text}

        results_all[topic_title] = {"topic": topic, "retrieved": retrieved, "model_out": parsed}

    # 6. Display results
    for t, info in results_all.items():
        st.header(f"Topic: {t}  (weight {info['topic']['weight']}%)")
        st.subheader("Retrieved passages:")
        for r in info["retrieved"]:
            st.markdown(f"**{r['id']}** — {r['text'][:300].replace(chr(10),' ')}...")

        st.subheader("Model Output")
        out = info["model_out"]
        if not out:
            st.write("No output.")
            continue

        if "raw" in out:
            st.code(out["raw"], language="text")
            continue

        st.markdown("**Summary**")
        st.write(out.get("summary",""))

        st.markdown("**Bullets**")
        bullets = out.get("bullets", [])
        if isinstance(bullets, list):
            for b in bullets:
                st.markdown(f"- {b}")
        else:
            for line in str(bullets).splitlines():
                if line.strip():
                    st.markdown(f"- {line.strip()}")

        st.markdown("**Keywords**")
        kws = out.get("keywords", [])
        if isinstance(kws, list):
            st.write(", ".join(kws))
        else:
            st.write(kws)

        if gen_mindmap:
            mm = out.get("mermaid_mindmap")
            if mm:
                st.markdown("**Mermaid mindmap (raw)**")
                st.code(mm, language="text")
                st.markdown("You can copy the mermaid text to an online mermaid renderer or render client-side in your app.")
        st.markdown("**Answer Guidance**")
        ag = out.get("answer_guidance", {})
        st.write(ag)

    st.success("RAG generation complete.")
