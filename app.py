import io, os, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import requests
import streamlit as st
import torch
from bs4 import BeautifulSoup
from ddgs import DDGS
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

APP_TITLE = "Local Agentic RAG + Web Search"
DEFAULT_LLM = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

@dataclass
class Chunk:
    text: str
    source_name: str
    chunk_id: int

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"\n[Page {page_num}]\n{text}")
    return "\n".join(parts).strip()

def read_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        doc = Document(tmp_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

def read_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try: return file_bytes.decode(encoding)
        except UnicodeDecodeError: pass
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    if suffix == ".pdf": return read_pdf(data)
    if suffix == ".docx": return read_docx(data)
    if suffix in {".txt", ".md", ".py", ".csv", ".json"}: return read_text(data)
    raise ValueError(f"Unsupported file type: {suffix}")

def chunk_text(text: str, source_name: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    text = " ".join(text.split())
    if not text: return []
    chunks, start, idx, n = [], 0, 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(text=chunk, source_name=source_name, chunk_id=idx))
            idx += 1
        if end == n: break
        start = max(0, end - overlap)
    return chunks

@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    return tokenizer, model

def build_faiss_index(embeddings: np.ndarray):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index

def embed_texts(embedder, texts: List[str]) -> np.ndarray:
    embs = embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    return embs.astype("float32")

def retrieve_docs(query: str, embedder, index, chunks: List[Chunk], top_k: int = 4):
    q_emb = embed_texts(embedder, [query])
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({"score": float(score), "chunk": chunks[int(idx)]})
    return results

def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")})
    return results

def fetch_webpage_text(url: str, max_chars: int = 5000) -> str:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 LocalAgent/1.0"}, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]): tag.extract()
    return " ".join(soup.get_text(separator=" ").split())[:max_chars]

def choose_tools(question: str, has_docs: bool):
    q = question.lower()
    web_signals = ["latest", "current", "today", "recent", "news", "web", "internet", "search", "who is", "what is happening"]
    doc_signals = ["document", "file", "pdf", "proposal", "uploaded", "report"]
    use_web = any(sig in q for sig in web_signals)
    use_docs = has_docs and (any(sig in q for sig in doc_signals) or not use_web)
    if has_docs and use_web: mode = "hybrid"
    elif use_web: mode = "web"
    elif has_docs: mode = "docs"
    else: mode = "direct"
    steps = {
        "docs": ["Search uploaded documents", "Answer from retrieved chunks"],
        "web": ["Search the web", "Fetch the top result if possible", "Answer with cited web sources"],
        "hybrid": ["Search uploaded documents", "Search the web", "Combine both into one grounded answer"],
        "direct": ["Answer directly from the local model"],
    }[mode]
    return {"mode": mode, "steps": steps}

def build_prompt(question: str, context_blocks: List[str], notes: str) -> str:
    context = "\n\n".join(context_blocks) if context_blocks else "No external context."
    notes_block = notes.strip() if notes.strip() else "No saved notes."
    system = "You are a careful local AI agent. Use provided context when available. Cite sources inline like [Doc 1] or [Web 2]. If context is missing, say so plainly."
    return f"""<|system|>
{system}
<|user|>
Saved notes:
{notes_block}

Context:
{context}

Question:
{question}

Give a concise grounded answer.
<|assistant|>
"""

def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 400, temperature: float = 0.2) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=temperature, top_p=0.9, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def reset_index():
    for key in ("chunks", "index", "doc_stats"):
        if key in st.session_state: del st.session_state[key]

def ensure_state():
    st.session_state.setdefault("agent_notes", "")
    st.session_state.setdefault("chat_log", [])

ensure_state()
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Local DeepSeek agent with document RAG, no-key web search, and lightweight planning/memory.")

with st.sidebar:
    st.header("Settings")
    llm_name = st.text_input("LLM model", value=DEFAULT_LLM)
    embed_model_name = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
    top_k = st.slider("Retrieved document chunks", 2, 8, 4)
    web_results_k = st.slider("Web results", 2, 8, 4)
    max_new_tokens = st.slider("Max new tokens", 128, 1024, 400, 32)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    show_trace = st.checkbox("Show agent trace", value=True)
    if st.button("Clear indexed docs"):
        reset_index(); st.success("Cleared indexed documents.")
    if st.button("Clear notes"):
        st.session_state["agent_notes"] = ""; st.success("Cleared notes.")
    if st.button("Clear chat"):
        st.session_state["chat_log"] = []; st.success("Cleared chat history.")

tab_chat, tab_notes = st.tabs(["Agent Chat", "Notes"])

with tab_notes:
    st.subheader("Agent Notes")
    st.session_state["agent_notes"] = st.text_area("Persistent scratchpad for this session", value=st.session_state["agent_notes"], height=220)
    st.download_button("Download notes", st.session_state["agent_notes"], file_name="agent_notes.md")

with tab_chat:
    uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt", "md", "py", "csv", "json"], accept_multiple_files=True)
    if uploaded_files and st.button("Build / Rebuild document index", type="primary"):
        all_chunks, doc_stats = [], []
        with st.spinner("Reading files and building embeddings..."):
            embedder = load_embedder(embed_model_name)
            for file in uploaded_files:
                try:
                    text = extract_text(file)
                    chunks = chunk_text(text, file.name)
                    all_chunks.extend(chunks)
                    doc_stats.append({"file": file.name, "chars": len(text), "chunks": len(chunks)})
                except Exception as exc:
                    st.error(f"Failed to process {file.name}: {exc}")
            if all_chunks:
                embeddings = embed_texts(embedder, [c.text for c in all_chunks])
                st.session_state["index"] = build_faiss_index(embeddings)
                st.session_state["chunks"] = all_chunks
                st.session_state["doc_stats"] = doc_stats
                st.success(f"Indexed {len(all_chunks)} chunks from {len(doc_stats)} file(s).")
            else:
                st.warning("No text could be indexed.")
    if "doc_stats" in st.session_state:
        with st.expander("Indexed files", expanded=False):
            st.table(st.session_state["doc_stats"])
    for role, msg in st.session_state["chat_log"]:
        with st.chat_message(role): st.write(msg)
    question = st.chat_input("Ask the agent a question")
    if question:
        st.session_state["chat_log"].append(("user", question))
        with st.chat_message("user"): st.write(question)
        has_docs = "index" in st.session_state and "chunks" in st.session_state
        plan = choose_tools(question, has_docs)
        trace = [f"Mode selected: {plan['mode']}"] + plan["steps"]
        context_blocks = []
        with st.chat_message("assistant"):
            with st.spinner("Agent is working..."):
                embedder = load_embedder(embed_model_name)
                tokenizer, model = load_llm(llm_name)
                if plan["mode"] in ("docs", "hybrid") and has_docs:
                    doc_hits = retrieve_docs(question, embedder, st.session_state["index"], st.session_state["chunks"], top_k=top_k)
                    for i, item in enumerate(doc_hits, start=1):
                        chunk = item["chunk"]
                        context_blocks.append(f"[Doc {i}: {chunk.source_name} | chunk {chunk.chunk_id}]\n{chunk.text}")
                    trace.append(f"Retrieved {len(doc_hits)} document chunks")
                if plan["mode"] in ("web", "hybrid"):
                    try:
                        web_hits = web_search(question, max_results=web_results_k)
                        for i, item in enumerate(web_hits, start=1):
                            context_blocks.append(f"[Web {i}: {item['title']} | {item['href']}]\nSnippet: {item['body']}")
                        trace.append(f"Retrieved {len(web_hits)} web results")
                        if web_hits:
                            try:
                                page_text = fetch_webpage_text(web_hits[0]["href"])
                                context_blocks.append(f"[Web page detail: {web_hits[0]['title']}]\n{page_text}")
                                trace.append("Fetched top web result")
                            except Exception as exc:
                                trace.append(f"Top page fetch skipped: {exc}")
                    except Exception as exc:
                        trace.append(f"Web search failed: {exc}")
                prompt = build_prompt(question, context_blocks, st.session_state["agent_notes"])
                answer = generate_answer(tokenizer, model, prompt, max_new_tokens=max_new_tokens, temperature=temperature) or "I could not generate an answer."
                st.write(answer)
                st.session_state["chat_log"].append(("assistant", answer))
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Add last answer to notes"):
                        st.session_state["agent_notes"] += "\n\n## Saved answer\n" + answer
                        st.success("Added to notes.")
                with c2:
                    if st.button("Add sources summary to notes"):
                        st.session_state["agent_notes"] += "\n\n## Saved sources\n" + "\n".join(context_blocks[:6])
                        st.success("Added sources to notes.")
            if show_trace:
                with st.expander("Agent trace", expanded=False):
                    for item in trace: st.write("- " + item)
            if context_blocks:
                with st.expander("Grounding context", expanded=False):
                    for block in context_blocks:
                        st.text(block[:2500]); st.markdown("---")

st.markdown("---")
st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
