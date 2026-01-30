# MODEL_ID switched to Qwen2-1.5B-Instruct
MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

import os
import io
import re
import numpy as np
import torch
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

DEFAULT_PDF_PATH = "DPDP_Act_2023_Reference.pdf"   # default local file
MAX_CHUNK_CHARS = 800
TOP_K = 2
MAX_NEW_TOKENS = 200

# -----------------------------
# MODELS (embedder + Qwen LLM)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_models(model_id: str, max_new_tokens: int, temperature: float):
    # 1) Embedder for retrieval
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # 2) Qwen2-1.5B-Instruct for generation (CPU-friendly config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,     # keep CPU-friendly; if you have GPU, see note below
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )

    # Ensure pad token exists (some chat models only set eos)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # Keep generation params adjustable at call-time, not fixed here
        return_full_text=False,
    )

    return embedder, gen

# Tip: If you have a CUDA GPU and enough VRAM, you can change to:
#   model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
# And remove device_map={"": "cpu"} above.

embedder, llm = load_models(MODEL_ID, MAX_NEW_TOKENS, 0.4)

# -----------------------------
# PDF -> TEXT
# -----------------------------
def pdf_to_text(file_like) -> str:
    reader = PdfReader(file_like)
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

# -----------------------------
# TEXT -> CHUNKS
# -----------------------------
def text_to_chunks(text: str, max_chunk_chars: int = 800):
    cleaned = text.replace("\r", "\n")
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) + 1 <= max_chunk_chars:
            buf = (buf + "\n" + p).strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_chunk_chars:
                chunks.extend([p[i:i + max_chunk_chars] for i in range(0, len(p), max_chunk_chars)])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)

    return [c for c in chunks if c.strip()]

# -----------------------------
# EMBEDDING & RETRIEVAL
# -----------------------------
def normalize_2d(mat: np.ndarray):
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def embed_texts(texts):
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return normalize_2d(embs)

def retrieve(query, docs, emd, top_k=2):
    q = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
    q = q / (np.linalg.norm(q) + 1e-12)

    sims = emd @ q  # cosine similarity due to normalization
    idxs = np.argsort(sims)[-top_k:][::-1]
    return [docs[i] for i in idxs], sims[idxs]

# -----------------------------
# PROMPT BUILD (Qwen chat template aware)
# -----------------------------
def build_qwen_prompt(tokenizer, query: str, context: str) -> str:
    """
    Prefer Qwen's chat template if available; otherwise fallback to a plain prompt.
    """
    system_msg = (
        "You are a helpful assistant for question answering. "
        "Use only the provided context to answer. If the answer is not in the context, say: I don't know."
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {query}\nAnswer in 2-5 sentences:"

    # If tokenizer has a chat template, use it.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Plain text fallback
        return f"{system_msg}\n\n{user_msg}\n"

# -----------------------------
# PROMPT TRIM (avoid context overflow)
# -----------------------------
def truncate_prompt_to_context(prompt: str, max_new_tokens: int = 150, safety_margin: int = 32) -> str:
    tokenizer = llm.tokenizer
    model = llm.model

    # Prefer tokenizer.model_max_length; fallback to model config
    max_ctx = getattr(tokenizer, "model_max_length", None)
    if not max_ctx or max_ctx == int(1e30):  # some tokenizers set huge sentinel
        max_ctx = getattr(model.config, "max_position_embeddings", 4096)

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    allowed = max_ctx - max_new_tokens - safety_margin
    allowed = max(allowed, 128)

    if len(ids) > allowed:
        ids = ids[-allowed:]  # keep the tail (most relevant to RAG)
        prompt = tokenizer.decode(ids)

    return prompt

# -----------------------------
# OUTPUT CLEANER
# -----------------------------
def clean_generation(text: str) -> str:
    t = text.strip()

    # Remove duplicated answer sections or loops
    if "Answer:" in t:
        # Keep the part after first 'Answer:' if the model echoes the prompt
        t = t.split("Answer:", 1)[-1].strip() or t

    # De-duplicate lines
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    dedup, seen = [], set()
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            dedup.append(ln)
            seen.add(key)

    t = " ".join(dedup).strip()

    # If still too short, return safe fallback
    if len(t) < 10:
        return "I don't know based on the provided document."

    return t

# -----------------------------
# GENERATION (RAG)
# -----------------------------
def generate_answer(query, docs, emd, top_k=2, max_new_tokens=200):
    retrieved_docs, scores = retrieve(query, docs, emd, top_k=top_k)
    context = "\n\n---\n\n".join(retrieved_docs)

    prompt = build_qwen_prompt(llm.tokenizer, query, context)
    prompt = truncate_prompt_to_context(prompt, max_new_tokens=max_new_tokens)

    out = llm(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
        pad_token_id=llm.tokenizer.pad_token_id or llm.tokenizer.eos_token_id,
        eos_token_id=llm.tokenizer.eos_token_id,
        # You can set additional stopping criteria if needed
    )[0]["generated_text"]

    final = clean_generation(out)
    return final, retrieved_docs, scores

# -----------------------------
# UI
# -----------------------------
st.title("🤖 RAG Chatbot (DPDP Act / Any PDF)")
st.write(
    "Upload a PDF to ask questions about it. "
    "If you don't upload, I will use the default local file: "
    f"**{DEFAULT_PDF_PATH}**"
)

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# Choose file: uploaded OR default
if uploaded_file:
    file_bytes = uploaded_file.read()
    file_label = f"Uploaded: {uploaded_file.name}"
else:
    if not os.path.exists(DEFAULT_PDF_PATH):
        st.error(f"Default PDF not found: {DEFAULT_PDF_PATH}. Please upload a PDF.")
        st.stop()
    with open(DEFAULT_PDF_PATH, "rb") as f:
        file_bytes = f.read()
    file_label = f"Default: {os.path.basename(DEFAULT_PDF_PATH)}"

# Extract text
try:
    text = pdf_to_text(io.BytesIO(file_bytes))
except Exception as e:
    st.error(f"Failed to read PDF: {e}")
    st.stop()

if not text.strip():
    st.error("No extractable text found in this PDF. Please try another file.")
    st.stop()

# Chunk + embed
docs = text_to_chunks(text, max_chunk_chars=MAX_CHUNK_CHARS)
if not docs:
    st.error("Couldn't create chunks from the PDF.")
    st.stop()

with st.spinner("Indexing document (embeddings)..."):
    emd = embed_texts(docs)

st.success(f"✅ Ready! 📄 {file_label} | {len(docs)} chunks")

# Ask question
user_query = st.text_input("Enter your question:")
if user_query:
    with st.spinner("Thinking..."):
        answer, sources, scores = generate_answer(
            user_query, docs, emd, top_k=TOP_K, max_new_tokens=MAX_NEW_TOKENS
        )

    st.subheader("🧠 Answer")
    st.write(answer)

    st.subheader("📚 Retrieved Context")
    for i, (src, sc) in enumerate(zip(sources, scores), start=1):
        preview = src[:400].replace("\n", " ")
        st.markdown(f"- **Context {i} (score={float(sc):.3f})**: {preview}{'...' if len(src) > 400 else ''}")