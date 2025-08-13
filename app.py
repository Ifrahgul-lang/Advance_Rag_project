import streamlit as st
import os, time, json, pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False
from pypdf import PdfReader
from docx import Document
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
from textblob import TextBlob
from langdetect import detect, DetectorFactory
import langid

DetectorFactory.seed = 0

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
EMB_CACHE = CACHE_DIR / "embeddings.pkl"
CHAT_CACHE = CACHE_DIR / "chat_history.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if "docs" not in st.session_state:
    st.session_state.docs = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model" not in st.session_state:
    st.session_state.model = None
if "top_k" not in st.session_state:
    st.session_state.top_k = 4

def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

def save_pickle(p: Path, obj):
    try:
        with open(p, "wb") as fh:
            pickle.dump(obj, fh)
    except Exception:
        pass

def load_pickle(p: Path):
    if p.exists():
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None
    return None

def detect_language(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "unknown"
    try:
        lang = detect(text)
        if lang and lang != "und":
            return lang
    except Exception:
        pass
    try:
        lang, _ = langid.classify(text)
        return lang
    except Exception:
        return "unknown"

def load_model(name=MODEL_NAME):
    if st.session_state.model is None:
        with st.spinner(f"Loading model {name}..."):
            st.session_state.model = SentenceTransformer(name)
    return st.session_state.model

def normalize_vecs(vecs: np.ndarray):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

def embed_texts(texts, show_progress=True):
    m = load_model()
    vecs = m.encode(texts, show_progress_bar=show_progress)
    vecs = np.array(vecs).astype("float32")
    return normalize_vecs(vecs)

def build_index(vectors: np.ndarray):
    if vectors is None:
        return None
    if _FAISS_AVAILABLE:
        dim = vectors.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(vectors)
        st.session_state.index = idx
        return idx
    else:
        st.session_state.index = {"vectors": vectors}
        return st.session_state.index

def search_index(q_vec: np.ndarray, k=4):
    if st.session_state.index is None:
        return np.array([]), np.array([])
    if _FAISS_AVAILABLE:
        D, I = st.session_state.index.search(q_vec, k)
        return D, I
    else:
        vectors = st.session_state.index.get("vectors")
        if vectors is None:
            return np.array([]), np.array([])
        sims = (vectors @ q_vec[0]).astype("float32")
        idxs = np.argsort(-sims)[:k]
        return sims[idxs].reshape(1, -1), idxs.reshape(1, -1)

def chunk_text(text, chunk_size=400, overlap=50):
    tokens = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i:i+chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

cached_docs, cached_vectors = [], None

if EMB_CACHE.exists():
    try:
        data = load_pickle(EMB_CACHE)
        if data and isinstance(data, dict):
            cached_docs = data.get("docs", [])
            cached_vectors = data.get("vectors", None)
    except Exception:
        pass

if cached_docs and not st.session_state.docs:
    st.session_state.docs = cached_docs
if cached_vectors is not None:
    st.session_state.embeddings = np.array(cached_vectors, dtype="float32")
    try:
        build_index(st.session_state.embeddings)
    except Exception:
        pass

saved_chat = load_pickle(CHAT_CACHE)
if saved_chat and not st.session_state.chat_history:
    st.session_state.chat_history = saved_chat

st.set_page_config(page_title="Advanced RAG App", layout="wide")
st.title("Advanced RAG — Universal RAG Toolkit")

with st.sidebar:
    st.header("Settings")
    font_color = st.color_picker("Font Color", "#111111")
    font_size = st.slider("Font Size", 12, 32, 16)
    bold = st.checkbox("Bold", False)
    italic = st.checkbox("Italic", False)
    underline = st.checkbox("Underline", False)
    top_k = st.slider("Top-K Retrieval", 1, 10, st.session_state.get("top_k", 4))
    st.session_state["top_k"] = top_k
    answer_lang = st.selectbox("Answer language (optional)", ["auto","en","es","fr","de","ar","hi","ur","zh-CN","ja","ru","pt"])
    st.caption("Similarity: cosine (normalized embeddings + FAISS IP index if available)")

    preload = st.button("Preload Embedding Model")

    if st.button("Clear Documents"):
        st.session_state.docs = []
        st.session_state.embeddings = None
        st.session_state.index = None
        try:
            if EMB_CACHE.exists(): EMB_CACHE.unlink()
        except Exception:
            pass
        st.success("Cleared docs and embeddings.")
        safe_rerun()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        try:
            if CHAT_CACHE.exists(): CHAT_CACHE.unlink()
        except Exception:
            pass
        st.success("Chat cleared.")
        safe_rerun()

    if st.button("Save Chat Now"):
        save_pickle(CHAT_CACHE, st.session_state.chat_history)
        st.success("Saved chat to disk.")

append_existing = st.sidebar.checkbox("Append to existing docs", value=False)

if preload:
    load_model()
    st.success("Embedding model preloaded.")

uploaded = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf","docx","txt"], accept_multiple_files=True)
if uploaded:
    if not append_existing:
        st.session_state.docs = []
    added = 0
    for f in uploaded:
        try:
            name = (f.name or "").lower()
            if name.endswith(".pdf"):
                reader = PdfReader(f)
                text = " ".join([(p.extract_text() or "") for p in reader.pages])
            elif name.endswith(".docx"):
                doc = Document(f)
                text = " ".join([p.text for p in doc.paragraphs])
            else:
                text = f.read().decode("utf-8", errors="ignore")
            chunks = chunk_text(text)
            st.session_state.docs.extend(chunks)
            added += len(chunks)
        except Exception as e:
            st.error(f"Failed to ingest {getattr(f,'name','file')}: {e}")
    if added:
        st.success(f"Added {added} chunks from uploaded files.")

yt_url = st.text_input("YouTube URL (optional)")
if st.button("Fetch YouTube Transcript"):
    try:
        if not yt_url.strip():
            raise ValueError("Empty URL")
        if "v=" in yt_url:
            vid = yt_url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in yt_url:
            vid = yt_url.split("/")[-1].split("?")[0]
        else:
            vid = yt_url.strip()
        transcripts = {}
        TRANSCRIPT_CACHE = CACHE_DIR / "yt_transcripts.json"
        if TRANSCRIPT_CACHE.exists():
            try:
                with open(TRANSCRIPT_CACHE, "r", encoding="utf-8") as fh:
                    transcripts = json.load(fh)
            except Exception:
                transcripts = {}
        if vid in transcripts:
            st.info("Transcript found in cache.")
            txt = transcripts[vid]
        else:
            raw = YouTubeTranscriptApi.get_transcript(vid)
            txt = " ".join([t.get("text","") for t in raw])
            transcripts[vid] = txt
            with open(TRANSCRIPT_CACHE, "w", encoding="utf-8") as fh:
                json.dump(transcripts, fh, ensure_ascii=False)
        chunks = chunk_text(txt)
        st.session_state.docs.extend(chunks)
        st.success(f"Added {len(chunks)} transcript chunks.")
    except Exception as e:
        st.error(f"Failed to fetch transcript: {e}")

if st.button("Build/Update Index"):
    if not st.session_state.docs:
        st.error("No documents to index.")
    else:
        with st.spinner("Embedding documents (incremental)..."):
            cached_docs, cached_vectors = [], None
            if EMB_CACHE.exists():
                data = load_pickle(EMB_CACHE)
                if data and isinstance(data, dict):
                    cached_docs = data.get("docs", [])
                    cached_vectors = data.get("vectors", None)
            start = 0
            vectors = None
            if cached_docs and cached_vectors is not None:
                if len(cached_docs) <= len(st.session_state.docs) and st.session_state.docs[:len(cached_docs)] == cached_docs:
                    start = len(cached_docs)
                    vectors = np.array(cached_vectors, dtype="float32")
            if start < len(st.session_state.docs):
                new_texts = st.session_state.docs[start:]
                new_vectors = embed_texts(new_texts)
                if vectors is None:
                    vectors = new_vectors
                else:
                    vectors = np.vstack([vectors, new_vectors])
            if vectors is None and cached_vectors is not None:
                vectors = np.array(cached_vectors, dtype="float32")
            if vectors is None:
                st.error("No vectors to index.")
            else:
                st.session_state.embeddings = vectors
                save_pickle(EMB_CACHE, {"docs": st.session_state.docs, "vectors": vectors})
                try:
                    build_index(vectors)
                    st.success("Index built and cached.")
                except Exception as e:
                    st.error(f"Failed to build index: {e}")

query = st.text_input("Ask a question from the indexed documents")
if st.button("Search"):
    if not query.strip():
        st.error("Please enter a query.")
    elif st.session_state.index is None:
        st.error("Index is empty. Build it first.")
    else:
        q_vec = embed_texts([query], show_progress=False)
        k = min(int(st.session_state.get("top_k", 4)), len(st.session_state.docs))
        D, I = search_index(q_vec, k)
        answers = []
        for dist, idx in zip(D[0], I[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(st.session_state.docs):
                continue
            snippet = st.session_state.docs[idx]
            polarity = TextBlob(snippet).sentiment.polarity
            lang = detect_language(snippet)
            translation = None
            if answer_lang and answer_lang != "auto":
                try:
                    translation = GoogleTranslator(source="auto", target=answer_lang).translate(snippet)
                except Exception:
                    translation = None
            display_text = translation if translation else snippet
            confidence = float(dist)
            answers.append({"snippet": display_text, "polarity": polarity, "confidence": confidence, "lang": lang})
        for a in answers:
            style = f"color:{font_color}; font-size:{font_size}px;"
            if bold: style += " font-weight:bold;"
            if italic: style += " font-style:italic;"
            if underline: style += " text-decoration:underline;"
            st.markdown(f"<div style='{style}'>📄 {a['snippet']}</div>", unsafe_allow_html=True)
            st.write(f"Detected language: {a['lang']} | Polarity: {a['polarity']:.2f} | Confidence: {a['confidence']:.3f}")
        st.session_state.chat_history.append({"q": query, "answers": answers, "time": time.time()})

st.subheader("Chat History")
if st.session_state.chat_history:
    for i, item in enumerate(reversed(st.session_state.chat_history)):
        st.write(f"Q: {item['q']}")
        for ans in item["answers"]:
            st.write(f"- {ans['snippet'][:400]}... (confidence {ans['confidence']:.2f})")
        if i >= 20:
            break
else:
    st.info("No chat history yet.")

col1, col2 = st.columns(2)
with col1:
    if st.button("Save Session"):
        save_pickle(CHAT_CACHE, st.session_state.chat_history)
        st.success("Session & chat saved to disk.")
with col2:
    if st.button("Load Session"):
        data = load_pickle(CHAT_CACHE)
        if data:
            st.session_state.chat_history = data
            st.success("Session loaded.")
            safe_rerun()
        else:
            st.error("No session file found.")

st.markdown("---")
st.caption("Advanced RAG App — caches in .cache; safe rerun supported. Multilingual detection (langdetect+langid), translation, cosine FAISS, styling controls.")
