"""
Digital Transformation of the Finance Function — RAG Knowledge Base.

Searches a curated library of peer-reviewed PDFs (loaded from static/library/)
using semantic search over recursive-character chunks and sentence-transformer
embeddings stored in ChromaDB. No API keys required.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBRARY_DIR = Path("static/library")
MANIFEST_PATH = LIBRARY_DIR / "manifest.json"
STATIC_URL_BASE = "app/static/library"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

SUGGESTED_QUERIES = [
    "What KPIs measure finance transformation success?",
    "How does RPA change the role of accountants?",
    "What is the impact of digitalization on management control?",
    "How do CFO traits influence digital transformation?",
    "Which cybersecurity risks affect accounting information systems?",
]

OOS_SIMILARITY_THRESHOLD = 0.35
OOS_NOTICE = (
    "**No strong matches in the corpus.** The library covers digital transformation "
    "of the finance function — try a related question, or check the **Library** page "
    "to see what's indexed."
)

PALETTE = {
    "navy": "#0B2545",
    "slate": "#1E3A5F",
    "gold": "#C9A96E",
    "off_white": "#F8FAFC",
    "charcoal": "#0F172A",
    "neutral": "#64748B",
    "success": "#16A34A",
    "alert": "#DC2626",
}

st.set_page_config(
    page_title="Finance Transformation RAG",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');

html, body, [class*="css"]  {{
    font-family: 'Inter', sans-serif;
}}

h1, h2, h3 {{
    font-family: 'Source Serif 4', serif;
    color: {PALETTE['navy']};
    letter-spacing: -0.01em;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {PALETTE['navy']} 0%, {PALETTE['slate']} 100%);
}}
[data-testid="stSidebar"] * {{
    color: {PALETTE['off_white']} !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {PALETTE['gold']} !important;
}}

.score-chip {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}}
.score-high   {{ background: #DCFCE7; color: {PALETTE['success']}; }}
.score-mid    {{ background: #FEF3C7; color: #B45309; }}
.score-low    {{ background: #FEE2E2; color: {PALETTE['alert']}; }}

.source-card {{
    border-left: 4px solid {PALETTE['gold']};
    padding: 12px 16px;
    background: #FFFFFF;
    border-radius: 4px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}}
.source-meta {{
    font-size: 0.85rem;
    color: {PALETTE['neutral']};
    margin-top: 6px;
}}
.kpi-card {{
    background: {PALETTE['navy']};
    color: {PALETTE['off_white']};
    padding: 18px 22px;
    border-radius: 8px;
    text-align: center;
}}
.kpi-card .value {{
    font-size: 2.2rem;
    font-weight: 700;
    color: {PALETTE['gold']};
    font-family: 'JetBrains Mono', monospace;
}}
.kpi-card .label {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}}

/* Buttons — pill shape, gold hover (scoped so sidebar buttons stay legible) */
.stButton > button {{
    border-radius: 999px;
    transition: all 0.15s ease;
}}
[data-testid="stMain"] .stButton > button {{
    background: #FFFFFF;
    color: {PALETTE['charcoal']};
    border: 1px solid #E2E8F0;
    font-size: 0.88rem;
    padding: 6px 14px;
    min-height: auto;
    font-weight: 500;
}}
[data-testid="stMain"] .stButton > button:hover {{
    border-color: {PALETTE['gold']};
    background: #FEF7E6;
    color: {PALETTE['charcoal']};
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Library discovery
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PaperMeta:
    filename: str
    title: str
    authors: str
    year: int | str
    journal: str
    doi: str
    subtopic: str
    confidence: str
    open_access: bool


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"papers": [], "open_slots": [], "sourcing_policy": ""}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def discover_pdfs() -> list[Path]:
    if not LIBRARY_DIR.exists():
        return []
    return sorted(LIBRARY_DIR.glob("*.pdf"))


def library_signature(pdfs: list[Path]) -> str:
    """Cache key — changes whenever a PDF is added/removed/replaced."""
    return "|".join(f"{p.name}:{p.stat().st_size}" for p in pdfs)


def papers_by_filename(manifest: dict) -> dict[str, PaperMeta]:
    out: dict[str, PaperMeta] = {}
    for p in manifest.get("papers", []):
        out[p["filename"]] = PaperMeta(
            filename=p["filename"],
            title=p.get("title", p["filename"]),
            authors=p.get("authors", ""),
            year=p.get("year", ""),
            journal=p.get("journal", ""),
            doi=p.get("doi", ""),
            subtopic=p.get("subtopic", ""),
            confidence=p.get("confidence", ""),
            open_access=p.get("open_access", False),
        )
    return out


# ──────────────────────────────────────────────────────────────────────
# Cached heavy resources
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def load_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Indexing PDF library...")
def build_vector_store(_signature: str):
    """Loads every PDF, splits into chunks, embeds, returns Chroma + chunk dicts.

    Re-runs only when the library signature changes (PDF added/removed/replaced).
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    manifest = load_manifest()
    paper_lookup = papers_by_filename(manifest)
    pdfs = discover_pdfs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts: list[str] = []
    metadatas: list[dict] = []
    chunks_per_doc: Counter = Counter()

    for pdf in pdfs:
        meta = paper_lookup.get(pdf.name)
        try:
            pages = PyPDFLoader(str(pdf)).load()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load {pdf.name}: {exc}")
            continue

        for page_doc in pages:
            page_index = page_doc.metadata.get("page", 0)
            for chunk in splitter.split_text(page_doc.page_content):
                if not chunk.strip():
                    continue
                texts.append(chunk)
                metadatas.append({
                    "source": pdf.name,
                    "page": int(page_index),
                    "display_page": int(page_index) + 1,
                    "title": meta.title if meta else pdf.stem,
                    "authors": meta.authors if meta else "",
                    "year": str(meta.year) if meta else "",
                    "journal": meta.journal if meta else "",
                    "subtopic": meta.subtopic if meta else "",
                })
                chunks_per_doc[pdf.name] += 1

    if not texts:
        return None, [], chunks_per_doc

    embeddings = load_embedding_model()
    vector_store = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        collection_name="finance_transformation",
    )

    chunk_records = [{"text": t, **m} for t, m in zip(texts, metadatas)]
    return vector_store, chunk_records, chunks_per_doc


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def similarity_class(distance: float) -> tuple[str, float]:
    """Convert Chroma distance → (css class, similarity-in-[0,1])."""
    similarity = max(0.0, 1.0 - float(distance))
    if similarity >= 0.65:
        return "score-high", similarity
    if similarity >= 0.40:
        return "score-mid", similarity
    return "score-low", similarity


def pdf_url(filename: str, page: int) -> str:
    return f"./{STATIC_URL_BASE}/{filename}#page={page}"


def use_suggested_query(query: str) -> None:
    """Callback: populate the search box and switch to the Search page."""
    st.session_state["search_query_input"] = query
    st.session_state["page_nav"] = "Search"


def render_suggested_chips(location: str) -> None:
    """Render SUGGESTED_QUERIES as clickable buttons in equal-width columns.

    `location` namespaces the widget keys so chips on Home and Search don't collide.
    """
    cols = st.columns(len(SUGGESTED_QUERIES))
    for i, q in enumerate(SUGGESTED_QUERIES):
        cols[i].button(
            q,
            key=f"sq_{location}_{i}",
            on_click=use_suggested_query,
            args=(q,),
            use_container_width=True,
        )


HISTORY_LIMIT = 5
HISTORY_LABEL_MAX = 40


def add_to_history(query: str) -> None:
    """Prepend `query` to the session-only search history (dedup, cap at HISTORY_LIMIT)."""
    if not query or not query.strip():
        return
    history = [q for q in st.session_state.get("query_history", []) if q != query]
    history.insert(0, query)
    st.session_state["query_history"] = history[:HISTORY_LIMIT]


def render_history_strip() -> None:
    """Render up to HISTORY_LIMIT recent-search buttons. Hidden when history is empty."""
    history = st.session_state.get("query_history", [])
    if not history:
        return
    st.caption("Recent")
    cols = st.columns(min(len(history), HISTORY_LIMIT))
    for i, q in enumerate(history[:HISTORY_LIMIT]):
        label = q if len(q) <= HISTORY_LABEL_MAX else q[: HISTORY_LABEL_MAX - 1] + "…"
        cols[i].button(
            label,
            key=f"hist_{i}",
            on_click=use_suggested_query,
            args=(q,),
            use_container_width=True,
            help=q,
        )


def render_empty_library_state() -> None:
    st.title("Knowledge base is empty")
    st.markdown(
        f"""
        No PDFs were found in `{LIBRARY_DIR}/`.

        **To get started:**
        1. Open `_DOWNLOAD_LIST.md` in the project root for the curated paper list.
        2. Download each PDF via your FAMNIT institutional access using the DOI link.
        3. Save each into `{LIBRARY_DIR}/` using the **target filename** specified in the manifest.
        4. Restart this app — papers appear automatically.
        """
    )
    manifest = load_manifest()
    if manifest.get("papers"):
        st.subheader("Expected library (from manifest.json)")
        df = pd.DataFrame(manifest["papers"])
        st.dataframe(
            df[["filename", "title", "authors", "year", "journal"]],
            use_container_width=True,
            hide_index=True,
        )


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────

manifest = load_manifest()
paper_lookup = papers_by_filename(manifest)
pdfs = discover_pdfs()

st.sidebar.markdown(
    "<h2 style='margin-top:0'>Finance Transformation RAG</h2>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Peer-reviewed knowledge base · ChromaDB + MiniLM-L6-v2")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Search", "Statistics", "Library"],
    label_visibility="collapsed",
    key="page_nav",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Sources")

if "active_sources" not in st.session_state:
    st.session_state.active_sources = {p.name: True for p in pdfs}

# Reconcile session state with current PDFs (handles add/remove between reruns)
for p in pdfs:
    st.session_state.active_sources.setdefault(p.name, True)
for stale in list(st.session_state.active_sources):
    if stale not in {p.name for p in pdfs}:
        del st.session_state.active_sources[stale]

if pdfs:
    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("All", use_container_width=True):
        for p in pdfs:
            st.session_state.active_sources[p.name] = True
            st.session_state[f"src_{p.name}"] = True
    if col_b.button("None", use_container_width=True):
        for p in pdfs:
            st.session_state.active_sources[p.name] = False
            st.session_state[f"src_{p.name}"] = False

    for p in pdfs:
        meta = paper_lookup.get(p.name)
        label = meta.title if meta else p.stem
        short = (label[:48] + "…") if len(label) > 50 else label
        st.session_state.active_sources[p.name] = st.sidebar.checkbox(
            short,
            value=st.session_state.active_sources.get(p.name, True),
            key=f"src_{p.name}",
            help=f"{meta.authors} ({meta.year})" if meta else None,
        )
else:
    st.sidebar.info("No PDFs in library yet. See README for setup.")


# ──────────────────────────────────────────────────────────────────────
# Build vector store
# ──────────────────────────────────────────────────────────────────────

if pdfs:
    vector_store, chunk_records, chunks_per_doc = build_vector_store(
        library_signature(pdfs)
    )
else:
    vector_store, chunk_records, chunks_per_doc = None, [], Counter()


# ──────────────────────────────────────────────────────────────────────
# HOME
# ──────────────────────────────────────────────────────────────────────

if page == "Home":
    st.title("Digital Transformation of the Finance Function")
    st.markdown(
        "<p style='font-size:1.1rem; color:#475569;'>A semantic search engine over a curated library of "
        "peer-reviewed academic literature for CFOs, controllers, and transformation leads.</p>",
        unsafe_allow_html=True,
    )

    if not pdfs:
        st.warning("Library is empty — see the *Library* page for setup instructions.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"<div class='kpi-card'><div class='value'>{len(pdfs)}</div>"
                f"<div class='label'>PDFs indexed</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='kpi-card'><div class='value'>{len(chunk_records)}</div>"
                f"<div class='label'>chunks</div></div>",
                unsafe_allow_html=True,
            )
        with c3:
            subtopics = {m["subtopic"] for m in chunk_records if m.get("subtopic")}
            st.markdown(
                f"<div class='kpi-card'><div class='value'>{len(subtopics)}</div>"
                f"<div class='label'>sub-topics</div></div>",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"<div class='kpi-card'><div class='value'>{CHUNK_SIZE}/{CHUNK_OVERLAP}</div>"
                f"<div class='label'>chunk / overlap</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### How it works")
    st.markdown(
        """
        1. **Curated corpus.** Peer-reviewed papers from *Accounting Horizons*, *International Journal of Accounting
           Information Systems*, *European Accounting Review*, *Journal of Emerging Technologies in Accounting*, and
           others — sourced via institutional library access.
        2. **Recursive chunking.** Each PDF is split with LangChain's `RecursiveCharacterTextSplitter` (paragraph →
           sentence → word → character separators) at `chunk_size=300`, `chunk_overlap=50`.
        3. **Embeddings.** Each chunk is encoded by the `all-MiniLM-L6-v2` sentence-transformer (384 dims).
        4. **Vector search.** Queries are encoded with the same model and matched by cosine similarity in ChromaDB.
        5. **Source-anchored results.** Every result links to the originating PDF at the exact page.
        """
    )

    st.markdown("### Try a query")
    st.caption("Click any suggestion to run it.")
    render_suggested_chips("home")


# ──────────────────────────────────────────────────────────────────────
# SEARCH
# ──────────────────────────────────────────────────────────────────────

elif page == "Search":
    st.title("Search the literature")

    if vector_store is None:
        render_empty_library_state()
    else:
        active = [name for name, on in st.session_state.active_sources.items() if on]
        st.caption(
            f"Searching across **{len(active)}** of **{len(pdfs)}** sources · "
            f"toggle sources in the sidebar to refine"
        )

        if not active:
            st.warning("No sources are active. Tick at least one PDF in the sidebar.")
        else:
            if not st.session_state.get("search_query_input"):
                st.caption("Try a suggestion:")
                render_suggested_chips("search")
            render_history_strip()
            query = st.text_input(
                "Your question",
                placeholder="e.g. What KPIs measure finance transformation success?",
                key="search_query_input",
            )
            num_results = st.slider("Number of results", 1, 10, 5)

            if query:
                add_to_history(query)
                where_filter = {"source": {"$in": active}} if len(active) < len(pdfs) else None
                with st.spinner("Searching..."):
                    if where_filter:
                        results = vector_store.similarity_search_with_score(
                            query, k=num_results, filter=where_filter
                        )
                    else:
                        results = vector_store.similarity_search_with_score(
                            query, k=num_results
                        )

                if not results:
                    st.warning("No results — try a different query or activate more sources.")

                if results:
                    max_similarity = max(
                        (max(0.0, 1.0 - float(score)) for _, score in results),
                        default=0.0,
                    )
                    if max_similarity < OOS_SIMILARITY_THRESHOLD:
                        st.warning(OOS_NOTICE)

                for i, (doc, distance) in enumerate(results, 1):
                    css_class, similarity = similarity_class(distance)
                    md = doc.metadata
                    src_url = pdf_url(md["source"], md["display_page"])

                    st.markdown(
                        f"""
                        <div class="source-card">
                          <div style="display:flex; justify-content:space-between; align-items:baseline;">
                            <strong>Result {i}</strong>
                            <span class="score-chip {css_class}">relevance {similarity:.2f}</span>
                          </div>
                          <p style="margin:10px 0; line-height:1.55;">{doc.page_content}</p>
                          <div class="source-meta">
                            <strong>{md.get('title', md['source'])}</strong> · {md.get('authors', '')}
                            ({md.get('year', '')}) · <em>{md.get('journal', '')}</em><br/>
                            Page {md['display_page']} ·
                            <a href="{src_url}" target="_blank">📄 Open PDF at p.{md['display_page']}</a>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ──────────────────────────────────────────────────────────────────────
# STATISTICS
# ──────────────────────────────────────────────────────────────────────

elif page == "Statistics":
    st.title("Library statistics")

    if not chunk_records:
        render_empty_library_state()
    else:
        lengths = [len(c["text"]) for c in chunk_records]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total chunks", len(chunk_records))
        c2.metric("Avg chunk size", f"{int(np.mean(lengths))} chars")
        c3.metric("Min chunk size", f"{min(lengths)} chars")
        c4.metric("Max chunk size", f"{max(lengths)} chars")

        st.subheader("Chunks per source PDF")
        df_per_doc = (
            pd.DataFrame(
                [{"source": k, "chunks": v} for k, v in chunks_per_doc.items()]
            )
            .sort_values("chunks", ascending=True)
        )
        st.bar_chart(df_per_doc.set_index("source"))

        st.subheader("Distribution of chunk lengths")
        df_hist = pd.DataFrame({"chunk_size": lengths})
        st.bar_chart(
            df_hist["chunk_size"].value_counts(bins=15, sort=False).sort_index()
        )

        st.subheader("Coverage by sub-topic")
        subtopic_counts = Counter(c["subtopic"] for c in chunk_records if c.get("subtopic"))
        if subtopic_counts:
            df_sub = pd.DataFrame(
                [{"sub-topic": k, "chunks": v} for k, v in subtopic_counts.items()]
            ).sort_values("chunks", ascending=True)
            st.bar_chart(df_sub.set_index("sub-topic"))


# ──────────────────────────────────────────────────────────────────────
# LIBRARY
# ──────────────────────────────────────────────────────────────────────

elif page == "Library":
    st.title("Knowledge library")
    st.caption(manifest.get("sourcing_policy", ""))

    if not pdfs:
        render_empty_library_state()
    else:
        active = sum(1 for v in st.session_state.active_sources.values() if v)
        st.markdown(
            f"**{len(pdfs)}** PDFs in library · **{active}** currently active in search · "
            f"**{len(chunk_records)}** total chunks indexed"
        )

        rows = []
        for p in pdfs:
            meta = paper_lookup.get(p.name)
            rows.append({
                "Active": st.session_state.active_sources.get(p.name, True),
                "Title": meta.title if meta else p.stem,
                "Authors": meta.authors if meta else "",
                "Year": meta.year if meta else "",
                "Journal": meta.journal if meta else "",
                "Sub-topic": meta.subtopic if meta else "",
                "DOI": f"https://doi.org/{meta.doi}" if meta and meta.doi else "",
                "Chunks": chunks_per_doc.get(p.name, 0),
                "File": p.name,
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Active": st.column_config.CheckboxColumn(disabled=True),
                "DOI": st.column_config.LinkColumn(display_text="open"),
            },
        )

        st.markdown("---")
        st.markdown("### Open slots (sub-topics still to fill)")
        open_slots = manifest.get("open_slots", [])
        if open_slots:
            for slot in open_slots:
                st.markdown(
                    f"- **{slot['subtopic']}** — search: `{slot.get('search_query', '')}`  \n"
                    f"  Preferred journals: *{', '.join(slot.get('preferred_journals', []))}*"
                )
        else:
            st.success("All sub-topics covered.")

        st.markdown("---")
        st.markdown("### Add a new paper")
        st.markdown(
            "1. Drop the PDF into `static/library/` using a filename like "
            "`NN_firstauthor_year_short-title.pdf`.\n"
            "2. Add an entry to `static/library/manifest.json` with title, authors, year, "
            "journal, DOI, and sub-topic.\n"
            "3. Restart the app — the paper appears in the sidebar with a checkbox."
        )
