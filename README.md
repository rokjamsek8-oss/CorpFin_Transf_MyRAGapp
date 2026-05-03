# Digital Transformation of the Finance Function — RAG

A semantic search engine over a curated library of **peer-reviewed academic papers** on the
digital transformation of the finance function. Built for CFOs, controllers, and transformation
leads who want to query the literature in plain English.

Built with **Streamlit**, **LangChain**, **ChromaDB**, and the `all-MiniLM-L6-v2` sentence-transformer.
No API keys required.

## Features

- Semantic search across PDF chunks with cosine similarity
- Source citations link directly to the originating PDF at the exact page
- Sidebar toggles to include/exclude individual papers from the search
- Statistics page (total chunks, distribution, sub-topic coverage)
- Library page with full citation metadata and DOI links
- Drop a new PDF into `static/library/`, restart the app, and it's indexed

## Source corpus

The library consists of peer-reviewed papers from journals including:

- *Accounting Horizons* (American Accounting Association)
- *International Journal of Accounting Information Systems* (Elsevier)
- *European Accounting Review* (Taylor & Francis)
- *Journal of Emerging Technologies in Accounting* (AAA)
- *Meditari Accountancy Research* (Emerald)
- *Humanities and Social Sciences Communications* (Nature, OA)

**Sourcing policy:** PDFs are accessed via institutional library subscription (FAMNIT / NUK
consortium) under the publishers' standard academic-research license. They are stored locally
and served by the deployed app for the purpose of academic research and coursework, and are
displayed as short retrieval snippets only.

The full citation list with DOIs lives in `static/library/manifest.json`. Open `_DOWNLOAD_LIST.md`
(local-only) for a step-by-step download checklist.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The app degrades gracefully if `static/library/` is empty — the Home
and Library pages will tell you what to do next.

## Project layout

```
my-rag-app/
├── app.py                       # Streamlit app — 4 pages
├── requirements.txt             # Python dependencies (chromadb pinned <0.6)
├── render.yaml                  # Render.com deploy config
├── .gitignore
├── README.md
├── .streamlit/
│   └── config.toml              # Static-serving + theme
└── static/
    └── library/
        ├── manifest.json        # Metadata for every paper in the corpus
        └── *.pdf                # The peer-reviewed corpus (added locally)
```

## Architecture notes

- **Chunking:** `RecursiveCharacterTextSplitter`, `chunk_size=300`, `chunk_overlap=50`,
  separators `["\n\n", "\n", ". ", " ", ""]`. Chosen for sharp top-1 retrieval on
  150–500-word academic-paper paragraphs.
- **Embeddings:** `all-MiniLM-L6-v2` (90 MB, 384-dim). Cached on first run.
- **Vector store:** ChromaDB in-memory. The vector store rebuilds when the library signature
  (filenames + sizes) changes — adding a PDF triggers exactly one re-index.
- **Render free-tier sizing:** 512 MB RAM. Tactics applied: `@st.cache_resource` for the model
  and the index; deferred imports inside cached functions; single-thread tokenizers; in-memory
  Chroma (no disk persistence).

## Deploy to Render

1. Push this repo to GitHub.
2. Create a new **Web Service** on [Render](https://render.com/), pointed at the repo.
3. Render reads `render.yaml`, installs from `requirements.txt`, and starts Streamlit.
4. First cold start downloads the embedding model (~90 MB) and indexes the library — allow
   ~30–60 seconds. Subsequent loads hit the in-process cache.

## License

The application code is released under the MIT License. The PDF corpus is **not** redistributed —
each PDF remains under its publisher's copyright and is downloaded directly by the deployer
via institutional access.
