# Agentic RAG System

An agentic Retrieval-Augmented Generation (RAG) pipeline built with **LangGraph** and **LangChain**. Questions are answered by a compiled LangGraph workflow that retrieves context from a local **ChromaDB** vector store, with embeddings generated locally through **Ollama** and LLM calls routed through **OpenRouter**.

## Features

- **Graph-orchestrated RAG**: the question-answering flow is a compiled LangGraph application (`graph/graph.py`), invoked with a single `{"question": ...}` input.
- **Local embeddings**: documents and queries are embedded with `nomic-embed-text` via Ollama, so the retrieval layer runs on your machine.
- **Persistent vector store**: ChromaDB collection (`rag-chroma`) stored on disk in `.chroma/`, exposed to the graph as a retriever.
- **Flexible LLM access**: chat models are called through OpenRouter using `langchain-openrouter`.
- **Web search tool**: Tavily integration for questions that need information beyond the local knowledge base.
- **Tracing**: LangSmith support for inspecting graph runs.

## Architecture

```
main.py  ──►  graph/graph.py (compiled LangGraph app)  ──►  retriever (ingestion.py)
                                                              │
                                                              ▼
                                          ChromaDB  ◄──  Ollama embeddings (nomic-embed-text)
```

`graph.png` in the repository root is a rendered diagram of the compiled graph.

## Knowledge base

The vector store is seeded from three blog posts by Lilian Weng:

- LLM-powered autonomous agents
- Prompt engineering
- Adversarial attacks on LLMs

Documents are loaded from the web, split into 250-token chunks (no overlap) with `RecursiveCharacterTextSplitter` using a tiktoken encoder, embedded, and stored in Chroma.

## Project structure

```
agentic-rag-system/
├── main.py            # Entry point: invokes the compiled graph
├── ingestion.py       # Document loading, chunking, embedding, and retriever setup
├── graph/             # LangGraph workflow definition
├── graph.png          # Visualization of the graph
├── .chroma/           # Persisted Chroma vector store
├── pyproject.toml     # Dependencies (managed with uv)
├── uv.lock
└── .python-version
```

## Tech stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM framework | LangChain |
| Vector store | ChromaDB (`langchain-chroma`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| LLM provider | OpenRouter (`langchain-openrouter`) |
| Web search | Tavily |
| Observability | LangSmith |
| Tooling | uv, pytest, black, isort |

## Getting started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally
- API keys for OpenRouter and Tavily (LangSmith is optional)

### Installation

```bash
git clone https://github.com/O4Osman/agentic-rag-system.git
cd agentic-rag-system
uv sync
ollama pull nomic-embed-text
```

### Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
```

### Run

```bash
uv run main.py
```

By default `main.py` asks the graph: *"What is agent memory?"*. Edit the `question` value in `main.py` to ask something else.

### Rebuilding the vector store

The repository ships with a pre-built index in `.chroma/`. To re-index from scratch, uncomment the document loading and `Chroma.from_documents(...)` blocks in `ingestion.py`, then run:

```bash
uv run ingestion.py
```

To use your own knowledge base, replace the URLs in `ingestion.py` with your own sources.

## Development

```bash
uv run pytest        # run tests
uv run black .       # format
uv run isort .       # sort imports
```

## License

Add a license file (for example MIT) before publishing this project for reuse.
