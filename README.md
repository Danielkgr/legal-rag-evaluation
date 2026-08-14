# Fair Work Act & Modern Awards RAG System

<div align="center">

A **production-grade Retrieval-Augmented Generation (RAG)** system purpose-built for Australian workplace law — engineered to handle the unique challenges of legal document retrieval including heavy cross-referencing, defined-term lookups, and interdependent statutory provisions.

**[Features](#-features)** · **[Architecture](#-architecture)** · **[Quick Start](#-quick-start)** · **[Usage](#-usage)** · **[Evaluation](#-evaluation)** · **[Project Structure](#-project-structure)**

</div>

---

## 📋 Overview

Legal documents present distinct challenges for RAG systems: provisions that reference one another, defined terms whose meanings span pages, and clauses that only make sense in context. This system is designed to address those challenges head-on with a **legal-aware pipeline** built around four pillars:

| Pillar | Approach |
|---|---|
| **Legal-Aware Chunking** | Preserves section boundaries, heading hierarchies, and cross-reference links during document splitting |
| **Hybrid Retrieval** | Combines dense vector similarity with sparse BM25 keyword matching for maximal recall without sacrificing precision |
| **Cross-Reference Tracking** | Extracts and indexes explicit references between provisions (e.g. *"see s 35"*) so related sections surface together |
| **Defined-Term Indexing** | Identifies statutory definitions at chunk creation time, enabling accurate term-resolution at query time |

---

## ✨ Features

- **Legal-aware document processing** — PDF parser detects section headings, subsection markers, and definition clauses to produce semantically coherent chunks.
- **Hybrid vector + BM25 search** — tunable blend of embedding similarity and keyword matching (`alpha` / `beta` weights) via ChromaDB.
- **Cross-reference & definition extraction** — automatic detection of statutory cross-references and defined terms per chunk.
- **Interactive chat interface** — powered by Gemma 4 (12B) for local inference with an optional fallback to the OpenAI API.
- **Comprehensive evaluation framework** — generates fact-based, hypothetical, and cross-referencing test queries; reports Precision@k, Recall@k, MRR, and MAP.

---

## 🏗️ Architecture

```
fairwork-rag/
├── src/
│   ├── data_preprocessing/          # Document ingestion pipeline
│   │   ├── pdf_parser.py            # Structure-preserving PDF parsing (pdfplumber + PyMuPDF)
│   │   ├── chunking.py              # Legal-aware chunker preserving section boundaries
│   │   └── metadata_extractor.py    # Cross-reference & definition detection
│   ├── embedding/                   # Embedding & index management
│   │   ├── openai_embeddings        # OpenAI text-embedding-3-large (default)
│   │   └── sentence_transformers    # Fallback: local embeddings via sentence-transformers
│   ├── retrieval/                   # Hybrid search engine
│   │   ├── vector_search            # ChromaDB dense-vector retrieval
│   │   └── bm25_search              # rank-bm25 sparse retrieval
│   ├── llm/                         # Language model layer
│   │   ├── gemma_inference          # Google Gemma 4 (12B) via Transformers + bitsandbytes
│   │   └── chat_bot.py              # RAG-enriched conversation agent
│   ├── evaluation/                  # Test generation & scoring
│   │   ├── eval_generator.py        # Fact / hypothetical / cross-ref query synthesis
│   │   └── eval_metrics.py          # Precision, recall, MRR, MAP implementation
│   ├── pipeline.py                  # Main orchestration class (CLI entry-point)
│   ├── chat.py                      # Interactive REPL chat interface
│   └── evaluate.py                  # Evaluation runner (generates + scores test sets)
├── data/
│   ├── raw/                         # Input PDF documents
│   ├── processed/                   # Chunked JSON, ChromaDB index, embeddings
│   └── eval_set/                    # Generated evaluation datasets
├── evaluation_set/                  # Output: generated eval queries + annotations
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── CHANGELOG.md                     # Version history (when available)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **OpenAI API key** — for embeddings (`text-embedding-3-large`)
- **GPU (optional)** — required only for local Gemma 4 inference; CPU works with the OpenAI LLM fallback

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
export OPENAI_API_KEY='sk-...'
```

---

## 📖 Usage

### Process Documents

Convert raw PDFs into searchable chunks and build the index:

```bash
python src/pipeline.py \
    --mode process \
    --pdfs data/raw/fair_work_act_2009.pdf \
    --output-dir data/processed
```

**Output:** `data/processed/chunks.json` + ChromaDB index (`fairwork_index/`).

### Search (Query Mode)

Run a keyword / vector search directly from the CLI:

```bash
python src/pipeline.py \
    --mode query \
    --query "What is the definition of employee?" \
    --k 10 \
    --load-index
```

### Interactive Chat

Start a conversational session backed by retrieved legal context:

```bash
python src/chat.py --load-index
```

> **Tip:** Set `GEMMA_ENABLED=true` in your environment to use the local Gemma 4 model instead of the OpenAI API for answer generation.

---

## 📊 Evaluation

### Run Evaluation

```bash
python src/evaluate.py \
    --pdfs data/raw/fair_work_act_2009.pdf \
    --num-queries 100 \
    --output-dir evaluation_set
```

This generates three categories of test queries with gold-standard relevance annotations:

| Query Type | Proportion | Description |
|---|---|---|
| **Fact-based** | 40 % | Definition lookups, specific section questions, numeric thresholds |
| **Hypothetical scenarios** | 35 % | Applied workplace situations requiring statutory interpretation |
| **Cross-referencing** | 25 % | Multi-provision lookups that chain between sections |

### Metrics Reported

| Metric | Description |
|---|---|
| **Precision@k** (k=5, 10) | Fraction of retrieved docs that are relevant — *target ≥ 90%* |
| **Recall@k** | Fraction of all relevant docs successfully retrieved |
| **MRR** | Mean Reciprocal Rank of the first relevant result |
| **MAP** | Mean Average Precision across all ranks |

---

## ⚙️ Configuration

The pipeline accepts several tuning parameters via CLI flags or environment variables:

| Parameter | Default | Description |
|---|---|---|
| `--embedding-model` | `text-embedding-3-large` | OpenAI embedding model identifier |
| `--retrieval-alpha` | `0.5` | Weight for vector (dense) search component |
| `--retrieval-beta` | `0.5` | Weight for BM25 (sparse) search component |
| `--output-dir` | `data/processed` | Directory for index and chunk output |

> **Note:** Set `alpha + beta = 1.0` for best results. Increase `alpha` for semantic similarity; increase `beta` for exact keyword recall.

---

## 🧩 Key Design Decisions

### Why hybrid retrieval?
Legal documents require both **semantic understanding** ("what is unfair dismissal?") and **exact term matching** ("section 340(1)"). Pure vector search excels at the former; BM25 dominates the latter. Combining them gives robust coverage across query types.

### Why legal-aware chunking?
Naive fixed-size chunking breaks provisions mid-section, severs cross-references from their targets, and creates chunks whose meaning depends on context outside the boundary. The legal-aware chunker uses detected heading hierarchy to produce sections as single, self-contained units.

---

## 📝 License

Internal project — see `LICENSE` for details.

---

<div align="center">

**Built with:** OpenAI · ChromaDB · rank-bm25 · Gemma 4 · pdfplumber · PyMuPDF

</div>
