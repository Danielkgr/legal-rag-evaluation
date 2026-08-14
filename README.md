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

### Interactive CLI (Recommended)

The fastest way to get started is the built-in interactive terminal interface — no flags to memorise, just type a number and go.

```bash
python src/cli.py
```

#### Main Menu

Every time you launch the CLI you're greeted with:

```text
  ───────────────────────────────────────────────────────
         FAIR WORK ACT & AWARDS — RAG SYSTEM
    Legal document retrieval, made easy.

   Select an action below to get started.
  ───────────────────────────────────────────────────────

  PROCESS        — Ingest PDF docs → build retrieval index
  QUERY          — Search the index for relevant passages
  CHAT           — Interactive Q&A with sourced answers
  EVALUATE       — Run full evaluation (generate + score)

What would you like to do?
  ›
```

#### Processing PDFs — Step by Step

Option **1 (Process)** walks you through everything:

```text
  ───────────────────────────────────────────────────────
   WHERE SHOULD WE FIND YOUR PDF DOCUMENTS?
  ───────────────────────────────────────────────────────
    src/data/raw (default)
    Browse for folder …
    Type custom path …

Where should we find your PDF documents?
  › src/data/raw

  Found 3 PDF file(s):
  ───────────────────────────────────────────────────────
    1. fair_work_act_2009.pdf (458KB)
    2. award_north_australia_medical.pdf (182KB)
    3. modern_award_transport.csv (12KB)
  ───────────────────────────────────────────────────────

Choose PDFs (comma-separated numbers, e.g. 1,3 or 'a' for all):
    All files
    1. fair_work_act_2009.pdf
    2. award_north_australia_medical.pdf
    3. modern_award_transport.csv

Which OpenAI embedding model?
    text-embedding-3-large (default, best quality)
    text-embedding-3-small (faster)
    text-embedding-ada-002 (legacy)
    Other (type custom)

Output directory [src/data/processed]: src/data/processed

  Processing 2 PDF(s) …

14:32:01 - INFO - Processing: data/raw/fair_work_act_2009.pdf
14:32:01 - INFO -   Parsed 312 pages
14:32:01 - INFO -   Created 1,847 chunks
14:32:01 - INFO - Processing: data/raw/award_north_australia.pdf
14:32:01 - INFO -   Parsed 89 pages
14:32:01 - INFO -   Created 423 chunks
14:32:02 - INFO - Building embedding index...
14:32:05 - INFO - Added 2,270 chunks to embedding index
14:32:05 - INFO - Index built successfully

✓ Processed 2,270 chunks and built index
```

#### Querying the Index

Option **2 (Query)** — pick an index, type a question, get results instantly:

```text
How should we access the retrieval index?
    Load existing index (fairwork_index)
    Build a fresh index from PDFs

Enter your search query: Who is entitled to annual leave under the Fair Work Act?
Number of results (k) [10]: 5

Search results for: 'Who is entitled to annual leave under the Fair Work Act?'
============================================================

[1] fair_work_act_2009 - Section 87
    Type: provision
    Score: 0.947
    Text: An employee who is not a casual employee is entitled to...

[2] fair_work_act_2009 - Section 88
    Type: provision
    Score: 0.831
    Text: Annual leave accumulates during each year of employment...
```

#### Interactive Chat

Option **3 (Chat)** launches the RAG-powered conversational interface:

```text
  ───────────────────────────────────────────────────────
   FAIR WORK ACT & AWARDS RAG CHAT
============================================================
Ask questions about Australian workplace law.
Commands: /help, /clear, /history, /rag <on|off>, /quit

You: What's the minimum notice period for long-term employees?

Assistant: Under s 119 of the Fair Work Act 2009, the minimum
notice period for an employee with continuous service of more
than 3 years is **one week** (s 119(2)(c)). For service of
more than 5 years it increases to two weeks.

Sources:
  [1] fair_work_act_2009 - Section 119
      Score: 0.912
  [2] fair_work_act_2009 - Section 120
      Score: 0.784
```

#### Running Evaluation

Option **4 (Evaluate)** orchestrates the full pipeline:

```text
Number of test queries to generate [100]: 100
Output directory [src/data/processed]: src/data/processed
Evaluation output directory [evaluation_set]: evaluation_set
...
Running evaluation with 100 queries …

============================================================
EVALUATION REPORT
============================================================

Document Summary:
  Documents processed: 2
  Chunks created: 2,270
  Queries generated: 100
  Annotations: 380

Retrieval Metrics:
  K=1:    P@1: 0.745  R@1: 0.312  F1@1: 0.441
  K=3:    P@3: 0.621  R@3: 0.589  F1@3: 0.605
  K=5:    P@5: 0.558  R@5: 0.712  F1@5: 0.624
  K=10:   P@10: 0.489  R@10: 0.853  F1@10: 0.620

  Mean Reciprocal Rank (MRR): 0.712
  Mean Average Precision (MAP): 0.648

  ✓ Precision target met (74.5% >= 90%)
```

#### Quick Reference

| CLI Option | Shortcut Key | Description                              |
|:----------:|:------------:|------------------------------------------|
| **Process** | `1` or `p`  | Select PDFs → build retrieval index      |
| **Query**   | `2` or `q`  | Search the index for relevant passages   |
| **Chat**    | `3` or `c`  | Interactive RAG-powered Q&A session      |
| **Evaluate**| `4` or `e`  | Full evaluation (generate + score)       |
| Quit        | `q` or `0`  | Exit the CLI                             |

> **Pro tip:** You can also run the quick-launch wrapper from the project root:
> ```bash
> ./fairwork
> ```
> Or install as a system command with `pip install -e .` and type `fairwork-rag`.

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
