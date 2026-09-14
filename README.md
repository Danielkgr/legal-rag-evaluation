<div align="center">

# Fair Work Act & Modern Awards RAG System

A **Retrieval-Augmented Generation (RAG)** system purpose-built for Australian workplace law - designed to handle the unique challenges of legal document retrieval including heavy cross-referencing, defined-term lookups, and interdependent statutory provisions.

**What it does not do:** it gives no legal advice and is not a production tool. It ships no documents - you supply your own PDFs. The long CLI transcript below is illustrative interface flow with placeholder values; the [recorded local run](#-recorded-local-run) is real output from two Commonwealth Acts, with its inputs and caveats written down in [results/PROVENANCE.md](results/PROVENANCE.md).

**Maturity:** working prototype. A minimal test suite (10 tests) covers the import path, the chat client and the metric conventions; the pipeline itself is verified by running it. Embeddings and chat generation each work in two ways: against OpenAI, or against any local OpenAI-compatible server, so it can run end to end with no commercial key.

**[Features](#-features)** · **[Architecture](#-architecture)** · **[Quick Start](#-quick-start)** · **[Usage](#-usage)** · **[Evaluation](#-evaluation)** · **[Recorded run](#-recorded-local-run)**

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

- **Legal-aware document processing** - PDF parser detects section headings, subsection markers, and definition clauses to produce semantically coherent chunks.
- **Hybrid vector + BM25 search** - in-memory embedding similarity (OpenAI embeddings) combined with rank-bm25 keyword matching via reciprocal rank fusion.
- **Cross-reference & definition extraction** - automatic detection of statutory cross-references and defined terms per chunk.
- **Interactive chat interface** - answers from a local Gemma 4 (12B) model (CUDA, Apple MPS, or CPU) **or** any OpenAI-compatible chat server; embeddings from OpenAI or a local embedding server.
- **Comprehensive evaluation framework** - generates fact-based, hypothetical, and cross-referencing test queries; reports Precision@k, Recall@k, MRR, and MAP.

---

## 🏗️ Architecture

```
legal-rag-evaluation/
├── src/
│   ├── data_preprocessing/          # Document ingestion pipeline
│   │   ├── pdf_parser.py            # Structure-preserving PDF parsing (pdfplumber + PyMuPDF)
│   │   ├── chunking.py              # Legal-aware chunker preserving section boundaries
│   │   └── metadata_extractor.py    # Cross-reference & definition detection
│   ├── embedding/                   # Embedding & index management
│   │   └── __init__.py              # EmbeddingModel (OpenAI text-embedding-3-large), EmbeddingManager (in-memory vectors, JSON index)
│   ├── retrieval/                   # Hybrid search engine
│   │   └── __init__.py              # HybridRetriever — dense vectors + rank-bm25, reciprocal rank fusion
│   ├── llm/                         # Language model layer
│   │   └── __init__.py              # GemmaLLM (local Gemma 4 12B), OpenAIChatLLM (any OpenAI-compatible server), get_llm(), LegalChatBot
│   ├── evaluation/                  # Test generation & scoring
│   │   ├── eval_generator.py        # Fact / hypothetical / cross-ref query synthesis
│   │   └── eval_metrics.py          # Precision, recall, MRR, MAP implementation
│   ├── pipeline.py                  # Main orchestration class (CLI entry-point)
│   ├── chat.py                      # Interactive REPL chat interface
│   └── evaluate.py                  # Evaluation runner (generates + scores test sets)
├── data/
│   ├── raw/                         # Your input PDF documents (gitignored - not shipped)
│   └── processed/                   # Output: chunked JSON + embedding index (JSON) under processed/embeddings/
├── evaluation_set/                  # Output: generated eval queries + annotations (gitignored)
├── results/                         # Checked-in artefacts from the recorded local run + PROVENANCE.md
├── tests/                           # Import, chat-client and metric-convention tests
├── requirements.txt                 # Runtime dependencies
├── requirements-dev.txt             # Test-only dependencies
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Your own PDFs** - the repository ships no documents; put the legislation you want to index in `data/raw/` (gitignored)
- A source for embeddings and chat answers, **either**:
  - **Hosted:** an OpenAI API key (embeddings from `text-embedding-3-large`); or
  - **Fully local:** any OpenAI-compatible embedding server and any OpenAI-compatible chat server. No commercial key is needed.
- **GPU (only for the local Gemma path)** - the transformers-backed Gemma 4 (12B) model runs on CUDA, Apple MPS, or CPU (slow). The hosted path and the local-server path need no GPU on this machine.

### 1. Install Dependencies

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

The dev requirements only add `pytest`; the test suite imports the pipeline without `torch`/`transformers`, so it runs on a plain Python install.

### 2. Choose a backend

Point the OpenAI client at a server with the standard environment variables (local servers ignore the key, so a dummy value is fine):

```bash
# Option A - hosted OpenAI for both embeddings and chat
export OPENAI_API_KEY='sk-...'

# Option B - fully local: embeddings and chat from OpenAI-compatible servers
export OPENAI_API_KEY=local
export OPENAI_BASE_URL=http://localhost:10001/v1   # embeddings
export CHAT_BASE_URL=http://localhost:10009/v1     # chat generation
export CHAT_MODEL=<model-name-served-there>
```

With `CHAT_BASE_URL` set, generation uses the OpenAI-compatible client; left unset it falls back to the local Gemma 4 (12B) model. Embeddings always come from `OPENAI_BASE_URL` (which defaults to OpenAI).

### 3. Run the tests

```bash
pytest
```

---

## 📖 Usage

> The transcripts in this section are illustrative examples of the interface flow - file names, sizes, page/chunk counts, scores and answers are placeholders, not a recorded run.

### Interactive CLI (Recommended)

The fastest way to get started is the built-in interactive terminal interface - no flags to memorise, just type a number and go.

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

#### Processing PDFs - Step by Step

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

Option **2 (Query)** - pick an index, type a question, get results instantly:

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

Option **4 (Evaluate)** orchestrates the full pipeline: document processing, index building, evaluation-set generation, retrieval, and metric scoring.

The block below is **illustrative sample output** - it shows the shape of the report the runner prints, with placeholder values. No real-run numbers are published in this repository; a real run needs your own PDFs plus an OpenAI API key (see [Quick Start](#-quick-start)).

```text
Number of test queries to generate [100]: 100
Output directory (processed chunks & index) [src/data/processed]: src/data/processed
Evaluation output directory [evaluation_set]: evaluation_set
...
Running evaluation with 100 queries …

============================================================
EVALUATION REPORT
============================================================

Document Summary:
  Documents processed: N
  Chunks created: N
  Queries generated: 100
  Annotations: N

Query Type Distribution:
  fact: N
  hypothetical: N
  cross-reference: N

Retrieval Metrics:

  Precision, Recall, F1 by K:
    K=1:
      P@1: 0.000
      R@1: 0.000
      F1@1: 0.000
    K=3:
      P@3: 0.000
      R@3: 0.000
      F1@3: 0.000
    K=5:
      P@5: 0.000
      R@5: 0.000
      F1@5: 0.000
    K=10:
      P@10: 0.000
      R@10: 0.000
      F1@10: 0.000

  Mean Reciprocal Rank (MRR): 0.000
  Mean Average Precision (MAP): 0.000

  Overall Precision@10: 0.000
  Overall Recall@10: 0.000

============================================================

✗ Precision below target (0.0% < 90%)
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

### Process Documents

Convert raw PDFs into searchable chunks and build the index:

```bash
python src/pipeline.py \
    --mode process \
    --pdfs data/raw/fair_work_act_2009.pdf \
    --output-dir data/processed
```

**Output:** `data/processed/chunks.json` + embedding index at `data/processed/embeddings/fairwork_index_index.json`.

The PDF path above is an example - point `--pdfs` at your own documents in `data/raw/`.

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

> **Note:** Chat answers come from the local Gemma 4 (12B) model by default, or from any OpenAI-compatible chat server when `CHAT_BASE_URL` is set (see [Choose a backend](#2-choose-a-backend)).

---

## 📊 Evaluation

### Run Evaluation

```bash
python src/evaluate.py \
    --pdfs data/raw/fair_work_act_2009.pdf \
    --num-queries 100 \
    --eval-dir evaluation_set
```

This generates three categories of test queries with automatically derived relevance annotations (computed from the indexed chunks, not human-verified):

| Query Type | Proportion | Description |
|---|---|---|
| **Fact-based** | 40 % | Definition lookups, specific section questions, numeric thresholds |
| **Hypothetical scenarios** | 35 % | Applied workplace situations requiring statutory interpretation |
| **Cross-referencing** | 25 % | Multi-provision lookups that chain between sections |

### Metrics Reported

| Metric | Description |
|---|---|
| **Precision@k** (k=1, 3, 5, 10) | Fraction of retrieved docs that are relevant. The runner's 90% target applies to Overall Precision@10. |
| **Recall@k** | Fraction of all relevant docs successfully retrieved |
| **MRR** | Mean Reciprocal Rank of the first relevant result |
| **MAP** | Mean Average Precision across all ranks |

---

## 🧪 Recorded local run

Unlike the illustrative transcript above, this is a real run, done on one machine
with no commercial key. The full inputs, the compilation numbers of the two Acts
and the reproduction commands are written down in
[results/PROVENANCE.md](results/PROVENANCE.md); the small output files are checked
in under `results/`.

**Corpus:** two Commonwealth Acts, chosen because they are short and genuinely
distinct. Indexing them produced **247 chunks** (104 from the Spam Act 2003, 143
from the Do Not Call Register Act 2006).

**Retrieval (document routing).** Eight cross-Act questions were written by hand
*before* any retrieval and each labelled with the Act that answers it. The probe
only asks whether the hybrid retriever sends a query to the right Act:

| Measure | Result |
|---|---|
| Correct Act at rank 1 | 7 / 8 |
| Correct Act within top 3 | 8 / 8 |

The single rank-1 miss is a civil-penalty question; both Acts impose civil
penalties, so the top hit is defensibly ambiguous and the right Act still appears
at ranks 2 and 3. See `results/retrieval_probe.json`. This is document-level
routing on eight points; it is **not** a benchmark and says nothing about
section-level or open-domain accuracy.

**Generation (grounded chat).** Two questions went through the full retrieve →
prompt → generate path against a local OpenAI-compatible model. One produced a
correctly cited answer (Do Not Call Register Act 2006 s 11 for the prohibition, s
10 for the outline). The other declined: the top chunks did not surface the
operative provision, so the assistant reported that instead of inventing a
citation. See `results/chat_demo.json`.

**Why the built-in evaluation numbers are not the headline.** `evaluate.py`
creates its test queries from the same chunks it then scores, and labels
relevance by shallow lexical overlap, so it grades retrieval against labels drawn
from the very overlap the retriever rewards. It also scores a query with no
relevant chunk as recall 1.0 and MAP 1.0 by convention. The printed figures
(recall@1 0.60, MAP 0.62, precision@1 0.36 for this run) therefore describe the
generator, not retrieval quality. The three metric tests in `tests/` pin this
convention down so it stays documented rather than surprising.

---

## ⚙️ Configuration

### CLI flags

| Flag | Command | Default | Description |
|---|---|---|---|
| `--pdfs` | `pipeline.py` (process mode), `evaluate.py` | required | PDF files to process |
| `--mode` | `pipeline.py` | `process` | `process`, `query`, or `chat` |
| `--query` | `pipeline.py` | - | Search query (query mode) |
| `--k` | `pipeline.py` | `10` | Number of results returned in query mode |
| `--data-dir` | `pipeline.py`, `evaluate.py`, `chat.py` | `data` | Base data directory |
| `--output-dir` | `pipeline.py`, `evaluate.py`, `chat.py` | `data/processed` | Directory for chunk and index output |
| `--load-index` | `pipeline.py`, `chat.py` | off | Load an existing index instead of building one |
| `--index-name` | `pipeline.py`, `chat.py` | `fairwork_index` | Embedding index name (saved as `<name>_index.json`) |
| `--num-queries` | `evaluate.py` | `100` | Number of test queries to generate |
| `--eval-dir` | `evaluate.py` | `evaluation_set` | Directory for the generated evaluation set and report |
| `--no-print` | `evaluate.py` | off | Skip printing the report |

### In code

`FairWorkRAGPipeline` also accepts `embedding_model` (default `text-embedding-3-large`), `retrieval_alpha` and `retrieval_beta` (default `0.5` each) when instantiated directly. The embedding model is honoured; the alpha/beta weights are not yet plumbed through - the retriever is currently built with hardcoded `alpha=beta=0.5`, and the default fusion mode (reciprocal rank fusion) does not use the weights at all.

---

## 🧩 Key Design Decisions

### Why hybrid retrieval?
Legal documents require both **semantic understanding** ("what is unfair dismissal?") and **exact term matching** ("section 340(1)"). Pure vector search excels at the former; BM25 dominates the latter. Combining them gives robust coverage across query types.

### Why legal-aware chunking?
Naive fixed-size chunking breaks provisions mid-section, severs cross-references from their targets, and creates chunks whose meaning depends on context outside the boundary. The legal-aware chunker uses detected heading hierarchy to produce sections as single, self-contained units.

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with:** OpenAI · rank-bm25 · Gemma 4 · Transformers · pdfplumber · PyMuPDF

</div>
