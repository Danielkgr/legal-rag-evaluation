# Fair Work Act & Modern Awards RAG System

A Retrieval-Augmented Generation (RAG) system for Australian workplace law, built specifically for the challenging aspects of legal document retrieval: heavy cross-referencing, defined terms, and provisions that only make sense when read with multiple others.

## Architecture

```
fairwork-rag/
├── src/
│   ├── data_preprocessing/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py      # PDF parsing with structure preservation
│   │   ├── chunking.py        # Legal-aware chunking with cross-ref tracking
│   │   └── metadata_extractor.py # Definitions, cross-refs, structure
│   ├── embedding/
│   │   └── __init__.py        # OpenAI text-embedding-3-large
│   ├── retrieval/
│   │   └── __init__.py        # Hybrid search (vector + BM25)
│   ├── llm/
│   │   └── __init__.py        # Gemma 4 12B local inference
│   ├── evaluation/
│   │   ├── eval_generator.py  # Test query generation
│   │   └── eval_metrics.py    # Precision/recall metrics
│   ├── pipeline.py            # Main orchestration
│   ├── chat.py                # Interactive chat interface
│   └── evaluate.py            # Evaluation runner
├── data/
│   ├── raw/                   # PDF documents
│   ├── processed/             # Chunks and embeddings
│   └── eval_set/
├── evaluation_set/            # Generated eval data
├── requirements.txt
└── README.md
```

## Key Features

1. **Legal-aware chunking**: Preserves section boundaries and links
2. **Hybrid retrieval**: Combines vector search + BM25 for optimal precision
3. **Cross-reference tracking**: Links related provisions
4. **High-precision evaluation**: Designed for ≥90% precision target

## Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage

### 1. Process Documents

```bash
python src/pipeline.py --mode process --pdfs data/raw/fair_work_act_2009.pdf
```

### 2. Interactive Chat

```bash
python src/chat.py --load-index
```

### 3. Run Evaluation

```bash
python src/evaluate.py --pdfs data/raw/fair_work_act_2009.pdf --num-queries 100
```

This generates:
- 100+ test queries (fact, hypothetical, cross-reference types)
- Relevance annotations
- Precision/recall metrics

## Evaluation Set

The evaluation set is the impressive part - it tests:

1. **Fact-based queries** (40%): Definition lookups, specific section questions
2. **Hypothetical scenarios** (35%): Applied scenarios requiring interpretation
3. **Cross-referencing queries** (25%): Queries requiring chained provisions

### Metrics Reported

- **Precision@k** (k=5,10) - Target ≥90%
- **Recall@k** - Proportion of relevant docs retrieved
- **MRR** (Mean Reciprocal Rank)
- **MAP** (Mean Average Precision)

## The Challenge

Legal documents are hard for RAG because:

1. **Cross-referencing**: "see s 35" requires following links
2. **Defined terms**: "employee means..." requires definition lookup
3. **Interdependence**: Provisions only make sense with others
4. **Precision matters**: Incorrect legal advice is costly

This system is designed to handle these challenges specifically.
