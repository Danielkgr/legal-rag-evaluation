# Run provenance

This directory holds the small, checked-in artefacts from one local run of the pipeline.  The bulky inputs and intermediate files stay out of git, so this file records what was used, so that a reviewer can obtain the same documents and repeat the run.

> [!IMPORTANT]
> The run used the code as it stood before the chunker was rewritten and the code moved into the `legal_rag` package.  The current chunker cuts at different places, so a repeat run will produce a different number of chunks and possibly different results.  The run has not been repeated with the current code.

## Documents indexed

Both are Commonwealth Acts, taken from their official compilations on the Federal Register of Legislation (www.legislation.gov.au), the authoritative publisher of Australian legislation.  The compilation identifiers below locate each document on that register.  Each PDF was rendered locally from the register page on 14 September 2026.  The PDF producer string is a local browser render, not the register's own export, and the cover-page compilation details were read back out of the rendered file to confirm what was indexed.

| Document | Numbered as | Compilation | Compilation date | Amendments to | Registered | Pages |
|---|---|---|---|---|---|---|
| `spam_act_2003.pdf` | Spam Act 2003, No. 129 of 2003 | No. 10 | 10 March 2016 | Act No. 4, 2016 | 9 June 2016 | 38 |
| `dnr_act_2006.pdf` | Do Not Call Register Act 2006, No. 88 of 2006 | No. 16 | 1 September 2021 | Act No. 13, 2021 | 21 September 2021 | 48 |

These two Acts were chosen because they are short, they deal with related subject matter (electronic marketing and telemarketing), and they are distinct, which makes them a fair test of whether the hybrid retriever routes a query to the right Act.

The PDFs are not committed.  `data/raw/` is ignored by git because it holds third-party inputs.  Commonwealth legislation on the register is published under a Creative Commons Attribution licence, and the compilation identifiers above let a reviewer download the same text.

**Chunking result.**  The chunker of that time produced 247 chunks, 104 from the Spam Act and 143 from the Do Not Call Register Act.  Section numbers were stored per chunk in `data/processed/chunks.json`, which is a generated intermediate and is not committed.

## Model servers

The run used two local OpenAI-compatible endpoints, so it needed no OpenAI or other commercial key.

| Role | Server | How the code reached it |
|---|---|---|
| Embeddings | `http://localhost:10001/v1` | The OpenAI client, through `OPENAI_BASE_URL` with a placeholder `OPENAI_API_KEY`.  This is the same code path the README documents for OpenAI `text-embedding-3-large`. |
| Generation | `http://localhost:10009/v1`, a local reasoning model | `OpenAIChatLLM`, through `CHAT_BASE_URL` and `CHAT_MODEL` |

## How the artefacts were produced

Environment variables for every command below:

```bash
export OPENAI_API_KEY=local
export OPENAI_BASE_URL=http://localhost:10001/v1   # embeddings
export CHAT_BASE_URL=http://localhost:10009/v1     # generation
export CHAT_MODEL=<model-name-served-there>
```

Index build, which writes `data/processed/`:

```bash
python src/pipeline.py --mode process \
    --pdfs data/raw/spam_act_2003.pdf data/raw/dnr_act_2006.pdf \
    --output-dir data/processed
```

That is the command as it was run, from the earlier layout in which the modules sat directly under `src/`.  The equivalent command now is `python -m legal_rag.pipeline --mode process --pdfs ... --output-dir data/processed`.

| Artefact | Produced by | What it is |
|---|---|---|
| `results/retrieval_probe.json` | `probe_retrieval.py` | Eight cross-Act questions written by hand before any retrieval ran, each labelled with the Act that answers it.  The script asks only which Act the top hits come from.  It is a document-level routing check on eight points, not a benchmark. |
| `results/chat_demo.json` | No committed script at the time | Two questions sent through the full chat path of retrieval, prompt, and generation.  One answer cites s 11 and s 10 of the Do Not Call Register Act 2006.  The other declines, because the top chunks did not surface the operative provision. |
| `evaluation_set/`, not committed | `python src/evaluate.py ...` | The built-in auto-labelled evaluation.  Its scores are not a retrieval result, for the reasons below. |

`scripts/chat_demo.py` was added later to show how `results/chat_demo.json` is produced.  It writes the same format, but it has not been run against the committed file, so the committed answers came from the earlier code, not from this script.

## Why the built-in evaluation scores are not reported as a result

The generator in `src/legal_rag/evaluation/eval_generator.py` writes 100 queries in three groups, and `tests/test_eval_generator.py` pins down how it labels them.

| Queries | How they are written | What counts as relevant |
|:--:|---|---|
| 40 | Templates filled from sampled chunks, such as "What is the definition of '<frequent word>' in the <Act>?" | Every chunk of the source Act, plus any chunk elsewhere whose section number equals or contains the query's section |
| 60 | Fair Work Act questions written into the code, such as "How much annual leave is an employee entitled to?" | Only chunks of a document named `fair_work_act_2009`, so on this corpus, nothing |

The generator writes one relevance row for every query and chunk, 100 times 247 or 24,700 rows.  The run recorded 5,310 of those rows as relevant.  That is consistent with the code, because only the 40 templated queries can have relevant rows, and each marks the 104 or 143 chunks of its source Act plus a few section-number matches.

Two consequences make the aggregate uninterpretable.

1. A templated query counts every chunk of its Act as relevant, so precision mostly measures whether the top hits came from the right Act, a weaker version of the routing probe.
2. The metric code scores a query with no relevant chunk as recall 1.0 and mean average precision 1.0 by convention.  With 60 such queries, recall@1 cannot fall below 0.60 and precision@1 cannot rise above 0.40, whatever the retriever does.  This run printed recall@1 0.603, MAP 0.619, and precision@1 0.360, close to those bounds.

The tests in `tests/test_eval_metrics.py` pin the convention down so that it stays documented, and the runner now reports how many queries had no relevant chunk.  The interpretable signal from this run is the hand-written routing probe in `results/retrieval_probe.json`.
