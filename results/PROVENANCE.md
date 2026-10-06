# Run provenance

This directory holds the small, checked-in artefacts from one local run of the
pipeline. The bulky inputs and intermediate files stay out of git (see
`.gitignore`), so this file records exactly what was used, so that a reviewer
can re-obtain the documents and re-create the run.

## Documents indexed

Both are Commonwealth Acts, taken from their **official compilations** on the
Federal Register of Legislation (www.legislation.gov.au), the authoritative
publisher of Australian legislation. The compilation identifiers below locate
each document on that register. Each PDF was rendered locally from the register
page on 14 September 2026 (the PDF producer string is a local browser render,
not the register's own export), and the cover-page compilation details were read
back out of the rendered file to confirm what was actually indexed.

| Document | Numbered as | Compilation | Compilation date | Amendments to | Registered | Pages |
|---|---|---|---|---|---|---|
| `spam_act_2003.pdf` | Spam Act 2003, No. 129 of 2003 | No. 10 | 10 March 2016 | Act No. 4, 2016 | 9 June 2016 | 38 |
| `dnr_act_2006.pdf` | Do Not Call Register Act 2006, No. 88 of 2006 | No. 16 | 1 September 2021 | Act No. 13, 2021 | 21 September 2021 | 48 |

These two Acts were chosen deliberately: they are short, cross-reference one
another's subject matter (electronic marketing and telemarketing), and are
genuinely distinct, which makes them a fair test of whether the hybrid retriever
routes a query to the right Act.

The PDFs are **not** committed. `data/raw/` is gitignored because it holds third-party inputs.
Commonwealth legislation is reproduced under the register's standard terms
(Creative Commons Attribution 3.0 Australia); the compilation identifiers above
let a reviewer download the same text.

**Chunking result:** 247 chunks in total, 104 from the Spam Act and 143 from
the Do Not Call Register Act. The legal-aware chunker split on section headings;
section numbers are stored per chunk in `data/processed/chunks.json` (gitignored,
it is a generated intermediate).

## Model servers (local, no commercial keys)

The run used two local OpenAI-compatible endpoints, so it needs no OpenAI or
other commercial key:

- **Embeddings:** an embedding server on `http://localhost:10001/v1`, reached by
  pointing the OpenAI client at it through `OPENAI_BASE_URL` with a dummy
  `OPENAI_API_KEY`. This exercises the same code path the README documents for
  OpenAI `text-embedding-3-large`.
- **Generation:** a chat server on `http://localhost:10009/v1` (a local
  reasoning model), reached through the `OpenAIChatLLM` client via
  `CHAT_BASE_URL` and `CHAT_MODEL`. This is the newer code path added so the
  generation side can target a local server instead of only a HuggingFace
  checkpoint.

## How the artefacts were produced

Environment variables for every command below:

```bash
export OPENAI_API_KEY=local
export OPENAI_BASE_URL=http://localhost:10001/v1   # embeddings
export CHAT_BASE_URL=http://localhost:10009/v1      # generation
export CHAT_MODEL=<model-name-served-there>
```

Index build (writes `data/processed/`, gitignored):

```bash
python src/pipeline.py --mode process \
    --pdfs data/raw/spam_act_2003.pdf data/raw/dnr_act_2006.pdf \
    --output-dir data/processed
```

That is the command as it was run, from the earlier layout in which the
modules sat directly under `src/`.  The code now lives in the `legal_rag`
package, and the equivalent command is
`python -m legal_rag.pipeline --mode process --pdfs ... --output-dir data/processed`.

- `probe_retrieval.py` -> `results/retrieval_probe.json`: eight cross-Act
  queries chosen by hand **before** any retrieval, each labelled with the Act
  that answers it. The script only asks which Act the top hits come from; it is
  a document-level routing check on eight points, not a benchmark.
- `results/chat_demo.json`: two questions sent through the full RAG chat path
  (retrieve, build a grounded prompt, generate). One answer is correctly cited
  to Do Not Call Register Act 2006 s 11 and s 10; the other declined to answer
  because the top chunks did not surface the operative provision, which the
  guardrail reports honestly rather than inventing a citation.
- `python src/evaluate.py ...` -> `evaluation_set/` (gitignored): the built-in
  auto-annotated evaluation. See the caveats below; its scores are not a
  retrieval result.

## Why the built-in evaluation scores are not reported as a result

The bundled generator creates the test queries **from** the indexed chunks and
then derives the relevance labels from the same chunks by shallow lexical
overlap. Across this run it wrote the full query-by-chunk matrix (100 queries
against 247 chunks, 24,700 pairs) and marked 5,310 of those pairs relevant, so
queries carry a large and uneven "relevant" set, and some carry none. Two
consequences make the aggregate uninterpretable:

1. The queries are paraphrases of the very chunks they are scored against, so
   retrieval is being graded against labels derived from the same overlap the
   retriever rewards; it measures the generator, not real-world retrieval.
2. The metric code scores a query with no relevant chunk as recall 1.0 and mean
   average precision 1.0 by convention. Those defaults inflate both numbers
   toward optimistic values (this run printed recall@1 0.603 and MAP 0.619, with
   precision@1 0.360), and the inflation is not meaningful.

The three tests in `tests/test_eval_metrics.py` pin this convention down so it
stays documented rather than surprising. The honest, interpretable signal from
this run is the hand-picked routing probe in `results/retrieval_probe.json`.
