# Build Healthcare RAG Two Ways (Custom vs LangChain)

**Name:** Alex Tran

## Data

- **diabetes.pdf** — CDC/NIDDK ["4 Steps to Manage Your Diabetes for Life"](https://www.niddk.nih.gov/-/media/Files/Health-Information/Health-Professionals/Diabetes/health-care-professionals/4StepsToManageDiabetes-English_508.pdf) (CDC National Diabetes Education Program patient booklet)
- **standards.pdf** — [ADA 2025 Standards of Medical Care in Diabetes, Abridged for Primary Care](https://www.novonordiskmedical.com/content/dam/medical/novonordiskmedical/ta/diabetes/disease-education/resource-documents/abridged-ada-2025-standards-of-medical-care-in-diabetes.pdf) (22 pages)

The complete ADA Standards of Care is split across ~20 separate paywalled journal-chapter URLs on
`diabetesjournals.org` rather than distributed as one PDF, so this uses the abridged primary-care
version ADA itself points clinicians to for a single-document, single-download version of the same
content.

## Setup

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY
python src/run_eval.py
```

**Why Python 3.11, not the system default:** this Mac's default `python3` is 3.14, which is too new
to have guaranteed prebuilt wheels for `faiss-cpu`/`sentence-transformers` yet. Used
`python3.11` (via miniconda) for a known-compatible dependency chain.

**Why `pypdf` instead of `PyPDF2`:** `PyPDF2` is unmaintained (merged into `pypdf`, its actively
maintained successor, same `PdfReader` API) so this uses `pypdf` for both implementations.

## Architecture

**Embeddings:** `all-MiniLM-L6-v2` (via `sentence-transformers` directly in the custom system,
via `langchain-huggingface`'s wrapper in the LangChain system) — held constant across both so the
comparison isolates chunking/framework differences, not embedding-model differences.

**Generation:** `gpt-4o-mini`, temperature 0, for both. The assignment's starter code doesn't
actually call an LLM — custom's `generate()` just slices the raw context string, and LangChain's
`qa_chain` is stubbed with `llm="mock_llm"`. Both were wired to a real `gpt-4o-mini` call with an
identical system prompt ("answer from context only, say so if the context doesn't contain the
answer") so the answers and relevance scores below are measuring a real generation step, not a
placeholder.

**Vector store:** FAISS (`IndexFlatIP` cosine-similarity search) in both — raw `faiss` in the
custom system, `langchain_community.vectorstores.FAISS` in the LangChain system.

**Chunking:**
- Custom: fixed 500-character windows, zero overlap, no structural awareness (per the assignment
  spec) → 123 chunks.
- LangChain: `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`, which splits on
  paragraph/sentence/word boundaries before falling back to a hard character cut → 158 chunks (more,
  smaller chunks, since it stops early at a boundary rather than always filling to 500 chars).

**LangChain version note:** the assignment's sample code (`from langchain.chains import
RetrievalQA`, `langchain.document_loaders`, `langchain.embeddings`, `langchain.vectorstores`)
targets LangChain ~0.1. A fresh `pip install langchain` today pulls **1.3.11**, which removed
`langchain.chains` (and `RetrievalQA`) entirely in favor of LCEL-composed runnables. `src/langchain_rag.py`
is written against the current 1.x API (`ChatPromptTemplate | llm | StrOutputParser()` composed
manually) since that's what actually installs today — this is itself a data point for the
reflection below.

## Part 3: Test Results (K=3)

| # | Test Query | Custom Time | Custom Score | LangChain Time | LangChain Score |
|---|---|---|---|---|---|
| 1 | What are metformin side effects? | 2.39s | 1/5 | 1.19s | 1/5 |
| 2 | A1C target range for type 2 diabetes? | 1.52s | 3/5 | 0.82s | 4/5 |
| 3 | How to treat hypoglycemia? | 1.24s | 3/5 | 0.91s | 3/5 |
| 4 | When to check blood glucose? | 1.37s | 5/5 | 0.89s | 5/5 |
| 5 | Foot care recommendations for diabetics? | 1.08s | 3/5 | 1.05s | 3/5 |
| **Avg** | | **1.52s** | **3.0/5** | **0.97s** | **3.2/5** |

Relevance is scored 1-5 by an LLM judge (`gpt-4o-mini`, `src/judge.py`) rating how well the
retrieved chunks address the query, since no human grader was available — full reasoning per query
is in `results/eval_results.json`.

## Part 4: Reflection

**1. Performance.** Nearly tied (3.0/5 custom vs. 3.2/5 LangChain), and the one query where they
diverged (Q2, A1C target — 3 vs. 4) is the clearest signal: LangChain's `RecursiveCharacterTextSplitter`
stops at paragraph boundaries, so it retrieved a cleaner, more self-contained passage, while custom's
fixed 500-char window landed mid-sentence more often. But the two lowest scores (Q1 metformin, Q5 foot
care — 1/5 and 3/5 on *both* systems identically) aren't a retrieval failure at all: neither source PDF
actually discusses metformin side effects or a dedicated foot-care protocol in any depth. Since both
systems use the same embeddings and near-identical top-k cosine search, chunking boundary quality was
the only real variable — and it produced a small, not large, edge for LangChain here.

**2. Code.** Custom is 58 lines, LangChain is 49 (both now include real LLM generation, unlike the
assignment's stub, which is why neither matches the assignment's 48-vs-12 estimate — the "12 lines"
version never actually calls a model). The honest gap shows up in *what breaks*, not line count:
LangChain's convenience layer (`RetrievalQA`, one call) is exactly what got removed in the 1.x rewrite,
so today the LangChain version requires nearly as much manual composition as the custom one. Use
custom code for a small, fixed pipeline you fully control and don't want to re-learn on every major
version bump. Use LangChain when you need its ecosystem (many loader/vectorstore/retriever backends
swappable behind one interface) and are willing to track its API churn.

**3. Healthcare / chunking.** Custom's fixed-width chunking visibly cut mid-word: the retrieved chunk
for "when to check blood glucose" starts `"hree sections..."` — the fixed 500-char boundary landed
inside "Three sections," silently dropping the first letter with no repair. LangChain's chunk for the
same query starts cleanly at `"15\nSelf Checks of Blood Sugar"`. Neither, however, preserved the ADA
standards' DKA/HHS glucose-target table as a structured table — PDF text extraction (`pypdf` and
`PyPDFLoader` alike) already flattens tables to whitespace-separated text before either chunker sees
it, so both retrieved it as a wall of run-on numbers (`"Keep glucose between 150 and 200 mg/dL... K+
>5.0 mmol/L Start insulin..."`) that a clinician would find hard to parse — a fixable problem, but
one that lives upstream of chunking, in PDF extraction.

**4. Understanding.** Writing the custom version forced explicit decisions LangChain's abstraction
hides by default: how `IndexFlatIP` + `normalize_L2` implements cosine similarity, that "add
documents" means "embed, then build a flat index" with no incremental-update story, and that
`retrieve()`+`generate()` are two separate calls you have to wire together yourself (LangChain's old
`RetrievalQA` did this invisibly, in one method call — and its removal in 1.x actually confirms this
was doing real, non-trivial work under the hood, not just syntactic sugar).

**5. Production.** For a hospital deployment I would: (a) chunk on document structure (headings,
tables) rather than a fixed character count, since the DKA/HHS table finding above shows a naive
splitter can turn a precise clinical threshold table into unreliable prose; (b) add a refusal/citation
requirement in the generation prompt with a link back to the source page, since Q1 and Q5 show the
model correctly says "not in context" when it should — that behavior needs to be enforced and audited,
not just hoped for; (c) replace the FAISS `IndexFlatIP` with an ANN index (HNSW/IVF) and a real vector
DB (e.g. pgvector, Pinecone) once the corpus is more than a couple of PDFs, since flat search is O(n)
per query; (d) pin the LangChain version explicitly, given how much of this assignment's own sample
code broke on a routine `pip install` six months later.

## Repo layout

```
src/
  custom_rag.py      # Part 1: hand-rolled chunking + FAISS + gpt-4o-mini
  langchain_rag.py    # Part 2: LangChain 1.x LCEL pipeline
  judge.py             # LLM-as-judge relevance scoring (1-5)
  run_eval.py           # Part 3: runs both systems over all 5 queries, times + scores each
data/                    # diabetes.pdf, standards.pdf (downloaded, see Data section above)
results/
  eval_results.json      # full per-query answers, chunks, scores, judge reasoning
  scorecard.md            # the Part 3 table
```
