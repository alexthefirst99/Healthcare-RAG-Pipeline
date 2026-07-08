# Build Healthcare RAG Two Ways (Custom vs LangChain)

**Name:** Alex Tran

## Data

- **diabetes.pdf** — CDC "4 Steps to Manage Your Diabetes for Life" (National Diabetes Education
  Program patient booklet), 20 pages.
- **standards.pdf** — "2025 ADA Standards of Medical Care in Diabetes: Updates!", 11 pages, with
  tables covering glycemic targets, DKA/HHS management, and neuropathy screening/treatment.

Both provided as course materials for this assignment.

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
  spec) → 82 chunks.
- LangChain: `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`, which splits on
  paragraph/sentence/word boundaries before falling back to a hard character cut → 100 chunks (more,
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
| 1 | What are metformin side effects? | 0.97s | 3/5 | 0.76s | 3/5 |
| 2 | A1C target range for type 2 diabetes? | 1.27s | 5/5 | 1.07s | 3/5 |
| 3 | How to treat hypoglycemia? | 0.75s | 3/5 | 0.78s | 3/5 |
| 4 | When to check blood glucose? | 1.04s | 5/5 | 1.05s | 5/5 |
| 5 | Foot care recommendations for diabetics? | 0.91s | 2/5 | 1.54s | 5/5 |
| **Avg** | | **0.99s** | **3.6/5** | **1.04s** | **3.8/5** |

Relevance is scored 1-5 by hand (Alex Tran), reading the actual retrieved chunks and the generated
answer for each of the 10 runs:

- **Q1:** Both systems failed to retrieve side effects. The refusal is honest, so not terrible, but
  it does not answer the user's question. 3/5 for both.
- **Q2:** Custom retrieved the correct A1C target range and answered with the number. 5/5. LangChain
  did not retrieve the real target passage, and the answer was vague. 3/5.
- **Q3:** Both failed to retrieve actual hypoglycemia treatment steps. The refusal is safe, but not
  useful. 3/5 for both.
- **Q4:** Both retrieved the exact correct card and answered directly. 5/5 for both.
- **Q5:** Custom failed retrieval and gave no foot-care content, while LangChain retrieved and
  answered the actual protocol. Custom gets 2/5, LangChain gets 5/5.

(An earlier pass used an LLM-as-judge for this scoring — see git history — but that was replaced with
manual scoring since the assignment calls for evaluating retrieval relevance yourself, not delegating
it to another model.)

## Part 4: Reflection

**1. Performance.** LangChain edged ahead overall, 3.8 vs. 3.6 — but the average hides the more
interesting result: the two systems don't have a "better" and "worse" version, they have different
strengths that showed up on different questions. On the A1C question, Custom found the exact target
range (7-7.5%) while LangChain gave a vague, disconnected answer ("below 7," no specific range) —
Custom wins clearly, 5 vs. 3. On the foot-care question, the opposite happened: LangChain's chunker
retrieved the ADA bulletin's actual neuropathy protocol (`"Assess for PAD via foot screening...
NEUROPATHY TREATMENT to include: a. Glycemic, lipid, BP and weight control b. Meds: gabapentinoids,
SNRIs..."`) and answered correctly, while Custom's chunker missed that passage entirely, retrieved
unrelated CDC-booklet content about emotional support and meal planning, and correctly refused —
LangChain wins clearly, 5 vs. 2. LangChain's higher average is only because that second gap (3
points) was larger than the first gap (2 points), not because it's the stronger system overall.
**The lesson: an average hides which system is actually better at what — two systems can trade wins
on different questions and still land close together in the final number, telling you almost nothing
about where each one's real strengths are.**

**2. Code.** Custom is 58 lines, LangChain is 49 (both include real LLM generation, unlike the
assignment's stub — which is why neither matches the assignment's 48-vs-12 estimate; the 12-line
version never actually calls a model). The honest gap shows up in *what breaks*, not line count:
LangChain's old convenience layer (`RetrievalQA`, one call) is exactly what got removed in the 1.x
rewrite, so today the LangChain version requires nearly as much manual composition as the custom one.
Use custom code for a small, fixed pipeline you fully control and don't want to re-learn on every
major version bump. Use LangChain when you need its ecosystem (many loader/vectorstore/retriever
backends swappable behind one interface) and are willing to track its API churn.

**3. Healthcare / chunking.** Custom's fixed-width chunking visibly cut medical terms mid-word: one
retrieved chunk for the A1C query starts `"erwise healthy with few coexisting chronic conditions can
have adult glycemic g..."` — the 500-char boundary landed inside "otherwise" and again mid-word at
"goals," with no repair. LangChain's `RecursiveCharacterTextSplitter` avoids this by stopping at
sentence/word boundaries. Neither system, however, preserved the ADA bulletin's structured tables
(glycemic targets, DKA/HHS thresholds) as tables — PDF text extraction (`pypdf` and `PyPDFLoader`
alike) already flattens them to whitespace-separated text before either chunker sees it, so a precise
clinical threshold table becomes a run-on wall of numbers in both systems. That's a fixable problem,
but it lives upstream of chunking, in PDF extraction, not in either RAG implementation.

**4. Understanding.** Writing the custom version forced explicit decisions LangChain's abstraction
hides by default: how `IndexFlatIP` + `normalize_L2` implements cosine similarity, that "add
documents" means "embed, then build a flat index" with no incremental-update story, and that
`retrieve()`+`generate()` are two separate calls you have to wire together yourself (LangChain's old
`RetrievalQA` did this invisibly, in one method call — and its removal in 1.x actually confirms this
was doing real, non-trivial work under the hood, not just syntactic sugar).

**5. Production.** For a hospital deployment I would: (a) chunk on document structure (headings,
tables) rather than a fixed character count, since the flattened-table finding above shows a naive
splitter can turn a precise clinical threshold table into unreliable prose; (b) never trust one
overall average by itself — Custom and LangChain each won big on a different question (Q2 vs. Q5)
and roughly cancelled out in the average, so a production eval needs to track performance per
question type, not just one combined number; (c) add a refusal/citation requirement in the generation prompt with
a link back to the source page, since Q1 and Q3 show the model correctly says "not in context" when it
should — that behavior needs to be enforced and audited, not just hoped for; (d) replace the FAISS
`IndexFlatIP` with an ANN index (HNSW/IVF) and a real vector DB (e.g. pgvector, Pinecone) once the
corpus is more than a couple of PDFs, since flat search is O(n) per query; (e) pin the LangChain
version explicitly, given how much of this assignment's own sample code broke on a routine
`pip install` six months later.

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
