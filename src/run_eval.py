"""Part 3: run 5 test queries on both systems, time each, score retrieval relevance."""
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from custom_rag import HealthcareRAG
from langchain_rag import LangChainRAG
from judge import score_relevance

ROOT = Path(__file__).resolve().parent.parent
DATA = [str(ROOT / "data" / "diabetes.pdf"), str(ROOT / "data" / "standards.pdf")]
RESULTS_DIR = ROOT / "results"

QUERIES = [
    "What are metformin side effects?",
    "A1C target range for type 2 diabetes?",
    "How to treat hypoglycemia?",
    "When to check blood glucose?",
    "Foot care recommendations for diabetics?",
]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=== Building Custom RAG ===")
    custom_rag = HealthcareRAG()
    custom_rag.add_documents(DATA)

    print("=== Building LangChain RAG ===")
    lc_rag = LangChainRAG(DATA)

    records = []
    for q in QUERIES:
        print(f"\nTesting: {q}")

        t0 = time.time()
        custom_answer, custom_chunks = custom_rag.generate(q)
        custom_time = time.time() - t0
        custom_score, custom_reason = score_relevance(q, custom_chunks)

        t0 = time.time()
        lc_answer, lc_chunks = lc_rag.generate(q)
        lc_time = time.time() - t0
        lc_score, lc_reason = score_relevance(q, lc_chunks)

        print(f"  Custom:    {custom_time:.2f}s, relevance {custom_score}/5")
        print(f"  LangChain: {lc_time:.2f}s, relevance {lc_score}/5")

        records.append(
            {
                "query": q,
                "custom": {
                    "time_sec": round(custom_time, 3),
                    "score": custom_score,
                    "score_reason": custom_reason,
                    "answer": custom_answer,
                    "retrieved_chunks": custom_chunks,
                },
                "langchain": {
                    "time_sec": round(lc_time, 3),
                    "score": lc_score,
                    "score_reason": lc_reason,
                    "answer": lc_answer,
                    "retrieved_chunks": lc_chunks,
                },
            }
        )

    with open(RESULTS_DIR / "eval_results.json", "w") as f:
        json.dump(records, f, indent=2)

    lines = [
        "| # | Test Query | Custom Time | Custom Score | LangChain Time | LangChain Score |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(records, 1):
        lines.append(
            f"| {i} | {r['query']} | {r['custom']['time_sec']}s | {r['custom']['score']}/5 "
            f"| {r['langchain']['time_sec']}s | {r['langchain']['score']}/5 |"
        )
    avg_custom_time = sum(r["custom"]["time_sec"] for r in records) / len(records)
    avg_lc_time = sum(r["langchain"]["time_sec"] for r in records) / len(records)
    avg_custom_score = sum(r["custom"]["score"] for r in records) / len(records)
    avg_lc_score = sum(r["langchain"]["score"] for r in records) / len(records)
    lines.append(
        f"| **Avg** | | **{avg_custom_time:.2f}s** | **{avg_custom_score:.1f}/5** "
        f"| **{avg_lc_time:.2f}s** | **{avg_lc_score:.1f}/5** |"
    )

    table = "\n".join(lines)
    (RESULTS_DIR / "scorecard.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
