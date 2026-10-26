"""Controlled retrieval experiment: compare chunking strategies and top-k.

Runs retrieval only (local embeddings + in-memory ChromaDB), so it needs no
API key and costs nothing. The golden dataset is unchanged between
configurations; only the retrieval settings vary. The persistent store in
.chroma/ is not touched.

Usage: python -m scripts.retrieval_experiment
Writes reports/retrieval_experiment.json and prints a table.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from app import config
from app.dataset import dataset_fingerprint, load_and_validate
from app.ingestion import ingest_knowledge_base
from app.metrics import mean, retrieval_metrics
from app.rag_pipeline import retrieve
from app.vector_store import make_in_memory_store

CONFIGURATIONS = [
    {"strategy": "character", "chunk_size": 500, "overlap": 50, "top_k": k} for k in (1, 2, 3)
] + [
    {"strategy": "paragraph", "chunk_size": 500, "overlap": 0, "top_k": k} for k in (1, 2, 3, 5)
]


def run(configurations=CONFIGURATIONS, cases=None) -> dict:
    cases = cases if cases is not None else load_and_validate()
    labeled = [c for c in cases if c.relevant_sources]
    results = []
    for cfg in configurations:
        store = make_in_memory_store()
        stats = ingest_knowledge_base(store, strategy=cfg["strategy"], chunk_size=cfg["chunk_size"], overlap=cfg["overlap"])
        rows, misses = [], []
        for case in labeled:
            retrieved = retrieve(case.question, top_k=cfg["top_k"], store=store)
            metrics = retrieval_metrics(retrieved.sources, case.relevant_sources)
            rows.append(metrics)
            if metrics["missing_sources"]:
                misses.append({"id": case.id, "missing": metrics["missing_sources"], "retrieved": retrieved.sources})
        results.append(
            {
                **cfg,
                "is_baseline": cfg["strategy"] == config.CHUNK_STRATEGY
                and cfg["chunk_size"] == config.CHUNK_SIZE
                and cfg["top_k"] == config.TOP_K,
                "chunks_in_store": stats["chunks"],
                "n_cases": len(rows),
                "hit_rate": mean(r["hit"] for r in rows),
                "precision": mean(r["precision"] for r in rows),
                "recall": mean(r["recall"] for r in rows),
                "mrr": mean(r["reciprocal_rank"] for r in rows),
                "misses": misses,
            }
        )
        store.client.delete_collection(store.collection_name)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "dataset_fingerprint": dataset_fingerprint(),
        "note": "Source-level metrics over cases with relevance labels; unanswerable cases excluded.",
        "results": results,
    }


def main() -> int:
    report = run()
    out = config.REPORTS_DIR / "retrieval_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'strategy':<10} {'k':>2} {'chunks':>6} {'hit':>6} {'prec':>6} {'recall':>6} {'mrr':>6}")
    for r in report["results"]:
        mark = "  <- current config" if r["is_baseline"] else ""
        print(
            f"{r['strategy']:<10} {r['top_k']:>2} {r['chunks_in_store']:>6} {r['hit_rate']:>6.3f} "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['mrr']:>6.3f}{mark}"
        )
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
